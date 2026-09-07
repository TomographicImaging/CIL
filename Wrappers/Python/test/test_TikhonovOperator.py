#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest
import warnings
from contextlib import contextmanager

import numpy

from cil.framework import (AcquisitionGeometry, BlockGeometry, DataContainer,
                           ImageGeometry, VectorGeometry)
from cil.optimisation.algorithms import CGLS, LSQR
from cil.optimisation.operators import (BlockOperator, DiagonalOperator,
                                        GradientOperator, IdentityOperator,
                                        LinearOperator, MatrixOperator,
                                        WaveletOperator)
from cil.optimisation.operators.TikhonovOperator import (
    BlockTikhonovOperator, TikhonovOperator, WeightedStructOperator,
    create_tikhonov_operator, resolve_form)
from cil.optimisation.operators.TikhonovOperator import (
    _geometry_size as _size)
from cil.optimisation.operators.TikhonovOperator import (
    _is_standard_form_capable)

from testclass import CCPiTestClass
from utils import initialise_tests

initialise_tests()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

@contextmanager
def count_allocated_elements():
    """
    Count the CIL containers allocated inside the block.

    Only the *leaf* geometries are patched. ``BlockGeometry.allocate``
    delegates to its children and allocates nothing itself, so leaf counting is exact and needs no special case for block
    ranges.

    ``DataContainer.clone`` is patched too, because ``.copy()`` is an alias for
    it and neither touches ``geometry.allocate``. That is how a
    ``self.d = self.v.copy()`` would otherwise slip past the counter.

    This measures CIL container allocations, which is what the memory budget in
    ``pullrequest-memory.md`` is written in. It does not see numpy temporaries
    created inside a third-party call such as ``pywt``.
    """
    tally = {'elements': 0, 'containers': 0}
    saved = [(cls, 'allocate', cls.allocate)
             for cls in (ImageGeometry, VectorGeometry, AcquisitionGeometry)]
    saved.append((DataContainer, 'clone', DataContainer.clone))

    def wrap(original):
        def counted(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            tally['elements'] += int(numpy.prod(self.shape))
            tally['containers'] += 1
            return result
        return counted

    for cls, name, original in saved:
        setattr(cls, name, wrap(original))
    try:
        yield tally
    finally:
        for cls, name, original in saved:
            setattr(cls, name, original)


class StubStructOperator(LinearOperator):
    """
    An identity that reports whatever orthogonality it is constructed with.

    Deliberately *not* a subclass of ``IdentityOperator``, so that the
    ``struct_is_identity`` shortcut does not fire and the dispatch is decided
    purely by ``is_orthogonal()``.
    """

    def __init__(self, geometry, orthogonal):
        super(StubStructOperator, self).__init__(domain_geometry=geometry,
                                                 range_geometry=geometry)
        self._orthogonal = orthogonal

    def is_orthogonal(self):
        return self._orthogonal

    def direct(self, x, out=None):
        if out is None:
            return x.copy()
        out.fill(x)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out=out)


def positive_weights(operator, seed):
    """A strictly positive weight container in Range(L), like IRLS produces."""
    weights = operator.range_geometry().allocate('random', seed=seed)
    return weights.abs() + 0.5


def leaves_of(container):
    """
    Flatten a container to its leaves, at whatever depth.

    A ``BlockDataContainer`` nested inside another has no ``as_array``, so the
    nested tests cannot compare level by level the way the flat ones do.
    """
    if hasattr(container, 'containers'):
        return [leaf for child in container.containers
                for leaf in leaves_of(child)]
    return [container]


# --------------------------------------------------------------------------- #
# the 'auto' rule on its own
# --------------------------------------------------------------------------- #

class TestResolveForm(CCPiTestClass):
    """
    ``resolve_form`` is the rule the factory applies, exposed so it can be
    asked without building an operator. These are the same cases as
    :class:`TestTikhonovFactory` under ``form='auto'``, but stated against the
    rule rather than against what came out of it -- there is no geometry here
    at all, which is the point.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.identity = IdentityOperator(self.ig)
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        # Orthogonal but not square, so L^* is only a left inverse.
        self.padded = WaveletOperator(self.ig, wname='db4', level=1)

    @property
    def invertible(self):
        return (None, self.identity, self.haar)

    def test_a_warm_starting_caller_always_gets_the_standard_form(self):
        """CGLS: the cheaper form is correct from anywhere, IRLS included."""
        for L in self.invertible:
            for weighted in (False, True):
                for zero_initial in (False, True):
                    with self.subTest(L=type(L).__name__, weighted=weighted,
                                      zero_initial=zero_initial):
                        self.assertEqual('standard', resolve_form(
                            'auto', CGLS, struct_operator=L,
                            weighted=weighted, zero_initial=zero_initial))

    def test_a_step_damping_caller_gets_it_only_from_a_cold_zero_start(self):
        """LSQR: damping the step is the right problem only at x_0 = 0."""
        cases = {(False, True): 'standard',   # one solve, from zero
                 (False, False): 'block',     # one solve, warm started
                 (True, True): 'block',       # IRLS, whatever `initial` says
                 (True, False): 'block'}
        for L in self.invertible:
            for (weighted, zero_initial), expected in cases.items():
                with self.subTest(L=type(L).__name__, weighted=weighted,
                                  zero_initial=zero_initial):
                    self.assertEqual(expected, resolve_form(
                        'auto', LSQR, struct_operator=L, weighted=weighted,
                        zero_initial=zero_initial))

    def test_no_caller_can_have_a_form_that_has_no_inverse(self):
        """The first question comes first: (WL)^-1 has to exist at all."""
        for L in (self.grad, self.padded):
            for solver in (LSQR, CGLS):
                with self.subTest(L=type(L).__name__,
                                  solver=solver.__name__):
                    self.assertEqual('block', resolve_form(
                        'auto', solver, struct_operator=L))

    def test_a_named_form_is_returned_unchanged(self):
        """Only 'auto' is decided here; anything else was already decided."""
        for form in ('standard', 'block'):
            for L in (None, self.grad):
                with self.subTest(form=form, L=type(L).__name__):
                    self.assertEqual(form, resolve_form(
                        form, LSQR, struct_operator=L, weighted=True,
                        zero_initial=False))

    def test_it_never_returns_auto(self):
        self.assertIn(resolve_form(), ('standard', 'block'))

    def test_an_unknown_form_raises(self):
        with self.assertRaises(ValueError):
            resolve_form('blocked')

    def test_the_solvers_carry_the_answer_to_the_second_question(self):
        """
        ``warm_starts_in_standard_form`` is what ``resolve_form`` reads off
        the ``solver`` it is given, and it is a class attribute precisely so
        that the rule can be asked before a solver exists.
        """
        self.assertFalse(LSQR.warm_starts_in_standard_form)
        self.assertTrue(CGLS.warm_starts_in_standard_form)

    def test_the_rule_agrees_with_the_solvers_that_apply_it(self):
        """
        The rule and the factory are the same rule: what ``resolve_form`` says
        is what a solver built the same way ends up with.
        """
        A = IdentityOperator(self.ig)
        data = self.ig.allocate('random', seed=7)
        for solver_class in (LSQR, CGLS):
            for L in (None, self.haar, self.grad):
                for weighted in (False, True):
                    with self.subTest(solver=solver_class.__name__,
                                      L=type(L).__name__, weighted=weighted):
                        expected = resolve_form(
                            'auto', solver_class, struct_operator=L,
                            weighted=weighted, zero_initial=True)
                        solver = solver_class(
                            initial=self.ig.allocate(0), operator=A, data=data,
                            alpha=0.7, struct_operator=L, weighted=weighted)
                        self.assertEqual(expected, solver.form)


# --------------------------------------------------------------------------- #
# the factory
# --------------------------------------------------------------------------- #

class TestTikhonovFactory(CCPiTestClass):

    def setUp(self):
        numpy.random.seed(11)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=12))
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.identity = IdentityOperator(self.ig)

    # -- alpha == 0 -------------------------------------------------------- #

    def test_alpha_zero_returns_the_bare_operator(self):
        for form in ('auto', 'standard', 'block'):
            result = create_tikhonov_operator(self.A, self.ig, regalpha=0.0,
                                              form=form)
            self.assertIs(result, self.A)

    def test_alpha_zero_ignores_a_struct_operator(self):
        result = create_tikhonov_operator(self.A, self.ig,
                                          struct_operator=self.grad,
                                          regalpha=0.0)
        self.assertIs(result, self.A)

    def test_unwrapped_operator_has_no_form_attribute(self):
        result = create_tikhonov_operator(self.A, self.ig, regalpha=0.0)
        self.assertEqual(getattr(result, 'form', 'none'), 'none')

    # -- form='auto' ------------------------------------------------------- #

    def test_auto_with_no_struct_operator_gives_standard(self):
        op = create_tikhonov_operator(self.A, self.ig, regalpha=0.7)
        self.assertIsInstance(op, TikhonovOperator)
        self.assertEqual(op.form, 'standard')

    def test_auto_with_identity_gives_standard(self):
        op = create_tikhonov_operator(self.A, self.ig,
                                      struct_operator=self.identity,
                                      regalpha=0.7)
        self.assertIsInstance(op, TikhonovOperator)

    def test_auto_with_orthogonal_wavelet_gives_standard(self):
        self.assertTrue(self.haar.is_orthogonal())
        op = create_tikhonov_operator(self.A, self.ig,
                                      struct_operator=self.haar, regalpha=0.7)
        self.assertIsInstance(op, TikhonovOperator)

    def test_auto_with_gradient_gives_block(self):
        self.assertFalse(self.grad.is_orthogonal())
        op = create_tikhonov_operator(self.A, self.ig,
                                      struct_operator=self.grad, regalpha=0.7)
        self.assertIsInstance(op, BlockTikhonovOperator)
        self.assertEqual(op.form, 'block')

    def test_auto_follows_is_orthogonal_not_the_concrete_class(self):
        """
        Dispatch is pinned to the declared capability, so a future orthogonal
        operator gets the standard form without touching the factory.
        """
        orthogonal = StubStructOperator(self.ig, orthogonal=True)
        other = StubStructOperator(self.ig, orthogonal=False)

        self.assertIsInstance(
            create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=orthogonal, regalpha=0.7),
            TikhonovOperator)
        self.assertIsInstance(
            create_tikhonov_operator(self.A, self.ig, struct_operator=other,
                                     regalpha=0.7),
            BlockTikhonovOperator)

    # -- explicit forms ---------------------------------------------------- #

    def test_explicit_block_is_always_block(self):
        for L in (None, self.identity, self.haar, self.grad):
            op = create_tikhonov_operator(self.A, self.ig, struct_operator=L,
                                          regalpha=0.7, form='block')
            self.assertIsInstance(op, BlockTikhonovOperator)

    def test_explicit_standard_with_gradient_raises_and_names_block(self):
        with self.assertRaises(ValueError) as ctx:
            create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=self.grad, regalpha=0.7,
                                     form='standard')
        self.assertIn("form='block'", str(ctx.exception))

    def test_unknown_form_raises_and_lists_the_valid_values(self):
        with self.assertRaises(ValueError) as ctx:
            create_tikhonov_operator(self.A, self.ig, regalpha=0.7,
                                     form='nonsense')
        for valid in ('auto', 'standard', 'block'):
            self.assertIn(valid, str(ctx.exception))

    def test_negative_alpha_raises(self):
        with self.assertRaises(ValueError):
            create_tikhonov_operator(self.A, self.ig, regalpha=-1.0)

    def test_solution_geometry_defaults_to_the_operator_domain(self):
        op = create_tikhonov_operator(self.A, regalpha=0.7, form='block')
        self.assertEqual(op.domain_geometry(), self.ig)

    # -- form='auto' and the caller ---------------------------------------- #

    def test_a_warm_starting_caller_keeps_the_standard_form(self):
        """
        `solver=CGLS` subtracts the alpha^2 x term
        itself and so runs on the true regularised normal equations from any
        starting point. Nothing about the caller can take the cheaper form away
        from it -- IRLS included.
        """
        for L in (None, self.identity, self.haar):
            for weighted in (False, True):
                for zero_initial in (False, True):
                    with self.subTest(L=type(L).__name__, weighted=weighted,
                                      zero_initial=zero_initial):
                        K = create_tikhonov_operator(
                            self.A, self.ig, struct_operator=L, regalpha=0.7,
                            weighted=weighted, solver=CGLS,
                            zero_initial=zero_initial)
                        self.assertIsInstance(K, TikhonovOperator)

    def test_a_step_damping_caller_keeps_it_only_from_zero(self):
        """
        `solver=LSQR` damps the step rather than
        the solution. From x_0 = 0 the two coincide, so the standard form is
        still the right problem; from anywhere else it is not.
        """
        for L in (None, self.identity, self.haar):
            with self.subTest(L=type(L).__name__):
                from_zero = create_tikhonov_operator(
                    self.A, self.ig, struct_operator=L, regalpha=0.7,
                    solver=LSQR, zero_initial=True)
                self.assertIsInstance(from_zero, TikhonovOperator)

                warm = create_tikhonov_operator(
                    self.A, self.ig, struct_operator=L, regalpha=0.7,
                    solver=LSQR, zero_initial=False)
                self.assertIsInstance(warm, BlockTikhonovOperator)

    def test_weighted_rules_out_the_standard_form_for_a_step_damping_caller(self):
        """
        `weighted=True` says IRLS, which restarts each inner solve from the
        previous outer iterate. A zero `initial` is then only where the first
        inner solve begins, so it is not enough.
        """
        for L in (None, self.identity, self.haar):
            with self.subTest(L=type(L).__name__):
                K = create_tikhonov_operator(
                    self.A, self.ig, struct_operator=L, regalpha=0.7,
                    weighted=True, solver=LSQR, zero_initial=True)
                self.assertIsInstance(K, BlockTikhonovOperator)

    def test_no_caller_can_ask_for_a_form_that_has_no_inverse(self):
        """The first question is still whether (WL)^-1 exists at all."""
        for weighted in (False, True):
            for solver in (LSQR, CGLS):
                with self.subTest(weighted=weighted, solver=solver.__name__):
                    K = create_tikhonov_operator(
                        self.A, self.ig, struct_operator=self.grad,
                        regalpha=0.7, weighted=weighted, solver=solver)
                    self.assertIsInstance(K, BlockTikhonovOperator)

    def test_explicit_standard_still_overrides_the_caller(self):
        """The rules live in 'auto'; an explicit request is still honoured."""
        K = create_tikhonov_operator(self.A, self.ig, regalpha=0.7,
                                     form='standard', weighted=True,
                                     solver=LSQR, zero_initial=False)
        self.assertIsInstance(K, TikhonovOperator)

    def test_explicit_block_still_overrides_the_caller(self):
        K = create_tikhonov_operator(self.A, self.ig, regalpha=0.7,
                                     form='block', solver=CGLS,
                                     zero_initial=True)
        self.assertIsInstance(K, BlockTikhonovOperator)

    def test_weighted_flag_allocates_the_weights_up_front(self):
        op = create_tikhonov_operator(self.A, self.ig, regalpha=0.7,
                                      form='block', weighted=True)
        self.assertIsNotNone(op.weights)
        unweighted = create_tikhonov_operator(self.A, self.ig, regalpha=0.7,
                                              form='block')
        self.assertIsNone(unweighted.weights)


# --------------------------------------------------------------------------- #
# WeightedStructOperator
# --------------------------------------------------------------------------- #

class TestWeightedStructOperator(CCPiTestClass):

    def setUp(self):
        numpy.random.seed(13)
        self.ig = ImageGeometry(16, 16)
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)

    def struct_operators(self):
        return [('identity', None), ('wavelet', self.haar), ('gradient', self.grad)]

    # -- weights lifecycle -------------------------------------------------- #

    def test_unweighted_has_no_weight_operator(self):
        op = WeightedStructOperator(self.ig)
        self.assertIsNone(op.weights)
        self.assertFalse(op.is_weighted)
        self.assertIsNone(op.tmp_range_struct)

    def test_enable_weights_creates_ones(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.haar)
        op.enable_weights()
        self.assertTrue(op.is_weighted)
        numpy.testing.assert_allclose(op.weights.as_array(),
                                      numpy.ones(self.ig.shape))

    def test_enable_weights_is_idempotent(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.haar)
        first = op.enable_weights()
        with count_allocated_elements() as tally:
            second = op.enable_weights()
        self.assertIs(first, second)
        self.assertEqual(tally['containers'], 0)

    def test_weighted_flag_matches_enable_weights(self):
        eager = WeightedStructOperator(self.ig, struct_operator=self.haar,
                                       weighted=True)
        lazy = WeightedStructOperator(self.ig, struct_operator=self.haar)
        lazy.enable_weights()
        self.assertEqual(eager.is_weighted, lazy.is_weighted)
        self.assertIsNotNone(lazy.tmp_range_struct)

    def test_set_weights_is_block_aware(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.grad,
                                    weighted=True)
        values = positive_weights(self.grad, seed=14)
        op.set_weights(values)
        for target, source in zip(op.weights.containers, values.containers):
            numpy.testing.assert_allclose(target.as_array(), source.as_array())

    def test_set_weights_on_an_unweighted_operator_raises(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.haar)
        with self.assertRaises(RuntimeError):
            op.set_weights(1.0)

    def test_is_invertible_tracks_orthogonality(self):
        self.assertTrue(WeightedStructOperator(self.ig).is_invertible())
        self.assertTrue(
            WeightedStructOperator(self.ig,
                                   struct_operator=self.haar).is_invertible())
        self.assertFalse(
            WeightedStructOperator(self.ig,
                                   struct_operator=self.grad).is_invertible())

    def test_inverse_raises_for_a_non_orthogonal_l(self):
        """
        (WL)^-1 is formed as L^* W^-1. For a non-orthogonal L that is simply a
        different operator, so it must refuse rather than return a plausible
        wrong answer.
        """
        op = WeightedStructOperator(self.ig, struct_operator=self.grad)
        z = op.range_geometry().allocate(0)
        u = op.domain_geometry().allocate(0)
        with self.assertRaises(ValueError):
            op.inverse(z)
        with self.assertRaises(ValueError):
            op.inverse_adjoint(u)

    def test_is_identity_only_when_both_factors_are_trivial(self):
        self.assertTrue(WeightedStructOperator(self.ig).is_identity())
        self.assertFalse(
            WeightedStructOperator(self.ig, weighted=True).is_identity())
        self.assertFalse(
            WeightedStructOperator(self.ig,
                                   struct_operator=self.haar).is_identity())

    # -- linearity ---------------------------------------------------------- #

    def test_dot_test_unweighted(self):
        for name, L in self.struct_operators():
            with self.subTest(L=name):
                op = WeightedStructOperator(self.ig, struct_operator=L)
                self.assertTrue(LinearOperator.dot_test(op, decimal=4))

    def test_dot_test_weighted(self):
        for name, L in self.struct_operators():
            with self.subTest(L=name):
                op = WeightedStructOperator(self.ig, struct_operator=L,
                                            weighted=True)
                op.set_weights(positive_weights(op, seed=15))
                self.assertTrue(LinearOperator.dot_test(op, decimal=4))

    def test_direct_matches_w_times_l(self):
        """WL x, checked against applying the two factors separately."""
        op = WeightedStructOperator(self.ig, struct_operator=self.haar,
                                    weighted=True)
        weights = positive_weights(self.haar, seed=16)
        op.set_weights(weights)

        x = self.ig.allocate('random', seed=17)
        expected = self.haar.direct(x) * weights
        self.assertNumpyArrayAlmostEqual(op.direct(x).as_array(),
                                         expected.as_array(), decimal=5)

    # -- inverses ----------------------------------------------------------- #

    def invertible_cases(self):
        """(name, L, weighted) for every configuration that has an inverse."""
        for name, L in [('identity', None), ('wavelet', self.haar)]:
            for weighted in (False, True):
                yield name, L, weighted

    def test_inverse_round_trips(self):
        for name, L, weighted in self.invertible_cases():
            with self.subTest(L=name, weighted=weighted):
                op = WeightedStructOperator(self.ig, struct_operator=L,
                                            weighted=weighted)
                if weighted:
                    op.set_weights(positive_weights(op, seed=18))

                x = self.ig.allocate('random', seed=19)
                # (WL)^-1 WL x == x
                self.assertNumpyArrayAlmostEqual(
                    op.inverse(op.direct(x)).as_array(), x.as_array(),
                    decimal=4)

                z = op.range_geometry().allocate('random', seed=20)
                # WL (WL)^-1 z == z
                self.assertNumpyArrayAlmostEqual(
                    op.direct(op.inverse(z)).as_array(), z.as_array(),
                    decimal=4)

    def test_inverse_adjoint_is_the_adjoint_of_the_inverse(self):
        r"""<(WL)^-1 z, u> == <z, (WL)^-* u>, the dot test for the inverse."""
        for name, L, weighted in self.invertible_cases():
            with self.subTest(L=name, weighted=weighted):
                op = WeightedStructOperator(self.ig, struct_operator=L,
                                            weighted=weighted)
                if weighted:
                    op.set_weights(positive_weights(op, seed=21))

                z = op.range_geometry().allocate('random', seed=22)
                u = op.domain_geometry().allocate('random', seed=23)

                lhs = op.inverse(z).dot(u)
                rhs = z.dot(op.inverse_adjoint(u))
                self.assertAlmostEqual(lhs / rhs, 1.0, places=4)

    def test_inverse_is_exact_not_a_commuting_approximation(self):
        """
        (WL)^-1 = L^-1 W^-1 with no assumption that W and L commute. A wavelet
        and a non-constant diagonal do not commute, so a version that reversed
        the two stages would fail here.
        """
        op = WeightedStructOperator(self.ig, struct_operator=self.haar,
                                    weighted=True)
        weights = positive_weights(self.haar, seed=24)
        op.set_weights(weights)

        z = op.range_geometry().allocate('random', seed=25)
        expected = self.haar.adjoint(z / weights)
        self.assertNumpyArrayAlmostEqual(op.inverse(z).as_array(),
                                         expected.as_array(), decimal=5)

    # -- out= --------------------------------------------------------------- #

    def test_out_matches_return_for_every_map(self):
        for name, L, weighted in self.invertible_cases():
            op = WeightedStructOperator(self.ig, struct_operator=L,
                                        weighted=weighted)
            if weighted:
                op.set_weights(positive_weights(op, seed=26))

            pairs = [
                ('direct', op.domain_geometry(), op.range_geometry()),
                ('adjoint', op.range_geometry(), op.domain_geometry()),
                ('inverse', op.range_geometry(), op.domain_geometry()),
                ('inverse_adjoint', op.domain_geometry(), op.range_geometry()),
            ]
            for method, source, target in pairs:
                with self.subTest(L=name, weighted=weighted, method=method):
                    x = source.allocate('random', seed=27)
                    returned = getattr(op, method)(x)
                    out = target.allocate(0)
                    result = getattr(op, method)(x, out=out)
                    self.assertIs(result, out)
                    self.assertNumpyArrayAlmostEqual(out.as_array(),
                                                     returned.as_array(),
                                                     decimal=5)


# --------------------------------------------------------------------------- #
# the two forms
# --------------------------------------------------------------------------- #

class TestTikhonovForms(CCPiTestClass):

    def setUp(self):
        numpy.random.seed(28)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=29))
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.alpha = 0.7

    # -- geometries --------------------------------------------------------- #

    def test_block_geometries(self):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                  regalpha=self.alpha)
        self.assertEqual(K.domain_geometry(), self.ig)
        self.assertEqual(K.range_geometry().get_item(0),
                         self.A.range_geometry())
        self.assertEqual(K.range_geometry().get_item(1),
                         self.haar.range_geometry())

    def test_standard_geometries(self):
        K = TikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                             regalpha=self.alpha)
        self.assertEqual(K.domain_geometry(), self.haar.range_geometry())
        self.assertEqual(K.range_geometry(), self.A.range_geometry())

    def test_standard_form_rejects_a_non_invertible_l(self):
        with self.assertRaises(ValueError):
            TikhonovOperator(self.A, self.ig, struct_operator=self.grad,
                             regalpha=self.alpha)

    # -- linearity ---------------------------------------------------------- #

    def test_dot_tests(self):
        cases = [
            ('block', None), ('block', self.haar), ('block', self.grad),
            ('standard', None), ('standard', self.haar),
        ]
        for form, L in cases:
            for weighted in (False, True):
                with self.subTest(form=form, L=type(L).__name__,
                                  weighted=weighted):
                    K = create_tikhonov_operator(
                        self.A, self.ig, struct_operator=L,
                        regalpha=self.alpha, form=form, weighted=weighted)
                    if weighted:
                        K.reg_operator.set_weights(
                            positive_weights(K.reg_operator, seed=30))
                    self.assertTrue(LinearOperator.dot_test(K, decimal=4))

    # -- what the forms compute --------------------------------------------- #

    def test_block_direct_stacks_a_and_scaled_wl(self):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                  regalpha=self.alpha)
        u = self.ig.allocate('random', seed=31)
        result = K.direct(u)

        self.assertNumpyArrayAlmostEqual(result[0].as_array(),
                                         self.A.direct(u).as_array(),
                                         decimal=5)
        self.assertNumpyArrayAlmostEqual(
            result[1].as_array(),
            (self.alpha * self.haar.direct(u)).as_array(), decimal=5)

    def test_standard_direct_undoes_the_change_of_variable(self):
        r"""K z = A u when z = WL u, which is the whole point of the form."""
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                K = TikhonovOperator(self.A, self.ig,
                                     struct_operator=self.haar,
                                     regalpha=self.alpha, weighted=weighted)
                if weighted:
                    K.reg_operator.set_weights(
                        positive_weights(K.reg_operator, seed=32))

                u = self.ig.allocate('random', seed=33)
                z = K.reg_operator.direct(u)
                self.assertNumpyArrayAlmostEqual(K.direct(z).as_array(),
                                                 self.A.direct(u).as_array(),
                                                 decimal=4)

    def test_standard_form_with_wl_identity_is_exactly_a(self):
        K = TikhonovOperator(self.A, self.ig, regalpha=self.alpha)
        self.assertTrue(K.reg_operator.is_identity())
        self.assertIsNone(K.tmp_domain)

        x = self.ig.allocate('random', seed=34)
        self.assertNumpyArrayAlmostEqual(K.direct(x).as_array(),
                                         self.A.direct(x).as_array(),
                                         decimal=6)
        y = self.A.range_geometry().allocate('random', seed=35)
        self.assertNumpyArrayAlmostEqual(K.adjoint(y).as_array(),
                                         self.A.adjoint(y).as_array(),
                                         decimal=6)

    def test_enable_weights_restores_the_standard_form_staging(self):
        """Turning IRLS on makes WL non-trivial, so tmp_domain must come back."""
        K = TikhonovOperator(self.A, self.ig, regalpha=self.alpha)
        self.assertIsNone(K.tmp_domain)
        K.enable_weights()
        self.assertIsNotNone(K.tmp_domain)

        K.reg_operator.set_weights(positive_weights(K.reg_operator, seed=36))
        u = self.ig.allocate('random', seed=37)
        z = K.reg_operator.direct(u)
        self.assertNumpyArrayAlmostEqual(K.direct(z).as_array(),
                                         self.A.direct(u).as_array(),
                                         decimal=4)

    # -- the adjoint override ------------------------------------------------ #

    def test_block_adjoint_override_matches_the_base_class(self):
        """
        The override exists purely to avoid an allocation; it must agree with
        ``BlockOperator.adjoint`` to the last decimal in every configuration.
        """
        for name, L in [('identity', None), ('wavelet', self.haar),
                        ('gradient', self.grad)]:
            for weighted in (False, True):
                with self.subTest(L=name, weighted=weighted):
                    K = BlockTikhonovOperator(self.A, self.ig,
                                              struct_operator=L,
                                              regalpha=self.alpha,
                                              weighted=weighted)
                    if weighted:
                        K.reg_operator.set_weights(
                            positive_weights(K.reg_operator, seed=38))

                    x = K.range_geometry().allocate('random', seed=39)
                    mine = K.adjoint(x)
                    theirs = BlockOperator.adjoint(K, x)
                    self.assertNumpyArrayAlmostEqual(mine.as_array(),
                                                     theirs.as_array(),
                                                     decimal=5)

    def test_block_adjoint_out_matches_return(self):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                  regalpha=self.alpha, weighted=True)
        K.reg_operator.set_weights(positive_weights(K.reg_operator, seed=40))
        x = K.range_geometry().allocate('random', seed=41)

        returned = K.adjoint(x)
        out = K.domain_geometry().allocate(0)
        result = K.adjoint(x, out=out)
        self.assertIs(result, out)
        self.assertNumpyArrayAlmostEqual(out.as_array(), returned.as_array(),
                                         decimal=6)

    # -- regalpha ------------------------------------------------------------ #

    def test_regalpha_is_settable_after_construction(self):
        K = BlockTikhonovOperator(self.A, self.ig, regalpha=self.alpha)
        self.assertAlmostEqual(K.regalpha, self.alpha)
        K.regalpha = 2.5
        self.assertAlmostEqual(K.regalpha, 2.5)

        u = self.ig.allocate('random', seed=42)
        self.assertNumpyArrayAlmostEqual(K.direct(u)[1].as_array(),
                                         (2.5 * u).as_array(), decimal=5)


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #

class TestTikhonovAllocation(CCPiTestClass):
    """
    The operator half of the budget in ``pullrequest-memory.md`` section 6.

    Two conditional containers, ``c_W = [W != I]`` and
    ``c_S = [W != I and L != I]``, plus the standard form's ``tmp_domain``
    which disappears when ``WL = I``.
    """

    def setUp(self):
        numpy.random.seed(43)
        self.ig = ImageGeometry(16, 16)
        self.m = int(numpy.prod(self.ig.shape))
        self.A = DiagonalOperator(self.ig.allocate('random', seed=44))
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.r_wavelet = int(numpy.prod(self.haar.range_geometry().shape))
        self.alpha = 0.7

    # -- the counter itself --------------------------------------------------#

    def test_counter_sees_a_leaf_allocation(self):
        with count_allocated_elements() as tally:
            self.ig.allocate(0)
        self.assertEqual(tally['elements'], self.m)
        self.assertEqual(tally['containers'], 1)

    def test_counter_sees_a_copy(self):
        x = self.ig.allocate(0)
        with count_allocated_elements() as tally:
            x.copy()
        self.assertEqual(tally['elements'], self.m)

    def test_counter_does_not_double_count_a_block_geometry(self):
        """BlockGeometry.allocate delegates, so only the leaves must be seen."""
        with count_allocated_elements() as tally:
            self.grad.range_geometry().allocate(0)
        self.assertEqual(tally['elements'], 2 * self.m)
        self.assertEqual(tally['containers'], 2)

    # -- WeightedStructOperator ---------------------------------------------#

    def test_weighted_struct_operator_allocation(self):
        m, rw = self.m, self.r_wavelet
        cases = [
            # (L, weighted, expected elements)
            (None,      False, 0),          # c_W = 0, c_S = 0
            (None,      True,  m),          # c_W = 1, c_S = 0  (L is trivial)
            (self.haar, False, 0),          # c_W = 0 removes both
            (self.haar, True,  2 * rw),     # c_W = 1, c_S = 1
            (self.grad, False, 0),
            (self.grad, True,  4 * m),      # r = 2m, so 2r
        ]
        for L, weighted, expected in cases:
            with self.subTest(L=type(L).__name__, weighted=weighted):
                with count_allocated_elements() as tally:
                    WeightedStructOperator(self.ig, struct_operator=L,
                                           weighted=weighted)
                self.assertEqual(tally['elements'], expected)

    def test_unweighted_never_allocates_the_staging_buffer(self):
        for L in (None, self.haar, self.grad):
            op = WeightedStructOperator(self.ig, struct_operator=L)
            self.assertIsNone(op.tmp_range_struct)
            self.assertIsNone(op.weight_operator)

    def test_identity_l_never_allocates_the_staging_buffer(self):
        """c_S is zero when either factor is trivial, weights or no weights."""
        op = WeightedStructOperator(self.ig, weighted=True)
        self.assertIsNotNone(op.weights)
        self.assertIsNone(op.tmp_range_struct)

    # -- the forms -----------------------------------------------------------#

    def test_standard_form_allocation(self):
        m, rw = self.m, self.r_wavelet
        cases = [
            (None,      False, 0),               # WL = I: no staging at all
            (None,      True,  m + m),           # weights + tmp_domain
            (self.haar, False, m),               # tmp_domain only
            (self.haar, True,  2 * rw + m),      # weights + c_S + tmp_domain
        ]
        for L, weighted, expected in cases:
            with self.subTest(L=type(L).__name__, weighted=weighted):
                with count_allocated_elements() as tally:
                    TikhonovOperator(self.A, self.ig, struct_operator=L,
                                     regalpha=self.alpha, weighted=weighted)
                self.assertEqual(tally['elements'], expected)

    def test_block_form_allocation(self):
        m, rw = self.m, self.r_wavelet
        cases = [
            (None,      False, 0),               # WL = I: plain scaled row
            (None,      True,  m + m),           # weights + accumulator
            (self.haar, False, m),               # accumulator only
            (self.haar, True,  2 * rw + m),
            (self.grad, False, m),
            (self.grad, True,  4 * m + m),
        ]
        for L, weighted, expected in cases:
            with self.subTest(L=type(L).__name__, weighted=weighted):
                with count_allocated_elements() as tally:
                    BlockTikhonovOperator(self.A, self.ig, struct_operator=L,
                                          regalpha=self.alpha,
                                          weighted=weighted)
                self.assertEqual(tally['elements'], expected)

    def test_degenerate_standard_form_costs_nothing(self):
        """
        The case ``pullrequest-memory.md`` flags as 'm worse than master, and it
        should not be': with WL = I the transformation is K = A exactly.
        """
        with count_allocated_elements() as tally:
            K = TikhonovOperator(self.A, self.ig, regalpha=self.alpha)
        self.assertEqual(tally['elements'], 0)
        self.assertIsNone(K.tmp_domain)

    # -- the steady state ----------------------------------------------------#

    def test_maps_allocate_nothing_when_given_out(self):
        for L in (None, self.haar):
            for weighted in (False, True):
                op = WeightedStructOperator(self.ig, struct_operator=L,
                                            weighted=weighted)
                if weighted:
                    op.set_weights(positive_weights(op, seed=45))

                pairs = [
                    ('direct', op.domain_geometry(), op.range_geometry()),
                    ('adjoint', op.range_geometry(), op.domain_geometry()),
                    ('inverse', op.range_geometry(), op.domain_geometry()),
                    ('inverse_adjoint', op.domain_geometry(),
                     op.range_geometry()),
                ]
                for method, source, target in pairs:
                    x = source.allocate('random', seed=46)
                    out = target.allocate(0)
                    with self.subTest(L=type(L).__name__, weighted=weighted,
                                      method=method):
                        with count_allocated_elements() as tally:
                            getattr(op, method)(x, out=out)
                        self.assertEqual(tally['containers'], 0)

    def test_block_adjoint_override_allocates_nothing(self):
        """
        ``BlockOperator.adjoint`` allocates one domain container per call.
        The override must not.
        """
        for name, L in [('identity', None), ('wavelet', self.haar),
                        ('gradient', self.grad)]:
            for weighted in (False, True):
                with self.subTest(L=name, weighted=weighted):
                    K = BlockTikhonovOperator(self.A, self.ig,
                                              struct_operator=L,
                                              regalpha=self.alpha,
                                              weighted=weighted)
                    if weighted:
                        K.reg_operator.set_weights(
                            positive_weights(K.reg_operator, seed=47))

                    x = K.range_geometry().allocate('random', seed=48)
                    out = K.domain_geometry().allocate(0)

                    with count_allocated_elements() as tally:
                        K.adjoint(x, out=out)
                    self.assertEqual(tally['containers'], 0)

    def test_base_class_adjoint_does_allocate(self):
        """
        The regression this override exists for. If this ever starts passing
        with zero, core ``BlockOperator`` has been fixed and the override can
        go.
        """
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                  regalpha=self.alpha)
        x = K.range_geometry().allocate('random', seed=49)
        out = K.domain_geometry().allocate(0)
        with count_allocated_elements() as tally:
            BlockOperator.adjoint(K, x, out=out)
        self.assertGreater(tally['containers'], 0)

    def test_block_direct_allocates_nothing_when_given_out(self):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                  regalpha=self.alpha, weighted=True)
        K.reg_operator.set_weights(positive_weights(K.reg_operator, seed=50))
        u = self.ig.allocate('random', seed=51)
        out = K.range_geometry().allocate(0)
        with count_allocated_elements() as tally:
            K.direct(u, out=out)
        self.assertEqual(tally['containers'], 0)

    def test_standard_form_maps_allocate_nothing_when_given_out(self):
        for weighted in (False, True):
            K = TikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                 regalpha=self.alpha, weighted=weighted)
            if weighted:
                K.reg_operator.set_weights(
                    positive_weights(K.reg_operator, seed=52))

            x = K.domain_geometry().allocate('random', seed=53)
            y = K.range_geometry().allocate('random', seed=54)
            with self.subTest(weighted=weighted):
                with count_allocated_elements() as tally:
                    K.direct(x, out=K.range_geometry().allocate(0))
                    K.adjoint(y, out=K.domain_geometry().allocate(0))
                # the two `out` buffers above, and nothing else
                self.assertEqual(tally['containers'], 2)


# --------------------------------------------------------------------------- #
# the r != m case
# --------------------------------------------------------------------------- #

class TestPaddedOrthogonalWavelet(CCPiTestClass):
    """
    ``is_orthogonal()`` is not sufficient for the standard form.

    ``WaveletOperator`` sizes its range from ``pywt.wavedecn_shapes``, which
    pads at the boundary: ``db4`` on a 16x16 image gives a 22x22 coefficient
    space. The filter bank is orthogonal,
    so ``is_orthogonal()`` returns ``True``, but the *operator* is a tall
    isometry, not an invertible map. ``L*L = I`` and ``LL* != I``, so ``L*`` is
    only a left inverse and ``K = A (WL)^-1`` is not well defined. The factory
    has to send this to the block form.

    ``bnd_cond='zero'`` rather than the default ``'symmetric'``: with
    symmetric extension the padded
    decomposition and the reconstruction are not even an adjoint pair, so
    ``dot_test`` fails on the bare ``WaveletOperator`` for every filter longer
    than ``haar``. That is upstream of this file, but it would otherwise be
    misattributed to the operators under test here.
    """

    def setUp(self):
        numpy.random.seed(55)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=56))
        self.db4 = WaveletOperator(self.ig, wname='db4', level=1,
                                   bnd_cond='zero')
        self.m = int(numpy.prod(self.ig.shape))
        self.r = int(numpy.prod(self.db4.range_geometry().shape))
        self.alpha = 0.7

    def test_the_struct_operator_is_a_valid_adjoint_pair(self):
        """Guards the premise of every other test in this class."""
        self.assertTrue(LinearOperator.dot_test(self.db4, decimal=4))

    def test_it_is_orthogonal_but_not_square(self):
        self.assertTrue(self.db4.is_orthogonal())
        self.assertGreater(self.r, self.m)

    def test_it_is_an_isometry_but_not_invertible(self):
        """The measurement the dispatch rule is built on."""
        x = self.ig.allocate('random', seed=57)
        left = (self.db4.adjoint(self.db4.direct(x)) - x).norm() / x.norm()
        self.assertLess(left, 1e-5)          # L*L = I

        z = self.db4.range_geometry().allocate('random', seed=58)
        right = (self.db4.direct(self.db4.adjoint(z)) - z).norm() / z.norm()
        self.assertGreater(right, 1e-2)      # LL* != I

    def test_auto_picks_block_despite_is_orthogonal(self):
        K = create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=self.db4,
                                     regalpha=self.alpha)
        self.assertIsInstance(K, BlockTikhonovOperator)

    def test_standard_form_refuses_and_explains_the_shape(self):
        with self.assertRaises(ValueError) as ctx:
            create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=self.db4,
                                     regalpha=self.alpha, form='standard')
        message = str(ctx.exception)
        self.assertIn('periodization', message)
        self.assertIn("form='block'", message)

    def test_inverse_refuses(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.db4)
        self.assertFalse(op.is_invertible())
        self.assertTrue(op.struct_is_orthogonal())
        with self.assertRaises(ValueError):
            op.inverse(op.range_geometry().allocate(0))

    def test_block_form_handles_it(self):
        """Excluded from the standard form, but not unsupported."""
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                K = BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=self.db4,
                                          regalpha=self.alpha,
                                          weighted=weighted)
                if weighted:
                    K.reg_operator.set_weights(
                        positive_weights(K.reg_operator, seed=59))
                self.assertTrue(LinearOperator.dot_test(K, decimal=4))

    def test_allocation_is_in_r_not_m(self):
        with count_allocated_elements() as tally:
            BlockTikhonovOperator(self.A, self.ig, struct_operator=self.db4,
                                  regalpha=self.alpha, weighted=True)
        # weights (r) + tmp_range_struct (r) + adjoint accumulator (m)
        self.assertEqual(tally['elements'], 2 * self.r + self.m)
        self.assertNotEqual(tally['elements'], 3 * self.m)


class TestPeriodizationWavelet(CCPiTestClass):
    """
    The wavelet configuration the standard form actually supports.

    ``bnd_cond='periodization'`` keeps the coefficient space the same size as
    the image, so an orthogonal filter bank gives an operator that is
    orthogonal *and* square, and so genuinely invertible.
    """

    def setUp(self):
        numpy.random.seed(60)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=61))
        self.db4 = WaveletOperator(self.ig, wname='db4', level=1,
                                   bnd_cond='periodization')
        self.alpha = 0.7

    def test_it_is_invertible(self):
        self.assertTrue(self.db4.is_orthogonal())
        self.assertEqual(self.db4.range_geometry().shape, self.ig.shape)

        z = self.db4.range_geometry().allocate('random', seed=62)
        right = (self.db4.direct(self.db4.adjoint(z)) - z).norm() / z.norm()
        self.assertLess(right, 1e-5)

    def test_auto_picks_standard(self):
        K = create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=self.db4,
                                     regalpha=self.alpha)
        self.assertIsInstance(K, TikhonovOperator)

    def test_standard_form_undoes_the_change_of_variable(self):
        K = TikhonovOperator(self.A, self.ig, struct_operator=self.db4,
                             regalpha=self.alpha, weighted=True)
        K.reg_operator.set_weights(positive_weights(K.reg_operator, seed=63))

        u = self.ig.allocate('random', seed=64)
        z = K.reg_operator.direct(u)
        self.assertNumpyArrayAlmostEqual(K.direct(z).as_array(),
                                         self.A.direct(u).as_array(),
                                         decimal=4)

    def test_dot_test(self):
        for form in ('standard', 'block'):
            with self.subTest(form=form):
                K = create_tikhonov_operator(self.A, self.ig,
                                             struct_operator=self.db4,
                                             regalpha=self.alpha, form=form,
                                             weighted=True)
                K.reg_operator.set_weights(
                    positive_weights(K.reg_operator, seed=65))
                self.assertTrue(LinearOperator.dot_test(K, decimal=4))


# --------------------------------------------------------------------------- #
# biorthogonal wavelets
# --------------------------------------------------------------------------- #

class TestBiorthogonalWavelet(CCPiTestClass):
    """
    Not every wavelet CIL supports is orthogonal.

    ``WaveletOperator`` accepts orthogonal *and* biorthogonal families, but
    ``is_orthogonal()`` returns ``self._wavelet.orthogonal``, which is ``False`` for ``bior*``
    and ``rbio*``. Such an ``L`` is genuinely invertible, yet
    :math:`L^{-1} \\neq L^*` and CIL exposes no ``inverse()``, so the standard
    form cannot represent it. It must go to the block form, which never forms
    an inverse.
    """

    def setUp(self):
        numpy.random.seed(58)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=59))
        # 'zero' for the same adjoint-pair reason as TestNonSquareStructOperator.
        self.bior = WaveletOperator(self.ig, wname='bior3.5', level=1,
                                    bnd_cond='zero')
        self.alpha = 0.7

    def test_the_premise(self):
        self.assertFalse(self.bior.is_orthogonal())
        self.assertTrue(LinearOperator.dot_test(self.bior, decimal=4))

    def test_auto_routes_to_the_block_form(self):
        K = create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=self.bior,
                                     regalpha=self.alpha)
        self.assertIsInstance(K, BlockTikhonovOperator)

    def test_block_form_is_a_valid_linear_operator(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                K = BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=self.bior,
                                          regalpha=self.alpha,
                                          weighted=weighted)
                if weighted:
                    K.reg_operator.set_weights(
                        positive_weights(K.reg_operator, seed=60))
                self.assertTrue(LinearOperator.dot_test(K, decimal=4))

    def test_standard_form_refuses_and_says_why(self):
        with self.assertRaises(ValueError) as ctx:
            create_tikhonov_operator(self.A, self.ig,
                                     struct_operator=self.bior,
                                     regalpha=self.alpha, form='standard')
        message = str(ctx.exception)
        self.assertIn("orthogonal", message)
        self.assertIn("form='block'", message)

    def test_inverse_refuses_rather_than_approximating(self):
        """
        ``adjoint`` is not the inverse here, so returning it would be a silent
        wrong answer rather than an error.
        """
        op = WeightedStructOperator(self.ig, struct_operator=self.bior)
        self.assertFalse(op.is_invertible())
        with self.assertRaises(ValueError):
            op.inverse(op.range_geometry().allocate(0))

    def test_adjoint_really_is_not_the_inverse(self):
        """Justifies the refusal above: L^* L is not the identity."""
        x = self.ig.allocate('random', seed=61)
        round_trip = self.bior.adjoint(self.bior.direct(x))
        difference = (round_trip - x).norm() / x.norm()
        self.assertGreater(difference, 1e-3)


# --------------------------------------------------------------------------- #
# the gradient, which is block-only
# --------------------------------------------------------------------------- #

class TestGradientBlockForm(CCPiTestClass):
    """
    ``GradientOperator`` is the case the block form exists for.

    It is not orthogonal, not square (``r = 2m`` in 2-D), and not invertible at
    all: constants are in its null space. Every other configuration in this file
    has *some* route to the standard form; this one has none, so the block form
    has to be complete on its own. The range is a ``BlockGeometry``, which
    exercises the block-valued weights and the block-aware ``DiagonalOperator``
    that no other case reaches.
    """

    def setUp(self):
        numpy.random.seed(66)
        self.ig = ImageGeometry(16, 16)
        self.m = int(numpy.prod(self.ig.shape))
        self.A = DiagonalOperator(self.ig.allocate('random', seed=67))
        self.grad = GradientOperator(self.ig)
        self.alpha = 0.7

    def operator(self, weighted, seed=68):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.grad,
                                  regalpha=self.alpha, weighted=weighted)
        if weighted:
            K.reg_operator.set_weights(
                positive_weights(K.reg_operator, seed=seed))
        return K

    # -- the premise ---------------------------------------------------------#

    def test_the_gradient_has_no_route_to_the_standard_form(self):
        self.assertFalse(self.grad.is_orthogonal())
        op = WeightedStructOperator(self.ig, struct_operator=self.grad)
        self.assertFalse(op.is_invertible())

    def test_constants_are_in_the_null_space(self):
        """Why no inverse can exist, as opposed to merely being unavailable."""
        constant = self.ig.allocate(3.0)
        self.assertLess(self.grad.direct(constant).norm(), 1e-8)

    def test_auto_picks_block(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                K = create_tikhonov_operator(self.A, self.ig,
                                             struct_operator=self.grad,
                                             regalpha=self.alpha,
                                             weighted=weighted)
                self.assertIsInstance(K, BlockTikhonovOperator)

    # -- geometries ----------------------------------------------------------#

    def test_range_is_a_three_way_split(self):
        r"""Range(K) = BlockGeometry(Range(A), BlockGeometry(m, m))."""
        K = self.operator(weighted=False)
        self.assertEqual(K.domain_geometry(), self.ig)
        self.assertEqual(K.range_geometry().get_item(0),
                         self.A.range_geometry())
        self.assertEqual(K.range_geometry().get_item(1),
                         self.grad.range_geometry())

    def test_weights_are_block_valued(self):
        K = self.operator(weighted=True)
        self.assertTrue(hasattr(K.weights, 'containers'))
        self.assertEqual(len(K.weights.containers), 2)

    # -- linearity -----------------------------------------------------------#

    def test_dot_test(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                self.assertTrue(
                    LinearOperator.dot_test(self.operator(weighted),
                                            decimal=4))

    def test_direct_stacks_a_and_scaled_wl(self):
        K = self.operator(weighted=True)
        u = self.ig.allocate('random', seed=69)
        result = K.direct(u)

        self.assertNumpyArrayAlmostEqual(result[0].as_array(),
                                         self.A.direct(u).as_array(),
                                         decimal=5)
        expected = self.alpha * (K.weights * self.grad.direct(u))
        for got, want in zip(result[1].containers, expected.containers):
            self.assertNumpyArrayAlmostEqual(got.as_array(), want.as_array(),
                                             decimal=5)

    def test_adjoint_override_matches_the_base_class(self):
        """
        The one case where the override's accumulator lives in image space
        while ``x[1]`` is block-valued, so the shapes genuinely differ.
        """
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                K = self.operator(weighted)
                x = K.range_geometry().allocate('random', seed=70)
                self.assertNumpyArrayAlmostEqual(
                    K.adjoint(x).as_array(),
                    BlockOperator.adjoint(K, x).as_array(), decimal=5)

    def test_adjoint_out_matches_return(self):
        K = self.operator(weighted=True)
        x = K.range_geometry().allocate('random', seed=71)
        returned = K.adjoint(x)
        out = K.domain_geometry().allocate(0)
        self.assertIs(K.adjoint(x, out=out), out)
        self.assertNumpyArrayAlmostEqual(out.as_array(), returned.as_array(),
                                         decimal=6)

    def test_direct_out_matches_return(self):
        K = self.operator(weighted=True)
        u = self.ig.allocate('random', seed=72)
        returned = K.direct(u)
        out = K.range_geometry().allocate(0)
        K.direct(u, out=out)
        self.assertNumpyArrayAlmostEqual(out[0].as_array(),
                                         returned[0].as_array(), decimal=6)
        for got, want in zip(out[1].containers, returned[1].containers):
            self.assertNumpyArrayAlmostEqual(got.as_array(), want.as_array(),
                                             decimal=6)

    # -- normal equations ----------------------------------------------------#

    def test_normal_equations_match_the_penalty(self):
        r"""
        The property the whole construction exists to deliver:
        :math:`K^*Ku = A^*Au + \alpha^2 (WL)^*(WL)u`.
        """
        K = self.operator(weighted=True)
        u = self.ig.allocate('random', seed=73)

        got = K.adjoint(K.direct(u))
        WL = K.reg_operator
        want = (self.A.adjoint(self.A.direct(u))
                + self.alpha ** 2 * WL.adjoint(WL.direct(u)))
        self.assertNumpyArrayAlmostEqual(got.as_array(), want.as_array(),
                                         decimal=4)

    # -- memory --------------------------------------------------------------#

    def test_allocation(self):
        cases = [(False, self.m),               # accumulator only
                 (True, 4 * self.m + self.m)]   # weights 2m + staging 2m + acc
        for weighted, expected in cases:
            with self.subTest(weighted=weighted):
                with count_allocated_elements() as tally:
                    BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=self.grad,
                                          regalpha=self.alpha,
                                          weighted=weighted)
                self.assertEqual(tally['elements'], expected)

    def test_maps_allocate_nothing_when_given_out(self):
        for weighted in (False, True):
            K = self.operator(weighted)
            u = self.ig.allocate('random', seed=74)
            x = K.range_geometry().allocate('random', seed=75)
            out_range = K.range_geometry().allocate(0)
            out_domain = K.domain_geometry().allocate(0)
            with self.subTest(weighted=weighted):
                with count_allocated_elements() as tally:
                    K.direct(u, out=out_range)
                    K.adjoint(x, out=out_domain)
                self.assertEqual(tally['containers'], 0)


# --------------------------------------------------------------------------- #
# the solvers
# --------------------------------------------------------------------------- #

class TestSolverAllocation(CCPiTestClass):
    """
    The three-tier allocation contract, end to end.

    ``set_up`` allocates the whole workspace; ``initialise_variables`` and
    ``update`` allocate nothing. The second and third tiers are what make an
    IRLS outer loop run at constant memory, since it re-enters both on every
    outer iteration.

    Expected sizes are built from the geometries rather than hard-coded, so the
    same formula covers ``r = m``, ``r = 2m`` and ``r != m``.
    """

    def setUp(self):
        numpy.random.seed(76)
        self.ig = ImageGeometry(16, 16)
        self.m = int(numpy.prod(self.ig.shape))
        self.A = DiagonalOperator(self.ig.allocate('random', seed=77))
        self.n = int(numpy.prod(self.A.range_geometry().shape))
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.data = self.A.range_geometry().allocate('random', seed=78)
        self.initial = self.ig.allocate(0)
        self.alpha = 0.5

    def cases(self):
        """Every configuration the factory accepts."""
        for solver in (LSQR, CGLS):
            for name, L in [('I', None), ('haar', self.haar),
                            ('grad', self.grad)]:
                for form in ('block', 'standard'):
                    if form == 'standard' and name == 'grad':
                        continue        # correctly rejected, tested elsewhere
                    for weighted in (False, True):
                        yield solver, name, L, form, weighted

    def build(self, solver, L, form, weighted):
        return solver(initial=self.initial, operator=self.A, data=self.data,
                      alpha=self.alpha, struct_operator=L, form=form,
                      weighted=weighted)

    def expected_elements(self, solver, L, form, weighted):
        r"""
        The budget, assembled from first principles.

        ``iterates`` are the solver's own containers: LSQR holds ``x, v, d,
        tmp_iterate`` in ``Domain(K)`` and ``u, tmp_range`` in ``Range(K)``;
        CGLS holds ``x, s, p`` and ``r, q``. On top of that the operator keeps
        at most three things, and each is conditional.
        """
        m, n = self.m, self.n
        r = m if L is None else _size(L.range_geometry())

        domain = m if form == 'block' else r
        range_ = (n + r) if form == 'block' else n
        iterates = (4 if solver is LSQR else 3) * domain + 2 * range_

        struct_is_identity = L is None
        c_W = r if weighted else 0                                  # W
        c_S = r if (weighted and not struct_is_identity) else 0     # L^* staging
        # The image-space buffer: the block form's adjoint accumulator, or the
        # standard form's (WL)^-1 staging. Both vanish exactly when WL = I.
        c_M = 0 if (struct_is_identity and not weighted) else m

        return iterates + c_W + c_S + c_M

    # -- tier 1 --------------------------------------------------------------#

    def test_set_up_allocates_exactly_the_budget(self):
        for solver, name, L, form, weighted in self.cases():
            with self.subTest(solver=solver.__name__, L=name, form=form,
                              weighted=weighted):
                with count_allocated_elements() as tally:
                    self.build(solver, L, form, weighted)
                self.assertEqual(
                    tally['elements'],
                    self.expected_elements(solver, L, form, weighted))

    def test_degenerate_case_costs_no_more_than_unregularised(self):
        """
        L = I without IRLS: the standard form collapses to K = A, so the
        regularised solver holds exactly what the unregularised one does. The
        penalty is carried entirely by the scalar recurrence.
        """
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                with count_allocated_elements() as plain:
                    solver(initial=self.initial, operator=self.A,
                           data=self.data, alpha=0)
                with count_allocated_elements() as regularised:
                    solver(initial=self.initial, operator=self.A,
                           data=self.data, alpha=self.alpha, form='standard')
                self.assertEqual(regularised['elements'], plain['elements'])

    # -- tiers 2 and 3 -------------------------------------------------------#

    def test_initialise_variables_allocates_nothing(self):
        for solver, name, L, form, weighted in self.cases():
            with self.subTest(solver=solver.__name__, L=name, form=form,
                              weighted=weighted):
                alg = self.build(solver, L, form, weighted)
                with count_allocated_elements() as tally:
                    alg.initialise_variables()
                self.assertEqual(tally['containers'], 0)

    def test_update_allocates_nothing(self):
        for solver, name, L, form, weighted in self.cases():
            with self.subTest(solver=solver.__name__, L=name, form=form,
                              weighted=weighted):
                alg = self.build(solver, L, form, weighted)
                with count_allocated_elements() as tally:
                    for _ in range(20):
                        alg.update()
                self.assertEqual(tally['containers'], 0)

    def test_unregularised_path_allocates_nothing_either(self):
        """
        The tier-2 leak was not specific to regularisation: ``self.d =
        self.v.copy()`` and ``self.p = self.s.copy()`` ran in every
        configuration.
        """
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                alg = solver(initial=self.initial, operator=self.A,
                             data=self.data, alpha=0)
                with count_allocated_elements() as tally:
                    alg.initialise_variables()
                    for _ in range(20):
                        alg.update()
                self.assertEqual(tally['containers'], 0)


class TestSolverForms(CCPiTestClass):
    """The `form` flag, and what the solvers do with it."""

    def setUp(self):
        numpy.random.seed(79)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=80) + 1.0)
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.data = self.A.range_geometry().allocate('random', seed=81)
        self.initial = self.ig.allocate(0)
        self.alpha = 0.5

    def test_form_is_recorded_as_a_string(self):
        expected = [(None, 'auto', 'standard'), (None, 'block', 'block'),
                    (self.haar, 'auto', 'standard'), (self.grad, 'auto', 'block')]
        for solver in (LSQR, CGLS):
            for L, requested, resolved in expected:
                with self.subTest(solver=solver.__name__, form=requested):
                    alg = solver(initial=self.initial, operator=self.A,
                                 data=self.data, alpha=self.alpha,
                                 struct_operator=L, form=requested)
                    self.assertEqual(alg.form, resolved)

    def test_unregularised_keeps_the_bare_operator(self):
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                alg = solver(initial=self.initial, operator=self.A,
                             data=self.data, alpha=0)
                self.assertIs(alg.operator, self.A)
                self.assertEqual(alg.form, 'none')

    def test_set_up_accepts_three_arguments(self):
        """`struct_operator` needs a default; test_algorithms relies on it."""
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                alg = solver(initial=self.initial, operator=self.A,
                             data=self.data)
                alg.set_up(initial=self.initial, operator=self.A,
                           data=self.data)
                self.assertTrue(alg.configured)

    def test_iterates_live_in_the_right_space(self):
        """Standard form iterates in Range(L); block form in the image space."""
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                block = solver(initial=self.initial, operator=self.A,
                               data=self.data, alpha=self.alpha,
                               struct_operator=self.grad, form='block')
                self.assertEqual(block.x.shape, self.ig.shape)

                standard = solver(initial=self.initial, operator=self.A,
                                  data=self.data, alpha=self.alpha,
                                  struct_operator=self.haar, form='standard')
                self.assertEqual(standard.x.shape,
                                 self.haar.range_geometry().shape)

    def test_get_output_is_always_in_the_solution_space(self):
        for solver in (LSQR, CGLS):
            for form in ('block', 'standard'):
                with self.subTest(solver=solver.__name__, form=form):
                    alg = solver(initial=self.initial, operator=self.A,
                                 data=self.data, alpha=self.alpha,
                                 struct_operator=self.haar, form=form)
                    alg.run(5, verbose=0)
                    self.assertEqual(alg.get_output().shape, self.ig.shape)

    def test_get_output_honours_out(self):
        for solver in (LSQR, CGLS):
            for form in ('block', 'standard'):
                with self.subTest(solver=solver.__name__, form=form):
                    alg = solver(initial=self.initial, operator=self.A,
                                 data=self.data, alpha=self.alpha,
                                 struct_operator=self.haar, form=form)
                    alg.run(5, verbose=0)
                    returned = alg.get_output()
                    out = self.ig.allocate(0)
                    self.assertIs(alg.get_output(out=out), out)
                    self.assertNumpyArrayAlmostEqual(out.as_array(),
                                                     returned.as_array(),
                                                     decimal=6)

                    with count_allocated_elements() as tally:
                        alg.get_output(out=out)
                    self.assertEqual(tally['containers'], 0)

    def test_the_two_forms_converge_to_the_same_solution(self):
        """
        The check that catches a discrepancy in where alpha enters. Both forms
        minimise ||Au - b||^2 + alpha^2 ||Lu||^2, by different routes.

        CGLS runs fewer iterations than LSQR. It applies CG to the explicitly
        formed regularised normal equations and loses conjugacy in float32
        once the residual reaches machine precision, after which the iterate
        diverges: on this problem ||x|| goes from 5.45 at 50 iterations to
        1.9e+13 at 200. LSQR eliminates each row with a Givens rotation and is
        stable at any count. That is a property of the solvers, not of the
        forms, so it is not what this test is measuring.
        """
        for solver, iterations in [(LSQR, 200), (CGLS, 20)]:
            with self.subTest(solver=solver.__name__):
                results = []
                for form in ('block', 'standard'):
                    alg = solver(initial=self.initial, operator=self.A,
                                 data=self.data, alpha=self.alpha,
                                 struct_operator=self.haar, form=form)
                    alg.run(iterations, verbose=0)
                    results.append(alg.get_output().copy())
                difference = (results[0] - results[1]).norm()
                self.assertLess(difference / results[0].norm(), 1e-3)


class TestWarmStartWarning(CCPiTestClass):
    """
    Standard-form LSQR damps the step, so it cannot start from a non-zero
    iterate. CGLS can, and must not be warned about.
    """

    def setUp(self):
        numpy.random.seed(82)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=83) + 1.0)
        self.data = self.A.range_geometry().allocate('random', seed=84)
        self.zero = self.ig.allocate(0)
        self.nonzero = self.ig.allocate('random', seed=85)
        self.alpha = 0.5

    def test_lsqr_warns_and_names_the_block_form(self):
        with self.assertWarnsRegex(UserWarning, "form='block'"):
            LSQR(initial=self.nonzero, operator=self.A, data=self.data,
                 alpha=self.alpha, form='standard')

    def test_lsqr_does_not_warn_from_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            LSQR(initial=self.zero, operator=self.A, data=self.data,
                 alpha=self.alpha, form='standard')

    def test_lsqr_does_not_warn_in_block_form(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            LSQR(initial=self.nonzero, operator=self.A, data=self.data,
                 alpha=self.alpha, form='block')

    def test_cgls_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            CGLS(initial=self.nonzero, operator=self.A, data=self.data,
                 alpha=self.alpha, form='standard')

    def test_supports_warm_start_flags_the_one_bad_case(self):
        cases = [(LSQR, 'standard', False), (LSQR, 'block', True),
                 (CGLS, 'standard', True), (CGLS, 'block', True)]
        for solver, form, expected in cases:
            with self.subTest(solver=solver.__name__, form=form):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    alg = solver(initial=self.nonzero, operator=self.A,
                                 data=self.data, alpha=self.alpha, form=form)
                self.assertEqual(alg.supports_warm_start, expected)

    def test_cgls_reaches_the_same_minimiser_from_either_start(self):
        """Justifies exempting CGLS: it subtracts the alpha^2 x term."""
        results = []
        for initial in (self.zero, self.nonzero):
            alg = CGLS(initial=initial, operator=self.A, data=self.data,
                       alpha=self.alpha, form='standard')
            alg.run(300, verbose=0)
            results.append(alg.get_output())
        difference = (results[0] - results[1]).norm() / results[0].norm()
        self.assertLess(difference, 1e-4)


# --------------------------------------------------------------------------- #
# the two module-level helpers
# --------------------------------------------------------------------------- #

class TestGeometrySize(CCPiTestClass):
    """
    ``_geometry_size`` is what every capability and budget decision counts with.

    It is used to compare Domain(L) against Range(L), and the allocation tests
    state their budgets in its units, so a block geometry it mis-counted would
    make both silently wrong rather than raise.
    """

    def test_a_leaf_geometry_is_the_product_of_its_shape(self):
        self.assertEqual(256, _size(ImageGeometry(16, 16)))
        self.assertEqual(120, _size(ImageGeometry(4, 5, 6)))
        self.assertEqual(7, _size(VectorGeometry(7)))

    def test_a_block_geometry_is_the_sum_of_its_children(self):
        ig = ImageGeometry(16, 16)
        self.assertEqual(512, _size(BlockGeometry(ig, ig)))
        self.assertEqual(263, _size(BlockGeometry(ig, VectorGeometry(7))))

    def test_it_recurses_through_blocks_of_blocks(self):
        """
        The nested case is not hypothetical: stacking two struct operators in a
        ``BlockOperator`` gives Range(L) exactly this shape. See
        :class:`TestNestedStructOperator`.
        """
        ig = ImageGeometry(16, 16)
        nested = BlockGeometry(BlockGeometry(ig, ig), VectorGeometry(3))
        self.assertEqual(515, _size(nested))

    def test_it_matches_what_the_geometry_actually_allocates(self):
        """Guards the units: the budget is in elements, not in containers."""
        ig = ImageGeometry(8, 5)
        geometries = [ig, VectorGeometry(11), BlockGeometry(ig, ig),
                      BlockGeometry(BlockGeometry(ig, ig), ig)]
        for geometry in geometries:
            with self.subTest(geometry=type(geometry).__name__):
                with count_allocated_elements() as tally:
                    geometry.allocate(0)
                self.assertEqual(_size(geometry), tally['elements'])


class TestStandardFormCapability(CCPiTestClass):
    """
    ``_is_standard_form_capable`` decides whether :math:`(WL)^{-1}` exists.

    The factory, ``WeightedStructOperator.is_invertible`` and the error text in
    ``_require_invertible`` all route through it, so it is tested directly here
    rather than only through the three of them.
    """

    def setUp(self):
        self.ig = ImageGeometry(16, 16)

    def test_no_struct_operator_is_capable(self):
        """L absent means L = I, which is the easiest inverse there is."""
        self.assertTrue(_is_standard_form_capable(None))

    def test_the_identity_is_capable(self):
        self.assertTrue(_is_standard_form_capable(IdentityOperator(self.ig)))

    def test_a_non_orthogonal_operator_is_not(self):
        self.assertFalse(_is_standard_form_capable(GradientOperator(self.ig)))
        self.assertFalse(
            _is_standard_form_capable(StubStructOperator(self.ig, False)))

    def test_orthogonality_alone_is_not_enough(self):
        """
        A padded wavelet reports ``is_orthogonal() == True`` and still fails.

        ``L*L = I`` makes ``L*`` a left inverse only. The square check is what
        turns that into a genuine inverse.
        """
        padded = WaveletOperator(self.ig, wname='db4', level=1)
        self.assertTrue(padded.is_orthogonal())
        self.assertGreater(_size(padded.range_geometry()),
                           _size(padded.domain_geometry()))
        self.assertFalse(_is_standard_form_capable(padded))

    def test_orthogonal_and_square_is_capable(self):
        square = WaveletOperator(self.ig, wname='db2', level=1,
                                 bnd_cond='periodization')
        self.assertTrue(square.is_orthogonal())
        self.assertEqual(_size(square.domain_geometry()),
                         _size(square.range_geometry()))
        self.assertTrue(_is_standard_form_capable(square))

    def test_the_weighted_operator_reports_the_same_verdict(self):
        """``is_invertible`` is this function, and W never changes the answer."""
        for struct in (None, IdentityOperator(self.ig),
                       GradientOperator(self.ig),
                       WaveletOperator(self.ig, wname='db4', level=1)):
            for weighted in (False, True):
                name = type(struct).__name__
                with self.subTest(struct=name, weighted=weighted):
                    op = WeightedStructOperator(self.ig, struct_operator=struct,
                                                weighted=weighted)
                    self.assertEqual(_is_standard_form_capable(struct),
                                     op.is_invertible())


# --------------------------------------------------------------------------- #
# operator norms and linearity
# --------------------------------------------------------------------------- #

class TestOperatorNorms(CCPiTestClass):
    r"""
    Every operator here is linear, and the block form's norm is predictable.

    ``BlockOperator.calculate_norm`` sums the squared row norms, so for
    :math:`K = [A; \alpha W L]` the result is exactly
    :math:`\sqrt{\|A\|^2 + \alpha^2 \|WL\|^2}` with no power-method noise. That
    makes it a real check on the second row carrying the alpha, rather than a
    tolerance-laden approximation.
    """

    def setUp(self):
        numpy.random.seed(90)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=91))
        self.wavelet = WaveletOperator(self.ig, wname='db2', level=1,
                                       bnd_cond='periodization')

    def test_everything_is_linear(self):
        cases = [
            ('WeightedStructOperator',
             WeightedStructOperator(self.ig, struct_operator=self.wavelet)),
            ('BlockTikhonovOperator',
             BlockTikhonovOperator(self.A, self.ig, struct_operator=self.wavelet,
                                   regalpha=1.5)),
            ('TikhonovOperator',
             TikhonovOperator(self.A, self.ig, struct_operator=self.wavelet,
                              regalpha=1.5)),
        ]
        for name, operator in cases:
            with self.subTest(operator=name):
                self.assertTrue(operator.is_linear())

    def test_an_orthogonal_wl_has_unit_norm(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.wavelet)
        self.assertAlmostEqual(1.0, op.norm(), places=6)

    def test_the_weights_scale_the_wl_norm(self):
        op = WeightedStructOperator(self.ig, struct_operator=self.wavelet,
                                    weighted=True)
        op.set_weights(op.range_geometry().allocate(3.0))
        self.assertAlmostEqual(3.0, op.norm(), places=6)

    def test_the_block_norm_combines_both_rows(self):
        for alpha in (0.5, 2.0, 5.0):
            with self.subTest(alpha=alpha):
                K = BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=self.wavelet,
                                          regalpha=alpha)
                expected = self.A.norm() ** 2 + alpha ** 2
                self.assertAlmostEqual(expected, K.norm() ** 2, places=6)

    def test_the_block_norm_follows_a_reset_alpha(self):
        """``regalpha`` is settable, and the norm has to notice."""
        K = BlockTikhonovOperator(self.A, self.ig,
                                  struct_operator=self.wavelet, regalpha=0.5)
        K.regalpha = 4.0
        self.assertAlmostEqual(self.A.norm() ** 2 + 16.0, K.norm() ** 2,
                               places=6)

    def test_the_standard_form_does_not_carry_the_alpha(self):
        """
        :math:`K = A(WL)^{-1}` with :math:`WL = I` is :math:`A`, whatever alpha
        is. In standard form the alpha lives in the solver recurrence instead,
        which is the whole reason the two forms need separate warm-start rules.
        """
        identity = IdentityOperator(self.ig)
        for alpha in (0.0, 2.0, 7.0):
            with self.subTest(alpha=alpha):
                K = TikhonovOperator(identity, self.ig, regalpha=alpha)
                self.assertAlmostEqual(1.0, K.norm(), places=6)


# --------------------------------------------------------------------------- #
# a struct operator whose range is a block of blocks
# --------------------------------------------------------------------------- #

class TestNestedStructOperator(CCPiTestClass):
    r"""
    Stacking struct operators gives Range(L) two levels of nesting.

    :math:`L = [\nabla; I]` penalises gradient and magnitude at once, and its
    range is ``BlockGeometry(BlockGeometry(ig, ig), ig)``. Every other case in
    this file is flat or one level deep, so this is the only one that reaches
    the recursive branch of ``_geometry_size`` and the block-of-blocks path
    through ``DiagonalOperator``.
    """

    def setUp(self):
        numpy.random.seed(95)
        self.ig = ImageGeometry(8, 8)
        self.m = int(numpy.prod(self.ig.shape))
        self.A = DiagonalOperator(self.ig.allocate('random', seed=96))
        self.L = BlockOperator(GradientOperator(self.ig),
                               IdentityOperator(self.ig), shape=(2, 1))
        self.alpha = 1.5
        # 2m from the gradient, m from the identity.
        self.r = 3 * self.m

    def operator(self, weighted=False, seed=97):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.L,
                                  regalpha=self.alpha, weighted=weighted)
        if weighted:
            K.reg_operator.set_weights(
                positive_weights(K.reg_operator, seed=seed))
        return K

    # -- the premise ---------------------------------------------------------#

    def test_the_range_is_nested(self):
        range_geometry = self.L.range_geometry()
        self.assertTrue(hasattr(range_geometry, 'geometries'))
        outer = range_geometry.geometries
        self.assertEqual(2, len(outer))
        self.assertTrue(hasattr(outer[0], 'geometries'))   # the gradient's split
        self.assertFalse(hasattr(outer[1], 'geometries'))  # the identity's leaf
        self.assertEqual(self.r, _size(range_geometry))

    def test_it_has_no_route_to_the_standard_form(self):
        self.assertFalse(_is_standard_form_capable(self.L))
        with self.assertRaises(ValueError):
            TikhonovOperator(self.A, self.ig, struct_operator=self.L,
                             regalpha=self.alpha)

    def test_auto_picks_block(self):
        K = create_tikhonov_operator(self.A, self.ig, struct_operator=self.L,
                                     regalpha=self.alpha, form='auto')
        self.assertEqual('block', K.form)

    # -- it behaves like an operator ---------------------------------------- #

    def test_dot_test(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                self.assertTrue(
                    LinearOperator.dot_test(self.operator(weighted), seed=98))

    def test_direct_stacks_a_and_scaled_wl(self):
        K = self.operator(weighted=True)
        x = self.ig.allocate('random', seed=99)
        result = K.direct(x)

        self.assertNumpyArrayAlmostEqual(self.A.direct(x).as_array(),
                                         result[0].as_array(), decimal=5)
        expected = self.alpha * K.reg_operator.direct(x)
        for got, want in zip(leaves_of(result[1]), leaves_of(expected)):
            self.assertNumpyArrayAlmostEqual(want.as_array(), got.as_array(),
                                             decimal=5)

    def test_adjoint_override_matches_the_base_class(self):
        K = self.operator(weighted=True)
        y = K.range_geometry().allocate('random', seed=100)
        mine = K.adjoint(y)
        theirs = BlockOperator.adjoint(K, y)
        self.assertNumpyArrayAlmostEqual(theirs.as_array(), mine.as_array(),
                                         decimal=5)

    def test_out_matches_return(self):
        K = self.operator(weighted=True)
        x = self.ig.allocate('random', seed=101)
        y = K.range_geometry().allocate('random', seed=102)

        forward = K.range_geometry().allocate(0)
        K.direct(x, out=forward)
        for got, want in zip(leaves_of(forward), leaves_of(K.direct(x))):
            self.assertNumpyArrayAlmostEqual(want.as_array(), got.as_array(),
                                             decimal=5)

        back = self.ig.allocate(0)
        K.adjoint(y, out=back)
        self.assertNumpyArrayAlmostEqual(K.adjoint(y).as_array(),
                                         back.as_array(), decimal=5)

    # -- the weights are nested too ----------------------------------------- #

    def test_the_weights_are_nested_containers(self):
        K = self.operator(weighted=True)
        weights = K.weights
        self.assertTrue(hasattr(weights, 'containers'))
        self.assertEqual(2, len(weights))
        self.assertTrue(hasattr(weights[0], 'containers'))
        self.assertEqual(self.r,
                         sum(int(numpy.prod(leaf.shape))
                             for leaf in leaves_of(weights)))

    def test_enable_weights_starts_from_ones_at_every_depth(self):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.L,
                                  regalpha=self.alpha, weighted=True)
        for leaf in leaves_of(K.weights):
            self.assertNumpyArrayAlmostEqual(numpy.ones(leaf.shape),
                                             leaf.as_array(), decimal=6)

    def test_set_weights_reaches_every_leaf(self):
        """
        ``set_weights`` recurses one level explicitly and relies on ``fill`` for
        the rest, so a nested target is the case that could quietly drop a leaf.
        """
        K = self.operator(weighted=True)
        values = positive_weights(K.reg_operator, seed=103)
        K.reg_operator.set_weights(values)
        for got, want in zip(leaves_of(K.weights), leaves_of(values)):
            self.assertNumpyArrayAlmostEqual(want.as_array(), got.as_array(),
                                             decimal=6)

    def test_set_weights_broadcasts_a_scalar_to_every_leaf(self):
        """
        A scalar must reach every leaf, at whatever depth.

        ``BlockDataContainer.fill`` does not recurse: filling a nested child
        with a scalar is a silent no-op, so ``set_weights`` has to descend to
        the leaves itself. IRLS always passes a full container, so only this
        test exercises the scalar path on a nested block.
        """
        K = self.operator(weighted=True)
        K.reg_operator.set_weights(0.25)
        for leaf in leaves_of(K.weights):
            self.assertNumpyArrayAlmostEqual(
                numpy.full(leaf.shape, 0.25, dtype=leaf.as_array().dtype),
                leaf.as_array(), decimal=6)

    def test_the_weights_bite(self):
        """Zero weights must annihilate the second row, at every depth."""
        K = self.operator(weighted=True)
        K.reg_operator.set_weights(0.0)
        x = self.ig.allocate('random', seed=104)
        for leaf in leaves_of(K.direct(x)[1]):
            self.assertNumpyArrayAlmostEqual(numpy.zeros(leaf.shape),
                                             leaf.as_array(), decimal=6)

    # -- and it costs what the budget says ---------------------------------- #

    def test_allocation(self):
        r"""
        Unweighted the second row is a plain ``WL``, and only the adjoint
        accumulator in Domain(A) is needed: ``m``. Weighted adds ``W`` and the
        ``WL`` staging buffer, both in Range(L): ``m + 2r``.
        """
        for weighted, expected in ((False, self.m), (True, self.m + 2 * self.r)):
            with self.subTest(weighted=weighted):
                with count_allocated_elements() as tally:
                    BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=self.L,
                                          regalpha=self.alpha,
                                          weighted=weighted)
                self.assertEqual(expected, tally['elements'])

    def test_direct_allocates_nothing_when_given_out(self):
        K = self.operator(weighted=True)
        x = self.ig.allocate('random', seed=105)
        forward = K.range_geometry().allocate(0)

        with count_allocated_elements() as tally:
            K.direct(x, out=forward)
        self.assertEqual(0, tally['elements'])

    def test_the_adjoint_pays_for_the_nested_block_operator(self):
        r"""
        The one place the update-allocates-nothing contract does not hold.

        ``BlockTikhonovOperator.adjoint`` is allocation-free for its own two
        rows, but here :math:`L` is itself a ``BlockOperator``, and
        ``BlockOperator.adjoint`` calls each sub-operator after the first
        *without* ``out=`` and then ``+=`` the result. That is one Domain(A)
        container per
        call, charged to :math:`L`, not to anything in this module -- it is the
        very behaviour ``BlockTikhonovOperator.adjoint`` overrides for itself.

        Pinned rather than accepted: a nested :math:`L` costs an extra ``m``
        per iteration, which the budget in ``pullrequest-memory.md`` does not
        cover. Fixing it means an ``out=``-aware ``BlockOperator.adjoint``
        upstream.
        """
        K = self.operator(weighted=True)
        y = K.range_geometry().allocate('random', seed=106)
        back = self.ig.allocate(0)

        with count_allocated_elements() as tally:
            K.adjoint(y, out=back)
        self.assertEqual(self.m, tally['elements'])
        self.assertEqual(1, tally['containers'])

        # Charged to L, not to the Tikhonov wrapper: the bare struct operator
        # allocates exactly the same on its own.
        scratch = self.ig.allocate(0)
        with count_allocated_elements() as bare:
            self.L.adjoint(y[1], out=scratch)
        self.assertEqual(tally['elements'], bare['elements'])
