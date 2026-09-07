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

import sys
import unittest
import warnings
from contextlib import contextmanager
from unittest import mock

import numpy

from cil.framework import (AcquisitionGeometry, BlockGeometry, DataContainer,
                           ImageGeometry, VectorGeometry)
from cil.optimisation.algorithms import Algorithm, CGLS, FISTA, IRLS, LSQR
from cil.optimisation.functions import LeastSquares, TotalVariation
from cil.optimisation.operators import (GradientOperator, IdentityOperator,
                                        MatrixOperator, WaveletOperator)
from cil.optimisation.utilities.callbacks import (Callback, InnerCallback,
                                                  IRLSEarlyStopping,
                                                  OuterCallback,
                                                  ProgressCallback)

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

    A local copy of the helper in ``test_TikhonovOperator.py``, so that this
    file stands on its own. Only the leaf geometries are patched:
    ``BlockGeometry.allocate`` delegates to its children and allocates nothing
    itself. ``DataContainer.clone`` is patched too, because ``.copy()`` is an
    alias for it and neither goes through ``geometry.allocate``.
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


def small_least_squares(rows=8, columns=5, seed=17):
    """A dense over-determined problem small enough to check by hand."""
    rng = numpy.random.default_rng(seed)
    operator = MatrixOperator(rng.standard_normal((rows, columns)))
    data = operator.range_geometry().allocate(0)
    data.fill(rng.standard_normal(rows))
    return operator, data


def vector(geometry, values):
    """A container of `geometry` holding `values`."""
    out = geometry.allocate(0)
    out.fill(numpy.asarray(values, dtype=out.as_array().dtype))
    return out


def anisotropic_tv(image):
    """:math:`\\sum_i |(\\nabla u)_i|`, the penalty IRLS actually approximates.

    IRLS reweights every entry of ``Range(L)`` independently, so with
    :class:`GradientOperator` as :math:`L` the penalty being approximated is the
    *anisotropic* total variation -- the sum of the absolute values of both
    partial derivatives -- not the isotropic one, which couples them through a
    pointwise 2-norm. Comparing against FISTA means matching this, hence
    ``isotropic=False`` there.
    """
    gradient = GradientOperator(image.geometry)
    return sum(float(component.abs().sum())
               for component in gradient.direct(image).containers)


class RecordingCallback(Callback):
    """Records the outer iteration numbers it was called at."""

    def __init__(self):
        super().__init__(verbose=0)
        self.iterations = []

    def __call__(self, algorithm):
        self.iterations.append(algorithm.iteration)


class StubAlgorithm:
    """Hands out a prepared sequence of iterates, like an outer loop would."""

    def __init__(self, iterates):
        self.iterates = list(iterates)
        self.calls = 0

    def get_output(self, out=None):
        value = self.iterates[min(self.calls, len(self.iterates) - 1)]
        self.calls += 1
        if out is not None:
            out.fill(value)
            return out
        return value


# --------------------------------------------------------------------------- #
# the user interface for setting up regularisation
# --------------------------------------------------------------------------- #

class TestRegularisationInterface(CCPiTestClass):
    """
    The path a user actually takes: build an inner solver with a structural
    operator and an alpha, hand it to IRLS, run.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.operator = IdentityOperator(self.ig)
        self.data = self.ig.allocate('random', seed=41)
        self.alpha = 0.7

    def build(self, solver=LSQR, struct_operator=None, form='auto',
              weighted=True, **kwargs):
        return solver(operator=self.operator, data=self.data,
                      initial=self.ig.allocate(0), alpha=self.alpha,
                      struct_operator=struct_operator, form=form,
                      weighted=weighted, **kwargs)

    def test_l1_needs_no_structural_operator(self):
        """``struct_operator=None`` is plain L1 on the solution itself."""
        irls = IRLS(inner_solver=self.build(), max_inner_iteration=2)
        irls.run(2, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_total_variation_through_the_gradient(self):
        irls = IRLS(inner_solver=self.build(
                        struct_operator=GradientOperator(self.ig)),
                    max_inner_iteration=2)
        irls.run(2, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_wavelet_sparsity(self):
        wavelet = WaveletOperator(self.ig, wname='haar', level=1)
        irls = IRLS(inner_solver=self.build(struct_operator=wavelet),
                    max_inner_iteration=2)
        irls.run(2, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_both_inner_solvers_are_accepted(self):
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                irls = IRLS(inner_solver=self.build(solver),
                            max_inner_iteration=2)
                irls.run(2, verbose=0)
                self.assertTrue(
                    numpy.isfinite(irls.get_output().as_array()).all())

    def test_weighted_lsqr_chooses_the_block_form(self):
        """
        Under LSQR, ``weighted=True`` picks block even for an L that could go
        standard.

        The identity is square and orthogonal, so ``form='auto'`` would
        otherwise take the standard form. LSQR damps the step, and IRLS
        restarts each inner solve from the previous outer iterate, so the
        starting point is not zero after the first pass.
        """
        solver = self.build(struct_operator=IdentityOperator(self.ig))
        self.assertFalse(solver.standard_form)

    def test_weighted_cgls_keeps_the_standard_form(self):
        """
        Under CGLS it is the other way round: the standard form iterates on
        one container of size ``r`` rather than a stack of ``m + r``, and CGLS
        warm starts correctly in it, so ``form='auto'`` takes the cheaper one.
        """
        for L in (None, IdentityOperator(self.ig),
                  WaveletOperator(self.ig, wname='haar', level=1)):
            with self.subTest(L=type(L).__name__):
                solver = self.build(CGLS, struct_operator=L)
                self.assertTrue(solver.standard_form)

    def test_weighted_cgls_still_takes_the_block_form_for_a_gradient(self):
        """No inverse, no standard form, whatever the solver can warm start."""
        solver = self.build(CGLS, struct_operator=GradientOperator(self.ig))
        self.assertTrue(solver.block_form)

    def test_lsqr_in_standard_form_is_forced_to_reset(self):
        solver = self.build(struct_operator=IdentityOperator(self.ig),
                            form='standard', weighted=False)
        self.assertTrue(solver.standard_form)
        with self.assertWarns(UserWarning):
            irls = IRLS(inner_solver=solver, max_inner_iteration=2)
        self.assertTrue(irls.reset_state)

    def test_cgls_warm_starts_in_either_form(self):
        solver = self.build(CGLS, struct_operator=IdentityOperator(self.ig),
                            form='standard', weighted=False)
        self.assertTrue(solver.supports_warm_start)
        irls = IRLS(inner_solver=solver, max_inner_iteration=2)
        self.assertFalse(irls.reset_state)

    def test_an_unregularised_solver_is_rejected(self):
        """alpha=0 returns the bare operator, which has no weights to set."""
        bare = LSQR(operator=self.operator, data=self.data,
                    initial=self.ig.allocate(0))
        with self.assertRaises(ValueError):
            IRLS(inner_solver=bare, max_inner_iteration=2)

    def test_weights_are_allocated_when_the_solver_did_not(self):
        solver = self.build(weighted=False)
        self.assertIsNone(solver.weights)
        IRLS(inner_solver=solver, max_inner_iteration=2)
        self.assertIsNotNone(solver.weights)


# --------------------------------------------------------------------------- #
# the form is settled when IRLS attaches
# --------------------------------------------------------------------------- #

class TestTheFormIsSettledAtAttachTime(CCPiTestClass):
    """
    ``form='auto'`` is resolved in the inner solver's ``set_up``, which runs
    before IRLS exists to be asked about. Without ``weighted=True`` there is
    nothing at that point to say a reweighting loop is coming, so LSQR from a
    zero start takes the standard form -- correct for one solve, wrong for the
    second. Attaching IRLS is when the answer becomes known, and where it gets
    applied.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.operator = IdentityOperator(self.ig)
        self.data = self.ig.allocate('random', seed=41)

    def build(self, solver=LSQR, form='auto', weighted=False,
              struct_operator=None):
        return solver(operator=self.operator, data=self.data,
                      initial=self.ig.allocate(0), alpha=0.7,
                      struct_operator=struct_operator, form=form,
                      weighted=weighted)

    def attach(self, solver, **kwargs):
        """Attach IRLS and return it along with any warnings raised."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            irls = IRLS(inner_solver=solver, max_inner_iteration=3, **kwargs)
        return irls, caught

    def test_an_unweighted_lsqr_is_rebuilt_in_block_form(self):
        solver = self.build()
        self.assertTrue(solver.standard_form)

        irls, caught = self.attach(solver)

        self.assertTrue(solver.block_form)
        self.assertEqual([], [str(w.message) for w in caught])
        self.assertFalse(irls.reset_state)

    def test_the_rebuild_agrees_with_having_said_weighted_from_the_start(self):
        """
        The correction is not an approximation of the right answer, it is the
        right answer: the two solvers differ only in when the form was decided.
        """
        rebuilt = IRLS(inner_solver=self.build(), max_inner_iteration=3)
        rebuilt.run(3, verbose=0)

        asked = IRLS(inner_solver=self.build(weighted=True),
                     max_inner_iteration=3)
        asked.run(3, verbose=0)

        self.assertNumpyArrayAlmostEqual(rebuilt.get_output().as_array(),
                                         asked.get_output().as_array())

    def test_the_warm_start_survives_the_rebuild(self):
        """
        The point of the whole exercise. ``reset_state`` stays False, so the
        inner solve resumes from the previous outer iterate rather than from
        the original initial.
        """
        irls = IRLS(inner_solver=self.build(), max_inner_iteration=3)
        irls.run(2, verbose=0)
        self.assertFalse(irls.reset_state)
        self.assertNotEqual(0.0, irls.inner_solver.initial.norm())

    def test_the_weights_land_on_the_rebuilt_operator(self):
        """
        Order matters: rebuilding replaces the operator, so weights allocated
        before it would be written to one nobody iterates on.
        """
        solver = self.build()
        irls, _ = self.attach(solver)
        self.assertIsNotNone(solver.weights)
        self.assertIs(solver.weights, solver.operator.weights)

        # Two outer iterations, not one: the first reweights from the zero
        # initial, where (0 + tau^2)^(-1/4) is exactly one for tau = 1.
        irls.run(2, verbose=0)
        self.assertTrue((solver.weights.as_array() != 1.0).any())

    def test_the_standard_form_buffer_is_not_allocated(self):
        """
        ``tmp_solution`` exists to map the iterate back through ``(WL)^-1``.
        After the rebuild there is no change of variable to undo.
        """
        irls, _ = self.attach(self.build())
        self.assertIsNone(irls.tmp_solution)

    def test_an_explicit_standard_form_is_left_alone(self):
        """A form asked for by name is the caller's decision, not a default."""
        solver = self.build(form='standard')
        irls, caught = self.attach(solver)

        self.assertTrue(solver.standard_form)
        self.assertEqual(1, len(caught))
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertTrue(irls.reset_state)

    def test_a_caller_who_asked_for_a_cold_start_is_not_rebuilt(self):
        """``reset_state=True`` never warm starts, so the standard form is fine."""
        solver = self.build()
        irls, caught = self.attach(solver, reset_state=True)

        self.assertTrue(solver.standard_form)
        self.assertEqual([], [str(w.message) for w in caught])
        self.assertTrue(irls.reset_state)

    def test_cgls_is_never_rebuilt(self):
        """
        CGLS runs on the true regularised normal equations in either form, so
        it keeps the cheaper one.
        """
        solver = self.build(CGLS)
        irls, caught = self.attach(solver)

        self.assertTrue(solver.standard_form)
        self.assertEqual([], [str(w.message) for w in caught])
        self.assertFalse(irls.reset_state)

    def test_rebuilding_a_block_form_solver_changes_nothing(self):
        solver = self.build(weighted=True)
        self.assertTrue(solver.block_form)
        operator = solver.operator

        self.assertFalse(solver.rebuild_in_block_form())
        self.assertIs(operator, solver.operator)

    def test_rebuilding_declines_a_form_that_was_named(self):
        solver = self.build(form='standard')
        self.assertFalse(solver.rebuild_in_block_form())
        self.assertTrue(solver.standard_form)


# --------------------------------------------------------------------------- #
# the maths, written out
# --------------------------------------------------------------------------- #

class TestWeightsAgainstTheFormula(CCPiTestClass):
    r"""
    The weight update is :math:`w = (|Lu|^2 + \tau^2)^{-1/4}`. These tests
    compute that in numpy, with no CIL operator in the way, and compare.
    """

    def setUp(self):
        self.operator, self.data = small_least_squares()
        self.geometry = self.operator.domain_geometry()
        self.alpha = 0.5
        self.tau = 0.3
        rng = numpy.random.default_rng(5)
        self.solution_array = rng.standard_normal(self.geometry.shape[0])

    def irls(self, struct_operator=None, tau=None, tau_factor=0.1):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.geometry.allocate(0), alpha=self.alpha,
                     struct_operator=struct_operator, weighted=True)
        return IRLS(inner_solver=inner, tau=self.tau if tau is None else tau,
                    tau_factor=tau_factor, max_inner_iteration=2)

    def test_identity_structure_reweights_the_solution_itself(self):
        irls = self.irls()
        solution = vector(self.geometry, self.solution_array)

        irls._update_weights(solution)

        expected = (self.solution_array ** 2 + self.tau ** 2) ** -0.25
        self.assertNumpyArrayAlmostEqual(
            expected, irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_a_structural_operator_reweights_its_range(self):
        r"""With :math:`L \neq I` the weights live in Range(L), on :math:`Lu`."""
        matrix = numpy.array([[1.0, -2.0, 0.0, 0.5, 0.0],
                              [0.0, 3.0, 1.0, 0.0, -1.0]])
        irls = self.irls(struct_operator=MatrixOperator(matrix))
        solution = vector(self.geometry, self.solution_array)

        irls._update_weights(solution)

        structure = matrix @ self.solution_array
        expected = (structure ** 2 + self.tau ** 2) ** -0.25
        self.assertNumpyArrayAlmostEqual(
            expected, irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_a_zero_solution_gives_the_tau_floor(self):
        """At u=0 every weight is :math:`\\tau^{-1/2}`, the largest it can be."""
        irls = self.irls()
        irls._update_weights(self.geometry.allocate(0))
        self.assertNumpyArrayAlmostEqual(
            numpy.full(self.geometry.shape[0], self.tau ** -0.5),
            irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_a_large_entry_is_barely_penalised(self):
        r"""
        As :math:`|Lu| \gg \tau` the weight tends to :math:`|Lu|^{-1/2}`, which
        is what turns the weighted 2-norm into the 1-norm:
        :math:`w^2 (Lu)^2 \to |Lu|`.
        """
        irls = self.irls(tau=1e-8)
        solution = vector(self.geometry, [100.0, 0.0, 0.0, 0.0, 0.0])
        irls._update_weights(solution)
        weight = irls.inner_solver.operator.weights.as_array()[0]
        self.assertAlmostEqual(100.0 ** -0.5, weight, places=6)
        self.assertAlmostEqual(100.0, (weight ** 2) * 100.0 ** 2, places=3)

    def test_tau_falls_by_its_factor_each_update(self):
        irls = self.irls(tau=1.0, tau_factor=0.25)
        solution = vector(self.geometry, self.solution_array)
        for expected in (0.25, 0.0625, 0.015625):
            irls._update_weights(solution)
            self.assertAlmostEqual(expected, irls.tau, places=12)

    def test_tau_stops_at_its_floor(self):
        irls = self.irls(tau=1e-7, tau_factor=0.001)
        solution = vector(self.geometry, self.solution_array)
        for _ in range(5):
            irls._update_weights(solution)
        self.assertEqual(1e-8, irls.tau)

    def test_the_weight_update_allocates_nothing(self):
        irls = self.irls()
        solution = vector(self.geometry, self.solution_array)
        irls._update_weights(solution)          # warm up any lazy state
        with count_allocated_elements() as tally:
            irls._update_weights(solution)
        self.assertEqual(0, tally['containers'])


class TestOneOuterIterationByHand(CCPiTestClass):
    """
    One outer iteration of IRLS, rebuilt step by step against a second solver
    driven by hand. Nothing here calls ``IRLS.update``; the point is that the
    sequence -- read the iterate, reweight, reset, run the inner solver -- is
    what the class does, and that it produces the same numbers.
    """

    def setUp(self):
        self.operator, self.data = small_least_squares()
        self.geometry = self.operator.domain_geometry()
        self.alpha = 0.4
        self.tau = 0.5
        self.inner_iterations = 4
        rng = numpy.random.default_rng(23)
        self.initial_array = rng.standard_normal(self.geometry.shape[0])

    def solver(self):
        return LSQR(operator=self.operator, data=self.data,
                    initial=vector(self.geometry, self.initial_array),
                    alpha=self.alpha, weighted=True)

    def test_one_iteration_matches_the_steps_written_out(self):
        irls = IRLS(inner_solver=self.solver(), tau=self.tau, tau_factor=0.1,
                    max_inner_iteration=self.inner_iterations)
        irls.run(1, verbose=0)

        # The same thing, by hand. run(1) performs exactly one update(),
        # because Algorithm.run adds an iteration at iteration == -1 and that
        # first __next__ only records the objective.
        by_hand = self.solver()
        solution = by_hand.get_output()                    # (1) read the iterate
        weights = (solution.as_array() ** 2                # (2) reweight
                   + self.tau ** 2) ** -0.25
        by_hand.operator.weights.fill(weights)
        by_hand.initial = solution                         # (3) warm start
        by_hand.initialise_variables()                     # (4) reset
        by_hand.run(self.inner_iterations, verbose=0)      # (5) inner solve

        self.assertNumpyArrayAlmostEqual(
            by_hand.get_output().as_array(),
            irls.get_output().as_array(), decimal=6)

    def test_the_first_weights_come_from_the_initial_guess(self):
        """
        The reweighting reads the iterate *before* the inner solver runs, so
        the weights of the first outer iteration are those of ``initial``.
        """
        irls = IRLS(inner_solver=self.solver(), tau=self.tau,
                    max_inner_iteration=self.inner_iterations)
        irls.run(1, verbose=0)

        expected = (self.initial_array ** 2 + self.tau ** 2) ** -0.25
        self.assertNumpyArrayAlmostEqual(
            expected, irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_reset_state_discards_the_warm_start(self):
        """
        With ``reset_state=True`` the inner solver restarts from the original
        initial each time, so its iterate differs from the warm-started one.
        """
        warm = IRLS(inner_solver=self.solver(), tau=self.tau,
                    max_inner_iteration=self.inner_iterations)
        cold = IRLS(inner_solver=self.solver(), tau=self.tau, reset_state=True,
                    max_inner_iteration=self.inner_iterations)
        warm.run(3, verbose=0)
        cold.run(3, verbose=0)
        self.assertFalse(numpy.allclose(warm.get_output().as_array(),
                                        cold.get_output().as_array()))

    def test_run_performs_exactly_the_requested_outer_iterations(self):
        irls = IRLS(inner_solver=self.solver(),
                    max_inner_iteration=self.inner_iterations)
        recorder = RecordingCallback()
        irls.run(4, callbacks=[recorder], verbose=0)
        # Callbacks fire after every __next__, including the one at iteration
        # -1 that only records the objective, so 0 is present and the real
        # outer iterations are 1..4.
        self.assertEqual([0, 1, 2, 3, 4], recorder.iterations)
        self.assertEqual(4, irls.iteration)


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #

class TestConvergence(CCPiTestClass):

    def test_outer_iterates_settle(self):
        """The relative change between outer iterates decreases towards zero."""
        operator, data = small_least_squares(rows=12, columns=6, seed=3)
        inner = LSQR(operator=operator, data=data,
                     initial=operator.domain_geometry().allocate(0),
                     alpha=0.3, weighted=True)
        irls = IRLS(inner_solver=inner, max_inner_iteration=10)

        changes, previous = [], None
        for _ in range(8):
            irls.run(1, verbose=0)
            current = irls.get_output().as_array().copy()
            if previous is not None and numpy.linalg.norm(previous):
                changes.append(float(numpy.linalg.norm(current - previous)
                                     / numpy.linalg.norm(previous)))
            previous = current

        self.assertLess(changes[-1], changes[0])
        self.assertLess(changes[-1], 1e-3)

    def test_sparse_recovery_beats_the_unregularised_solution(self):
        """
        The point of L1: on an under-determined system with a sparse truth,
        least squares returns the dense minimum-norm solution and IRLS does
        not.
        """
        rng = numpy.random.default_rng(11)
        rows, columns = 30, 60
        matrix = rng.standard_normal((rows, columns)) / numpy.sqrt(rows)
        truth = numpy.zeros(columns)
        truth[rng.choice(columns, size=5, replace=False)] = rng.standard_normal(5)

        operator = MatrixOperator(matrix)
        geometry = operator.domain_geometry()
        data = operator.range_geometry().allocate(0)
        data.fill(matrix @ truth)

        plain = LSQR(operator=operator, data=data,
                     initial=geometry.allocate(0))
        plain.run(200, verbose=0)

        inner = LSQR(operator=operator, data=data,
                     initial=geometry.allocate(0), alpha=0.05, weighted=True)
        irls = IRLS(inner_solver=inner, tau=1.0, tau_factor=0.5,
                    max_inner_iteration=50)
        irls.run(20, verbose=0)

        norm = numpy.linalg.norm(truth)
        plain_error = numpy.linalg.norm(
            plain.get_output().as_array() - truth) / norm
        irls_error = numpy.linalg.norm(
            irls.get_output().as_array() - truth) / norm

        self.assertLess(irls_error, plain_error)
        self.assertLess(irls_error, 0.25)

    def test_both_inner_solvers_reach_the_same_place(self):
        operator, data = small_least_squares(rows=14, columns=7, seed=8)
        geometry = operator.domain_geometry()
        outputs = {}
        for solver in (LSQR, CGLS):
            inner = solver(operator=operator, data=data,
                           initial=geometry.allocate(0), alpha=0.3,
                           weighted=True)
            irls = IRLS(inner_solver=inner, max_inner_iteration=20)
            irls.run(8, verbose=0)
            outputs[solver.__name__] = irls.get_output().as_array().copy()

        difference = numpy.linalg.norm(outputs['LSQR'] - outputs['CGLS'])
        self.assertLess(difference / numpy.linalg.norm(outputs['LSQR']), 1e-3)


class TestAgainstFISTATotalVariation(CCPiTestClass):
    """
    IRLS with the gradient as :math:`L` solves the same TV problem that FISTA
    solves with a :class:`TotalVariation` proximal term, so the two should land
    in the same place. ``isotropic=False`` because IRLS reweights each partial
    derivative independently, which is the anisotropic penalty.
    """

    def setUp(self):
        self.ig = ImageGeometry(16, 16)
        self.operator = IdentityOperator(self.ig)
        self.alpha = 0.3

        # A piecewise-constant phantom, which is what TV is for, plus noise.
        array = numpy.zeros((16, 16), dtype=numpy.float32)
        array[4:12, 4:12] = 1.0
        rng = numpy.random.default_rng(2)
        self.truth = self.ig.allocate(0)
        self.truth.fill(array)
        self.data = self.ig.allocate(0)
        self.data.fill(array + 0.1 * rng.standard_normal(array.shape))

    def objective(self, image):
        residual = float((self.operator.direct(image) - self.data).norm()) ** 2
        return residual + self.alpha ** 2 * anisotropic_tv(image)

    def test_irls_reaches_the_fista_objective(self):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.ig.allocate(0), alpha=self.alpha,
                     struct_operator=GradientOperator(self.ig), weighted=True)
        irls = IRLS(inner_solver=inner, max_inner_iteration=25)
        irls.run(15, verbose=0)

        fista = FISTA(initial=self.ig.allocate(0),
                      f=LeastSquares(self.operator, self.data, c=1.0),
                      g=(self.alpha ** 2) * TotalVariation(
                          max_iteration=100, isotropic=False))
        fista.run(300, verbose=0)

        irls_objective = self.objective(irls.get_output())
        fista_objective = self.objective(fista.get_output())

        # IRLS smooths the 1-norm with tau, so it cannot beat a true proximal
        # solver outright; it should get close from above.
        self.assertLess(irls_objective,
                        fista_objective * 1.05 + 1e-6)
        self.assertLess(self.objective(self.data) * 0.9, fista_objective * 3)

    def test_irls_denoises_towards_the_truth(self):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.ig.allocate(0), alpha=self.alpha,
                     struct_operator=GradientOperator(self.ig), weighted=True)
        irls = IRLS(inner_solver=inner, max_inner_iteration=25)
        irls.run(15, verbose=0)

        before = float((self.data - self.truth).norm())
        after = float((irls.get_output() - self.truth).norm())
        self.assertLess(after, before)


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #

class TestAllocationCounter(CCPiTestClass):
    """
    The instrument, before the measurements taken with it.

    Every memory claim in this file is an assertion about
    ``count_allocated_elements``, so a counter that quietly saw nothing would
    make all of them pass. These pin what it does see -- CIL container
    allocations, by whichever of the two routes -- and that it leaves the
    classes it patches as it found them.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.m = self.ig.shape[0] * self.ig.shape[1]

    def test_an_allocation_is_one_container_of_its_size(self):
        with count_allocated_elements() as tally:
            self.ig.allocate(0)
        self.assertEqual(1, tally['containers'])
        self.assertEqual(self.m, tally['elements'])

    def test_a_copy_is_counted_too(self):
        """
        ``.copy()`` is an alias for ``clone``, which deep-copies and never goes
        near ``geometry.allocate``. Patching ``clone`` is what stops a
        ``self.d = self.v.copy()`` slipping past the counter.
        """
        x = self.ig.allocate(0)
        with count_allocated_elements() as tally:
            x.copy()
            x.clone()
        self.assertEqual(2, tally['containers'])
        self.assertEqual(2 * self.m, tally['elements'])

    def test_working_in_place_costs_nothing(self):
        """The zero the memory tests assert is reachable, not just unreached."""
        x = self.ig.allocate(1.0)
        y = self.ig.allocate(2.0)
        with count_allocated_elements() as tally:
            x.multiply(y, out=x)
            x.sapyb(1.0, y, 2.0, out=x)
        self.assertEqual(0, tally['containers'])

    def test_a_block_is_counted_once_per_leaf(self):
        """
        ``BlockGeometry.allocate`` delegates to its children and allocates
        nothing itself, so counting the leaves is exact and does not
        double-count the block.
        """
        with count_allocated_elements() as tally:
            BlockGeometry(self.ig, self.ig).allocate(0)
        self.assertEqual(2, tally['containers'])
        self.assertEqual(2 * self.m, tally['elements'])

    def test_every_patched_geometry_is_seen(self):
        """
        Acquisition and vector geometries count as well as image ones. IRLS
        allocates in Range(A) and, for a MatrixOperator, in a VectorGeometry,
        so a counter watching only images would miss half of the budget.
        """
        acquisition = AcquisitionGeometry.create_Parallel2D().set_panel(
            6).set_angles([0, 30, 60])
        with count_allocated_elements() as tally:
            acquisition.allocate(0)
            VectorGeometry(5).allocate(0)
        self.assertEqual(2, tally['containers'])
        self.assertEqual(3 * 6 + 5, tally['elements'])

    def test_the_allocation_still_happens(self):
        """
        The wrapper returns what it wrapped. Counting must not disturb the run
        being counted, so the container comes back filled as asked.
        """
        with count_allocated_elements():
            allocated = self.ig.allocate(2.5)
        numpy.testing.assert_array_equal(2.5, allocated.as_array())
        self.assertEqual(self.ig.shape, allocated.shape)

    def test_the_patches_are_removed_even_when_the_block_raises(self):
        """
        The counter is a global patch on three geometry classes and on
        ``DataContainer.clone``. If a failing assertion inside the block left
        them patched, every later test in the process would be counting into a
        dead tally.
        """
        original = (ImageGeometry.allocate, VectorGeometry.allocate,
                    AcquisitionGeometry.allocate, DataContainer.clone)
        with self.assertRaises(ZeroDivisionError):
            with count_allocated_elements():
                self.ig.allocate(0)
                1 / 0
        self.assertEqual(original,
                         (ImageGeometry.allocate, VectorGeometry.allocate,
                          AcquisitionGeometry.allocate, DataContainer.clone))

        with count_allocated_elements() as tally:
            self.ig.allocate(0)
        self.assertEqual(1, tally['containers'])


class TestMemory(CCPiTestClass):
    """
    The three-tier contract seen from the outer loop: once ``set_up`` has run,
    an IRLS outer iteration allocates nothing, so the loop runs at constant
    memory however long it goes on.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.operator = IdentityOperator(self.ig)
        self.data = self.ig.allocate('random', seed=13)

    def inner(self, struct_operator=None, weighted=True, form='auto'):
        return LSQR(operator=self.operator, data=self.data,
                    initial=self.ig.allocate(0), alpha=0.5,
                    struct_operator=struct_operator, form=form,
                    weighted=weighted)

    def test_an_outer_iteration_allocates_nothing(self):
        irls = IRLS(inner_solver=self.inner(), max_inner_iteration=3)
        irls.run(1, verbose=0)                      # pay for any lazy set-up
        with count_allocated_elements() as tally:
            irls.run(3, verbose=0)
        self.assertEqual(0, tally['containers'])

    def test_a_structural_operator_costs_nothing_extra_per_iteration(self):
        irls = IRLS(inner_solver=self.inner(GradientOperator(self.ig)),
                    max_inner_iteration=3)
        irls.run(1, verbose=0)
        with count_allocated_elements() as tally:
            irls.run(3, verbose=0)
        self.assertEqual(0, tally['containers'])

    def test_standard_form_holds_one_extra_solution_buffer(self):
        """
        In standard form ``get_output`` maps back through :math:`(WL)^{-1}`,
        which needs somewhere to write. IRLS allocates that once, in its
        constructor, rather than on every outer iteration.
        """
        inner = self.inner(struct_operator=IdentityOperator(self.ig),
                           form='standard', weighted=False)
        with self.assertWarns(UserWarning):
            irls = IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertIsNotNone(irls.tmp_solution)

        irls.run(1, verbose=0)
        with count_allocated_elements() as tally:
            irls.run(3, verbose=0)
        self.assertEqual(0, tally['containers'])

    def test_attaching_to_an_unweighted_solver_costs_the_weights_once(self):
        """
        The documented exception to the contract: a solver built without
        ``weighted=True`` pays at attach time instead of in ``set_up``.

        With L the identity that is the weights in Range(L), and the
        accumulator the weighted adjoint then needs in the solution space --
        two containers of m, once, never repeated.
        """
        m = self.ig.shape[0] * self.ig.shape[1]
        inner = self.inner(weighted=False, form='block')
        with count_allocated_elements() as tally:
            IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertEqual(2, tally['containers'])
        self.assertEqual(2 * m, tally['elements'])

    def test_a_structural_operator_adds_its_staging_buffer(self):
        """
        A genuine two-stage :math:`WL` needs an intermediate in Range(L) as
        well, which the identity case collapses away. For the gradient on a
        square image Range(L) is 2m, so the weights and that intermediate come
        to 4m, spread over four leaf containers. The adjoint accumulator does
        not appear because a non-identity L already made ``set_up`` allocate
        it, which is the identity case's second container.
        """
        m = self.ig.shape[0] * self.ig.shape[1]
        inner = self.inner(GradientOperator(self.ig), weighted=False,
                           form='block')
        with count_allocated_elements() as tally:
            IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertEqual(4, tally['containers'])
        self.assertEqual(4 * m, tally['elements'])

    def test_a_weighted_solver_costs_nothing_at_attach_time(self):
        inner = self.inner(weighted=True)
        with count_allocated_elements() as tally:
            IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertEqual(0, tally['containers'])


# --------------------------------------------------------------------------- #
# the stopping condition
# --------------------------------------------------------------------------- #

class TestIRLSEarlyStopping(CCPiTestClass):

    def setUp(self):
        self.geometry = VectorGeometry(4)

    def iterate(self, values):
        return vector(self.geometry, values)

    def test_the_first_call_only_records(self):
        """
        Nothing to compare against at iteration -1, where ``__next__`` records
        the initial objective without running an update.
        """
        callback = IRLSEarlyStopping(epsilon=1.0, verbose=0)
        algorithm = StubAlgorithm([self.iterate([1.0, 1.0, 1.0, 1.0])])
        callback(algorithm)                      # must not raise
        self.assertEqual(numpy.inf, callback.change)

    def test_it_stops_once_the_iterates_settle(self):
        first = self.iterate([1.0, 0.0, 0.0, 0.0])
        second = self.iterate([1.0 + 1e-9, 0.0, 0.0, 0.0])
        callback = IRLSEarlyStopping(epsilon=1e-4, verbose=0)
        algorithm = StubAlgorithm([first, second])

        callback(algorithm)
        with self.assertRaises(StopIteration):
            callback(algorithm)

    def test_it_keeps_going_while_the_iterates_move(self):
        first = self.iterate([1.0, 0.0, 0.0, 0.0])
        second = self.iterate([2.0, 0.0, 0.0, 0.0])
        callback = IRLSEarlyStopping(epsilon=1e-4, verbose=0)
        algorithm = StubAlgorithm([first, second])

        callback(algorithm)
        callback(algorithm)                      # must not raise
        self.assertAlmostEqual(1.0, callback.change, places=6)

    def test_the_change_is_relative_to_the_previous_iterate(self):
        first = self.iterate([10.0, 0.0, 0.0, 0.0])
        second = self.iterate([11.0, 0.0, 0.0, 0.0])
        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = StubAlgorithm([first, second])

        callback(algorithm)
        callback(algorithm)
        self.assertAlmostEqual(0.1, callback.change, places=6)

    def test_a_zero_iterate_does_not_divide_by_zero(self):
        callback = IRLSEarlyStopping(epsilon=1e-4, verbose=0)
        algorithm = StubAlgorithm([self.iterate([0.0, 0.0, 0.0, 0.0]),
                                   self.iterate([1.0, 0.0, 0.0, 0.0])])
        callback(algorithm)
        callback(algorithm)                      # must not raise
        self.assertEqual(numpy.inf, callback.change)

    def test_the_scratch_is_released_when_out_is_ignored(self):
        """
        Block form hands back the live iterate rather than filling ``out``, so
        the buffer is dead weight and the callback drops it.
        """
        class BlockFormAlgorithm(StubAlgorithm):
            def get_output(self, out=None):
                value = self.iterates[min(self.calls,
                                          len(self.iterates) - 1)]
                self.calls += 1
                return value                     # ignores `out`

        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = BlockFormAlgorithm([self.iterate([1.0, 0.0, 0.0, 0.0]),
                                        self.iterate([2.0, 0.0, 0.0, 0.0])])
        callback(algorithm)
        self.assertIsNotNone(callback.scratch)
        callback(algorithm)
        self.assertIsNone(callback.scratch)

    def test_the_scratch_is_kept_when_out_is_honoured(self):
        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = StubAlgorithm([self.iterate([1.0, 0.0, 0.0, 0.0]),
                                   self.iterate([2.0, 0.0, 0.0, 0.0])])
        callback(algorithm)
        callback(algorithm)
        self.assertIsNotNone(callback.scratch)

    def test_comparing_costs_no_allocation_once_running(self):
        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = StubAlgorithm([self.iterate([1.0, 0.0, 0.0, 0.0]),
                                   self.iterate([2.0, 0.0, 0.0, 0.0]),
                                   self.iterate([3.0, 0.0, 0.0, 0.0])])
        callback(algorithm)                      # allocates the two buffers
        callback(algorithm)
        with count_allocated_elements() as tally:
            callback(algorithm)
        self.assertEqual(0, tally['containers'])


class TestInnerSolverBreakdown(CCPiTestClass):
    """
    An inner solve that converges before its iteration budget must stop, not
    produce nan. IRLS runs the inner solver for a fixed count every outer
    iteration, so a single breakdown poisons the whole reconstruction.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.data = self.ig.allocate('random', seed=41)

    def degenerate_solver(self):
        r"""
        :math:`K = [I; \alpha I]` gives :math:`K^T K = (1 + \alpha^2) I`, a
        multiple of the identity, which LSQR solves exactly in one iteration.
        Everything after that is a breakdown of the bidiagonalisation.
        """
        return LSQR(operator=IdentityOperator(self.ig), data=self.data,
                    initial=self.ig.allocate(0), alpha=0.7, weighted=True)

    def test_lsqr_stops_instead_of_returning_nan(self):
        solver = self.degenerate_solver()
        solver.run(10, verbose=0)
        self.assertTrue(numpy.isfinite(solver.get_output().as_array()).all())
        self.assertTrue(numpy.isfinite(solver.loss).all())

    def test_the_exact_solution_survives_the_extra_iterations(self):
        """Stopping keeps the converged iterate rather than overwriting it."""
        one_step = self.degenerate_solver()
        one_step.run(1, verbose=0)
        many = self.degenerate_solver()
        many.run(10, verbose=0)
        self.assertNumpyArrayAlmostEqual(one_step.get_output().as_array(),
                                         many.get_output().as_array(),
                                         decimal=6)

    def test_irls_survives_a_degenerate_inner_solve(self):
        irls = IRLS(inner_solver=self.degenerate_solver(),
                    max_inner_iteration=10)
        irls.run(4, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_a_nondegenerate_problem_still_uses_its_whole_budget(self):
        """The guard must not fire early on an ordinary problem."""
        operator, data = small_least_squares(rows=12, columns=6, seed=3)
        solver = LSQR(operator=operator, data=data,
                      initial=operator.domain_geometry().allocate(0),
                      alpha=0.3, weighted=True)
        solver.run(5, verbose=0)
        self.assertEqual(5, solver.iteration)


class TestIRLSStoppingIntegration(CCPiTestClass):

    def setUp(self):
        self.operator, self.data = small_least_squares(rows=12, columns=6,
                                                       seed=3)
        self.geometry = self.operator.domain_geometry()

    def irls(self, tol=None):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.geometry.allocate(0), alpha=0.3,
                     weighted=True)
        return IRLS(inner_solver=inner, max_inner_iteration=10, tol=tol)

    def test_tol_is_off_by_default(self):
        irls = self.irls()
        self.assertIsNone(irls.tol)
        recorder = RecordingCallback()
        irls.run(6, callbacks=[recorder], verbose=0)
        self.assertEqual(6, irls.iteration)

    def test_tol_stops_the_outer_loop_early(self):
        irls = self.irls(tol=1e-3)
        irls.run(40, verbose=0)
        self.assertLess(irls.iteration, 40)

    def test_stopping_early_reaches_the_same_answer(self):
        stopped = self.irls(tol=1e-6)
        stopped.run(40, verbose=0)
        full = self.irls()
        full.run(40, verbose=0)
        self.assertNumpyArrayAlmostEqual(full.get_output().as_array(),
                                         stopped.get_output().as_array(),
                                         decimal=4)

    def test_a_caller_supplied_stopper_is_not_duplicated(self):
        irls = self.irls(tol=1e-3)
        mine = IRLSEarlyStopping(epsilon=1e-12, verbose=0)
        irls.run(5, callbacks=[mine], verbose=0)
        # With epsilon=1e-12 the caller's stopper never fires, so if IRLS had
        # also attached its own at 1e-3 the loop would have stopped short.
        self.assertEqual(5, irls.iteration)

    def test_a_tighter_tolerance_runs_longer(self):
        loose = self.irls(tol=1e-2)
        loose.run(40, verbose=0)
        tight = self.irls(tol=1e-8)
        tight.run(40, verbose=0)
        self.assertLess(loose.iteration, tight.iteration)


# --------------------------------------------------------------------------- #
# the progress bars
# --------------------------------------------------------------------------- #

class RecordingBar:
    """A stand-in for ``tqdm``, recording where the bar was moved to."""

    def __init__(self, total=None, desc=None, disable=False, **kwargs):
        self.total = total
        self.desc = desc
        self.disable = disable
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def update(self, delta):
        self.n += delta

    def set_postfix(self, **kwargs):
        pass


class TestProgressBars(CCPiTestClass):
    """
    One bar per loop, each counting the iterations of its own loop.

    Both of these went wrong at once, and the second was hidden by the first.
    The outer loop was reported twice, on IRLS's own "Outer Loop" bar and on
    the base class's default ``ProgressCallback``, which counts differently.
    And the IRLS bars counted the calls they received: from ``iteration == -1``
    :meth:`Algorithm.run` takes one extra step that only records the initial
    objective, so the outer bar ran off the end at ``n + 1`` out of ``n`` while
    the one beside it read ``n``.
    """

    # `cil.optimisation.algorithms.IRLS` is the class, re-exported by the
    # package, so the module holding the name `tqdm` has to be fetched from
    # sys.modules rather than by attribute lookup.
    module = sys.modules[IRLS.__module__]

    def setUp(self):
        self.operator, self.data = small_least_squares(rows=12, columns=6,
                                                       seed=3)
        self.geometry = self.operator.domain_geometry()

    def irls(self, solver=LSQR, max_inner_iteration=4):
        inner = solver(operator=self.operator, data=self.data,
                       initial=self.geometry.allocate(0), alpha=0.3,
                       weighted=True)
        return IRLS(inner_solver=inner,
                    max_inner_iteration=max_inner_iteration)

    def bars(self, irls, iterations, **kwargs):
        """Every bar a run opened, in the order it opened them."""
        opened = []

        def record(**bar_kwargs):
            opened.append(RecordingBar(**bar_kwargs))
            return opened[-1]

        with mock.patch.object(self.module, 'tqdm', record):
            irls.run(iterations, **kwargs)
        return opened

    def callbacks_handed_down(self, irls, iterations, **kwargs):
        """The callback lists that reached ``Algorithm.run``, outer first."""
        handed = []
        original = Algorithm.run

        def spy(algorithm, iterations=None, callbacks=None, verbose=1):
            handed.append(list(callbacks or []))
            return original(algorithm, iterations, callbacks=callbacks,
                            verbose=verbose)

        with mock.patch.object(Algorithm, 'run', spy):
            irls.run(iterations, **kwargs)
        return handed

    def test_the_outer_loop_is_not_reported_twice(self):
        outer = self.callbacks_handed_down(self.irls(), 3, verbose=0)[0]
        self.assertEqual(1, sum(isinstance(cb, OuterCallback) for cb in outer))
        self.assertEqual(0, sum(isinstance(cb, ProgressCallback)
                                for cb in outer))

    def test_the_inner_loop_carries_only_its_own_bar(self):
        inner = self.callbacks_handed_down(self.irls(), 3, verbose=0)[1]
        self.assertEqual(1, sum(isinstance(cb, InnerCallback) for cb in inner))
        self.assertEqual(0, sum(isinstance(cb, (OuterCallback,
                                                ProgressCallback))
                                for cb in inner))

    def test_a_caller_can_ask_for_the_base_class_bar_back(self):
        mine = [ProgressCallback(verbose=0)]
        outer = self.callbacks_handed_down(self.irls(), 3, callbacks=mine,
                                           verbose=0)[0]
        self.assertEqual(1, sum(isinstance(cb, ProgressCallback)
                                for cb in outer))
        self.assertEqual(1, sum(isinstance(cb, OuterCallback) for cb in outer))
        # and the caller's own list is left as they passed it
        self.assertEqual(1, len(mine))

    def test_the_outer_bar_finishes_on_the_last_outer_iteration(self):
        bars = self.bars(self.irls(), 3)
        outer = [bar for bar in bars if bar.desc == 'Outer Loop']
        self.assertEqual(1, len(outer))
        self.assertEqual(3, outer[0].total)
        self.assertEqual(3, outer[0].n)

    def test_an_inner_bar_finishes_on_the_last_inner_iteration(self):
        """
        Including the first, which is the one that starts from
        ``iteration == -1`` and so takes the extra bootstrap step. The rest
        resume from wherever the previous inner solve left off, which is why
        the position cannot be read off ``algorithm.iteration`` either.
        """
        bars = self.bars(self.irls(max_inner_iteration=4), 3)
        inner = [bar for bar in bars if bar.desc == 'Inner Loop']
        self.assertEqual(3, len(inner))
        for number, bar in enumerate(inner):
            with self.subTest(inner_solve=number):
                self.assertEqual(4, bar.total)
                self.assertEqual(4, bar.n)

    def test_no_bar_runs_off_the_end(self):
        """CGLS may stop an inner solve early, but never overshoots."""
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                for bar in self.bars(self.irls(solver), 3):
                    self.assertLessEqual(bar.n, bar.total)
                    self.assertGreaterEqual(bar.n, 0)

    def test_verbose_zero_silences_both_bars(self):
        bars = self.bars(self.irls(), 3, verbose=0)
        self.assertTrue(bars)
        self.assertTrue(all(bar.disable for bar in bars))


if __name__ == '__main__':
    unittest.main()
