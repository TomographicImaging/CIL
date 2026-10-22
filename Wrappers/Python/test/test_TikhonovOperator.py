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

import numpy

from cil.framework import ImageGeometry
from cil.optimisation.operators import (BlockOperator, DiagonalOperator,
                                        GradientOperator, IdentityOperator,
                                        LinearOperator, WaveletOperator)
from cil.optimisation.operators.TikhonovOperator import (
    BlockTikhonovOperator, TikhonovOperator, WeightedOperator,
    _build_tikhonov_operator)

from testclass import CCPiTestClass
from utils import initialise_tests

initialise_tests()


def weighted(L, ig, seed):
    struct = WeightedOperator(ig, operator=L)
    struct.set_weights(
        struct.range_geometry().allocate('random', seed=seed).abs() + 0.5)
    return struct


class TestTikhonovOperator(CCPiTestClass):
    # the standard form K = A (WL)^-1, and the builder both solvers call

    def setUp(self):
        numpy.random.seed(28)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=29))
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.alpha = 0.7

    def test_build(self):
        # alpha == 0 returns the bare operator unwrapped
        for form in ('standard', 'block'):
            for L in (None, self.grad):
                result = _build_tikhonov_operator(self.A, struct_operator=L,
                                                  regalpha=0.0, form=form,
                                                  solution_geometry=self.ig)
                self.assertIs(result, self.A)

        # block is the default form, and solution_geometry defaults to the
        # operator domain
        op = _build_tikhonov_operator(self.A, regalpha=0.7)
        self.assertIsInstance(op, BlockTikhonovOperator)
        self.assertEqual(op.domain_geometry(), self.ig)
        op = _build_tikhonov_operator(self.A, struct_operator=self.haar,
                                      regalpha=0.7, form='standard',
                                      solution_geometry=self.ig)
        self.assertIsInstance(op, TikhonovOperator)

        # an unknown form raises and lists the valid values
        with self.assertRaises(ValueError) as ctx:
            _build_tikhonov_operator(self.A, regalpha=0.7, form='nonsense',
                                     solution_geometry=self.ig)
        for valid in ('standard', 'block'):
            self.assertIn(valid, str(ctx.exception))

        with self.assertRaises(ValueError):
            _build_tikhonov_operator(self.A, regalpha=-1.0,
                                     solution_geometry=self.ig)

    def test_geometries(self):
        K = TikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                             regalpha=self.alpha)
        self.assertEqual(K.domain_geometry(), self.haar.range_geometry())
        self.assertEqual(K.range_geometry(), self.A.range_geometry())
        self.assertEqual(K.solution_geometry(), self.ig)

    def test_dot_test(self):
        for L in (None, self.haar):
            for use_weights in (False, True):
                struct = weighted(L, self.ig, seed=30) if use_weights else L
                K = TikhonovOperator(self.A, self.ig, struct_operator=struct,
                                     regalpha=self.alpha)
                self.assertTrue(LinearOperator.dot_test(K, decimal=4))

    def test_direct(self):
        # K z = A u when z = L u, which is the whole point of the form
        for use_weights in (False, True):
            struct = (weighted(self.haar, self.ig, seed=32) if use_weights
                      else self.haar)
            K = TikhonovOperator(self.A, self.ig, struct_operator=struct,
                                 regalpha=self.alpha)
            u = self.ig.allocate('random', seed=33)
            z = K.map_from_solution(u)
            self.assertNumpyArrayAlmostEqual(K.direct(z).as_array(),
                                             self.A.direct(u).as_array(),
                                             decimal=4)

        # with L = I the transformation is K = A exactly, with no staging
        K = TikhonovOperator(self.A, self.ig, regalpha=self.alpha)
        self.assertTrue(K.struct_is_identity)
        self.assertIsNone(K.tmp_domain)
        x = self.ig.allocate('random', seed=34)
        self.assertNumpyArrayAlmostEqual(K.direct(x).as_array(),
                                         self.A.direct(x).as_array(),
                                         decimal=6)
        y = self.A.range_geometry().allocate('random', seed=35)
        self.assertNumpyArrayAlmostEqual(K.adjoint(y).as_array(),
                                         self.A.adjoint(y).as_array(),
                                         decimal=6)

        # W != I even when L = I, so a weighted identity keeps the staging
        W_L = weighted(None, self.ig, seed=36)
        K = TikhonovOperator(self.A, self.ig, struct_operator=W_L,
                             regalpha=self.alpha)
        self.assertIsNotNone(K.tmp_domain)
        u = self.ig.allocate('random', seed=37)
        z = K.map_from_solution(u)
        self.assertNumpyArrayAlmostEqual(K.direct(z).as_array(),
                                         self.A.direct(u).as_array(),
                                         decimal=4)

    def test_map_solution_round_trip(self):
        K = TikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                             regalpha=self.alpha)
        u = self.ig.allocate('random', seed=86)
        x = K.map_from_solution(u)
        self.assertNumpyArrayAlmostEqual(x.as_array(),
                                         self.haar.direct(u).as_array(),
                                         decimal=5)

        back = K.map_to_solution(x)
        self.assertNumpyArrayAlmostEqual(back.as_array(), u.as_array(),
                                         decimal=4)

        out = self.ig.allocate(0)
        self.assertIs(K.map_to_solution(x, out=out), out)
        self.assertNumpyArrayAlmostEqual(out.as_array(), back.as_array(),
                                         decimal=6)

    def test_rejects_non_invertible_L(self):
        # the check fires where the inverse is formed: on the first apply
        op = TikhonovOperator(self.A, self.ig, struct_operator=self.grad,
                              regalpha=self.alpha)
        with self.assertRaises(ValueError) as ctx:
            op.direct(op.domain_geometry().allocate(0))
        self.assertIn('orthogonal, square L', str(ctx.exception))
        self.assertIn("form='block'", str(ctx.exception))

        # a padded wavelet is orthogonal but not square, so it is refused
        # too, while its square periodized variant goes through
        padded = WaveletOperator(self.ig, wname='db4', level=1,
                                 bnd_cond='zero')
        op = TikhonovOperator(self.A, self.ig, struct_operator=padded,
                              regalpha=self.alpha)
        with self.assertRaises(ValueError):
            op.direct(op.domain_geometry().allocate(0))

        square = WaveletOperator(self.ig, wname='db4', level=1,
                                 bnd_cond='periodization')
        K = TikhonovOperator(self.A, self.ig,
                             struct_operator=weighted(square, self.ig,
                                                      seed=59),
                             regalpha=self.alpha)
        u = self.ig.allocate('random', seed=64)
        z = K.map_from_solution(u)
        self.assertNumpyArrayAlmostEqual(K.direct(z).as_array(),
                                         self.A.direct(u).as_array(),
                                         decimal=4)

    def test_norm(self):
        # K = A(WL)^-1 with WL = I is A whatever alpha is: in standard form
        # the alpha lives in the solver recurrence instead
        identity = IdentityOperator(self.ig)
        for alpha in (0.0, 2.0, 7.0):
            K = TikhonovOperator(identity, self.ig, regalpha=alpha)
            self.assertAlmostEqual(1.0, K.norm(), places=6)

        # the weights scale the norm of WL itself; norm() caches, so each
        # configuration needs a fresh operator
        square = WaveletOperator(self.ig, wname='db2', level=1,
                                 bnd_cond='periodization')
        op = WeightedOperator(self.ig, operator=square)
        self.assertAlmostEqual(1.0, op.norm(), places=6)
        op = WeightedOperator(self.ig, operator=square)
        op.set_weights(op.range_geometry().allocate(3.0))
        self.assertAlmostEqual(3.0, op.norm(), places=6)


class TestBlockTikhonovOperator(CCPiTestClass):
    # the block form K = [A; alpha WL]

    def setUp(self):
        numpy.random.seed(66)
        self.ig = ImageGeometry(16, 16)
        self.A = DiagonalOperator(self.ig.allocate('random', seed=67))
        self.haar = WaveletOperator(self.ig, wname='haar', level=1)
        self.grad = GradientOperator(self.ig)
        self.alpha = 0.7

    def test_geometries(self):
        K = BlockTikhonovOperator(self.A, self.ig, struct_operator=self.haar,
                                  regalpha=self.alpha)
        self.assertEqual(K.domain_geometry(), self.ig)
        self.assertEqual(K.range_geometry().get_item(0),
                         self.A.range_geometry())
        self.assertEqual(K.range_geometry().get_item(1),
                         self.haar.range_geometry())

    def test_dot_test(self):
        # the gradient has no route to the standard form, so the block form
        # has to be complete on its own; its block-valued range exercises
        # the block-aware weights
        nested = BlockOperator(self.grad, IdentityOperator(self.ig),
                               shape=(2, 1))
        for L in (None, self.haar, self.grad, nested):
            for use_weights in (False, True):
                struct = weighted(L, self.ig, seed=30) if use_weights else L
                K = BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=struct,
                                          regalpha=self.alpha)
                self.assertTrue(LinearOperator.dot_test(K, decimal=4))

    def test_direct(self):
        def leaves(container):
            if hasattr(container, 'containers'):
                return [leaf for child in container.containers
                        for leaf in leaves(child)]
            return [container]

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

        # L = [grad; I] penalises gradient and magnitude at once, and its
        # range is a block of blocks: the weights have to bite at depth
        nested = BlockOperator(self.grad, IdentityOperator(self.ig),
                               shape=(2, 1))
        K = BlockTikhonovOperator(self.A, self.ig,
                                  struct_operator=weighted(nested, self.ig,
                                                           seed=97),
                                  regalpha=self.alpha)
        x = self.ig.allocate('random', seed=99)
        result = K.direct(x)
        self.assertNumpyArrayAlmostEqual(self.A.direct(x).as_array(),
                                         result[0].as_array(), decimal=5)
        expected = self.alpha * K.struct_operator.direct(x)
        for got, want in zip(leaves(result[1]), leaves(expected)):
            self.assertNumpyArrayAlmostEqual(want.as_array(), got.as_array(),
                                             decimal=5)

        # zero weights must annihilate the second row, at every depth
        K.struct_operator.set_weights(0.0)
        for leaf in leaves(K.direct(x)[1]):
            self.assertNumpyArrayAlmostEqual(numpy.zeros(leaf.shape),
                                             leaf.as_array(), decimal=6)

    def test_adjoint(self):
        # the override exists purely to avoid an allocation; it must agree
        # with BlockOperator.adjoint in every configuration, including the
        # gradient's, where the accumulator lives in image space while x[1]
        # is block-valued
        for L in (None, self.haar, self.grad):
            for use_weights in (False, True):
                struct = weighted(L, self.ig, seed=38) if use_weights else L
                K = BlockTikhonovOperator(self.A, self.ig,
                                          struct_operator=struct,
                                          regalpha=self.alpha)

                x = K.range_geometry().allocate('random', seed=39)
                mine = K.adjoint(x)
                theirs = BlockOperator.adjoint(K, x)
                self.assertNumpyArrayAlmostEqual(mine.as_array(),
                                                 theirs.as_array(),
                                                 decimal=5)

                out = K.domain_geometry().allocate(0)
                self.assertIs(K.adjoint(x, out=out), out)
                self.assertNumpyArrayAlmostEqual(out.as_array(),
                                                 mine.as_array(), decimal=6)

    def test_normal_equations(self):
        # K^*K u = A^*A u + alpha^2 (WL)^*(WL) u, the property the whole
        # construction exists to deliver
        K = BlockTikhonovOperator(self.A, self.ig,
                                  struct_operator=weighted(self.grad, self.ig,
                                                           seed=68),
                                  regalpha=self.alpha)
        u = self.ig.allocate('random', seed=73)
        got = K.adjoint(K.direct(u))
        WL = K.struct_operator
        want = (self.A.adjoint(self.A.direct(u))
                + self.alpha ** 2 * WL.adjoint(WL.direct(u)))
        self.assertNumpyArrayAlmostEqual(got.as_array(), want.as_array(),
                                         decimal=4)

    def test_regalpha(self):
        K = BlockTikhonovOperator(self.A, self.ig, regalpha=self.alpha)
        self.assertAlmostEqual(K.regalpha, self.alpha)
        K.regalpha = 2.5
        self.assertAlmostEqual(K.regalpha, 2.5)

        u = self.ig.allocate('random', seed=42)
        self.assertNumpyArrayAlmostEqual(K.direct(u)[1].as_array(),
                                         (2.5 * u).as_array(), decimal=5)
        # the norm has to notice: ||K||^2 = ||A||^2 + alpha^2 ||L||^2
        self.assertAlmostEqual(self.A.norm() ** 2 + 2.5 ** 2, K.norm() ** 2,
                               places=5)

    def test_norm(self):
        # BlockOperator.calculate_norm sums the squared row norms, so for an
        # orthogonal L the result is exactly sqrt(||A||^2 + alpha^2)
        square = WaveletOperator(self.ig, wname='db2', level=1,
                                 bnd_cond='periodization')
        for alpha in (0.5, 2.0, 5.0):
            K = BlockTikhonovOperator(self.A, self.ig,
                                      struct_operator=square,
                                      regalpha=alpha)
            expected = self.A.norm() ** 2 + alpha ** 2
            self.assertAlmostEqual(expected, K.norm() ** 2, places=4)
