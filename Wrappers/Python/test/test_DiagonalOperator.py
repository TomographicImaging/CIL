#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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
from unittest import mock

import numpy

from cil.framework import BlockDataContainer, BlockGeometry, ImageGeometry
from cil.optimisation.operators import BlockOperator, ZeroOperator
from cil.optimisation.operators import DiagonalOperator

import test_Operator
from testclass import CCPiTestClass
from utils import initialise_tests

initialise_tests()


# --------------------------------------------------------------------------- #
# helpers

IG = ImageGeometry(4, 3)
IG_OTHER = ImageGeometry(5, 2)


def rand(geometry, seed, dtype=numpy.float32):
    """A filled container. `allocate('random')` is not seeded per call here."""
    container = geometry.allocate(0, dtype=dtype)
    values = numpy.random.RandomState(seed).rand(*container.shape)
    if numpy.issubdtype(dtype, numpy.complexfloating):
        values = values + 1j * numpy.random.RandomState(seed + 1).rand(
            *container.shape)
    container.fill(values.astype(dtype))
    return container


def flat_pair(seed=0, geometry=IG, dtype=numpy.float32):
    """A two-block diagonal and a matching two-block argument."""
    diagonal = BlockDataContainer(rand(geometry, seed, dtype),
                                  rand(geometry, seed + 10, dtype))
    x = BlockDataContainer(rand(geometry, seed + 20, dtype),
                           rand(geometry, seed + 30, dtype))
    return diagonal, x


def nested_pair(seed=0):
    """
    A block of blocks, and a matching argument.

    The first entry is itself a block and the second is a leaf, so the same
    container exercises both branches of the dispatch at the same depth.
    """
    diagonal = BlockDataContainer(
        BlockDataContainer(rand(IG, seed), rand(IG, seed + 10)),
        rand(IG, seed + 20))
    x = BlockDataContainer(
        BlockDataContainer(rand(IG, seed + 30), rand(IG, seed + 40)),
        rand(IG, seed + 50))
    return diagonal, x


def leaves(container):
    """Every leaf DataContainer of a possibly nested BlockDataContainer."""
    if isinstance(container, BlockDataContainer):
        return [leaf for child in container.containers
                for leaf in leaves(child)]
    return [container]


def borrowed(name):
    """
    A test from `test_Operator.TestOperator`, re-run from this file.

    `DiagonalOperator` there is the same block-aware class, so the rebinding is
    a formality; borrowing rather than restating keeps the plain path checked
    against whatever `test_Operator` says it should do, including after that
    file changes.
    """
    original = getattr(test_Operator.TestOperator, name)

    def run(self):
        with mock.patch.object(test_Operator, 'DiagonalOperator',
                               DiagonalOperator):
            original(self)

    run.__name__ = name
    run.__doc__ = f"`test_Operator.TestOperator.{name}`, run against this class."
    return run


# --------------------------------------------------------------------------- #
# dispatch


class TestDiagonalOperatorDispatch(CCPiTestClass):
    """
    Which branch the class takes.

    The whole point of the change is that one name covers both cases, so the
    branch is worth pinning down rather than inferring from behaviour.
    """

    def test_plain_container_takes_the_plain_branch(self):
        operator = DiagonalOperator(rand(IG, 0))
        self.assertIsNone(operator.diagonal_operator_list)

    def test_block_container_builds_one_operator_per_block(self):
        diagonal, _ = flat_pair()
        operator = DiagonalOperator(diagonal)
        self.assertEqual(2, len(operator.diagonal_operator_list))

    def test_nested_container_recurses(self):
        diagonal, _ = nested_pair()
        operator = DiagonalOperator(diagonal)

        children = operator.diagonal_operator_list
        # First child is a block, so it takes the block branch again; second
        # is a leaf and does not.
        self.assertIsNotNone(children[0].diagonal_operator_list)
        self.assertIsNone(children[1].diagonal_operator_list)


# --------------------------------------------------------------------------- #
# the plain case still behaves as it did


class TestPlainDiagonalOperator(CCPiTestClass):
    """
    The non-block path behaves as it always did.

    The first two below are the pre-existing diagonal tests, borrowed rather
    than copied -- direct, adjoint and norm, real and complex. The rest are the
    geometry handling and the real-diagonal fast path.
    """

    test_direct_adjoint_and_norm = borrowed('test_DiagonalOperator')
    test_direct_adjoint_and_norm_complex = borrowed(
        'test_DiagonalOperator_complex')

    def test_domain_geometry_is_taken_from_the_diagonal(self):
        operator = DiagonalOperator(rand(IG, 0))
        self.assertEqual(operator.domain_geometry().shape, IG.shape)
        self.assertEqual(operator.range_geometry().shape, IG.shape)

    def test_explicit_domain_geometry_is_kept(self):
        operator = DiagonalOperator(rand(IG, 0), domain_geometry=IG)
        self.assertEqual(operator.domain_geometry().shape, IG.shape)

    def test_adjoint_of_a_real_diagonal_never_conjugates(self):
        """
        `conjugate()` allocates a whole container per call and the solvers call
        adjoint once per iteration, so the real path -- which is self-adjoint --
        must not go anywhere near it.
        """
        diagonal = rand(IG, 3)
        operator = DiagonalOperator(diagonal)
        x = rand(IG, 4)
        out = IG.allocate(0)
        with mock.patch.object(type(diagonal), 'conjugate',
                               side_effect=AssertionError(
                                   'adjoint of a real diagonal called '
                                   'conjugate(), which allocates')):
            operator.adjoint(x, out=out)
        self.assertNumpyArrayAlmostEqual((diagonal * x).as_array(),
                                         out.as_array())


# --------------------------------------------------------------------------- #
# a block of leaves


class TestBlockDiagonalOperator(CCPiTestClass):

    def setUp(self):
        self.diagonal, self.x = flat_pair(seed=0)
        self.operator = DiagonalOperator(self.diagonal)

    def test_direct_multiplies_block_by_block(self):
        result = self.operator.direct(self.x)
        self.assertEqual(len(result), len(self.diagonal))
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(
                result[i].as_array(), (self.diagonal[i] * self.x[i]).as_array())

    def test_adjoint_multiplies_block_by_block(self):
        result = self.operator.adjoint(self.x)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(
                result[i].as_array(), (self.diagonal[i] * self.x[i]).as_array())

    def test_out_is_filled_and_returned(self):
        out = self.x.copy()
        returned = self.operator.direct(self.x, out=out)
        self.assertIs(returned, out)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(
                out[i].as_array(), (self.diagonal[i] * self.x[i]).as_array())

    def test_out_and_no_out_agree(self):
        out = self.x.copy()
        self.operator.direct(self.x, out=out)
        without = self.operator.direct(self.x)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(out[i].as_array(),
                                          without[i].as_array())

    def test_adjoint_out_is_filled_and_returned(self):
        out = self.x.copy()
        returned = self.operator.adjoint(self.x, out=out)
        self.assertIs(returned, out)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(
                out[i].as_array(), (self.diagonal[i] * self.x[i]).as_array())

    def test_direct_does_not_modify_its_input(self):
        before = [leaf.as_array().copy() for leaf in leaves(self.x)]
        self.operator.direct(self.x)
        for leaf, original in zip(leaves(self.x), before):
            numpy.testing.assert_array_equal(leaf.as_array(), original)

    def test_direct_in_place_on_its_own_argument(self):
        # out is x: every block is written after it is read, so aliasing is
        # safe for an elementwise operator and should give the same answer.
        expected = self.operator.direct(self.x)
        aliased = self.x.copy()
        self.operator.direct(aliased, out=aliased)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(aliased[i].as_array(),
                                          expected[i].as_array())

    def test_norm_is_the_largest_entry_over_all_blocks(self):
        expected = max(numpy.abs(leaf.as_array()).max()
                       for leaf in leaves(self.diagonal))
        numpy.testing.assert_allclose(self.operator.norm(), expected, rtol=1e-6)
        numpy.testing.assert_allclose(self.operator.calculate_norm(), expected,
                                      rtol=1e-6)

    def test_dot_test(self):
        self.assertTrue(self.operator.dot_test(self.operator))

    def test_geometry_is_a_block_geometry(self):
        domain = self.operator.domain_geometry()
        self.assertIsInstance(domain, BlockGeometry)
        self.assertEqual(len(domain.geometries), len(self.diagonal))
        for geometry in domain.geometries:
            self.assertEqual(geometry.shape, IG.shape)
        self.assertIsInstance(self.operator.range_geometry(), BlockGeometry)

    def test_blocks_may_have_different_geometries(self):
        diagonal = BlockDataContainer(rand(IG, 0), rand(IG_OTHER, 1))
        x = BlockDataContainer(rand(IG, 2), rand(IG_OTHER, 3))
        operator = DiagonalOperator(diagonal)

        result = operator.direct(x)
        numpy.testing.assert_allclose(result[0].as_array(),
                                      (diagonal[0] * x[0]).as_array())
        numpy.testing.assert_allclose(result[1].as_array(),
                                      (diagonal[1] * x[1]).as_array())
        self.assertEqual(operator.domain_geometry().geometries[0].shape,
                         IG.shape)
        self.assertEqual(operator.domain_geometry().geometries[1].shape,
                         IG_OTHER.shape)
        self.assertTrue(operator.dot_test(operator))

    def test_adjoint_conjugates_a_complex_diagonal(self):
        diagonal, x = flat_pair(seed=0, dtype=numpy.complex64)
        operator = DiagonalOperator(diagonal)
        result = operator.adjoint(x)
        for i in range(len(diagonal)):
            numpy.testing.assert_allclose(
                result[i].as_array(),
                (diagonal[i].conjugate() * x[i]).as_array(), rtol=1e-6)

    def test_matches_a_block_operator_with_zeros_off_the_diagonal(self):
        # The stated reason for the class: the same map as a BlockOperator
        # holding DiagonalOperators on the diagonal and ZeroOperators off it,
        # without paying for the zero blocks.
        equivalent = BlockOperator(
            DiagonalOperator(self.diagonal[0]), ZeroOperator(IG),
            ZeroOperator(IG), DiagonalOperator(self.diagonal[1]),
            shape=(2, 2))

        mine = self.operator.direct(self.x)
        theirs = equivalent.direct(self.x)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(mine[i].as_array(),
                                          theirs[i].as_array(), rtol=1e-6)

        mine = self.operator.adjoint(self.x)
        theirs = equivalent.adjoint(self.x)
        for i in range(len(self.diagonal)):
            numpy.testing.assert_allclose(mine[i].as_array(),
                                          theirs[i].as_array(), rtol=1e-6)


# --------------------------------------------------------------------------- #
# a block of blocks


class TestNestedBlockDiagonalOperator(CCPiTestClass):
    """
    Blocks of blocks: the dispatch is recursive, so depth should not matter.
    """

    def setUp(self):
        self.diagonal, self.x = nested_pair(seed=0)
        self.operator = DiagonalOperator(self.diagonal)

    def test_direct_reaches_every_leaf(self):
        result = self.operator.direct(self.x)
        numpy.testing.assert_allclose(
            result[0][0].as_array(),
            (self.diagonal[0][0] * self.x[0][0]).as_array())
        numpy.testing.assert_allclose(
            result[0][1].as_array(),
            (self.diagonal[0][1] * self.x[0][1]).as_array())
        numpy.testing.assert_allclose(
            result[1].as_array(), (self.diagonal[1] * self.x[1]).as_array())

    def test_adjoint_reaches_every_leaf(self):
        result = self.operator.adjoint(self.x)
        numpy.testing.assert_allclose(
            result[0][0].as_array(),
            (self.diagonal[0][0] * self.x[0][0]).as_array())
        numpy.testing.assert_allclose(
            result[1].as_array(), (self.diagonal[1] * self.x[1]).as_array())

    def test_out_is_filled_at_depth(self):
        out = self.x.copy()
        returned = self.operator.direct(self.x, out=out)
        self.assertIs(returned, out)
        numpy.testing.assert_allclose(
            out[0][0].as_array(),
            (self.diagonal[0][0] * self.x[0][0]).as_array())
        numpy.testing.assert_allclose(
            out[1].as_array(), (self.diagonal[1] * self.x[1]).as_array())

    def test_direct_does_not_modify_its_input(self):
        before = [leaf.as_array().copy() for leaf in leaves(self.x)]
        self.operator.direct(self.x)
        for leaf, original in zip(leaves(self.x), before):
            numpy.testing.assert_array_equal(leaf.as_array(), original)

    def test_norm_is_the_largest_entry_over_every_leaf(self):
        expected = max(numpy.abs(leaf.as_array()).max()
                       for leaf in leaves(self.diagonal))
        numpy.testing.assert_allclose(self.operator.norm(), expected, rtol=1e-6)

    def test_geometry_is_nested_rather_than_none(self):
        # `diagonal.geometry` is None for a block of blocks --
        # BlockDataContainer.geometry builds a BlockGeometry from
        # `el.geometry.copy()` and BlockGeometry has no copy(), so the
        # AttributeError is swallowed. The operator must compose its geometry
        # from the children instead, or it has no domain and no range, and
        # everything that allocates from them, dot_test included, raises.
        domain = self.operator.domain_geometry()
        self.assertIsInstance(domain, BlockGeometry)
        self.assertIsInstance(domain.geometries[0], BlockGeometry)
        self.assertEqual(domain.geometries[0].geometries[0].shape, IG.shape)
        self.assertEqual(domain.geometries[1].shape, IG.shape)
        self.assertIsInstance(self.operator.range_geometry(), BlockGeometry)

    def test_geometry_allocates_a_container_of_the_right_shape(self):
        allocated = self.operator.domain_geometry().allocate(0)
        self.assertEqual(len(allocated), len(self.x))
        self.assertEqual(len(allocated[0]), len(self.x[0]))
        self.assertEqual(allocated[0][0].shape, self.x[0][0].shape)
        self.assertEqual(allocated[1].shape, self.x[1].shape)

    def test_dot_test(self):
        self.assertTrue(self.operator.dot_test(self.operator))

    def test_three_levels_deep(self):
        diagonal = BlockDataContainer(
            BlockDataContainer(BlockDataContainer(rand(IG, 0), rand(IG, 1)),
                               rand(IG, 2)),
            rand(IG, 3))
        x = BlockDataContainer(
            BlockDataContainer(BlockDataContainer(rand(IG, 4), rand(IG, 5)),
                               rand(IG, 6)),
            rand(IG, 7))
        operator = DiagonalOperator(diagonal)

        result = operator.direct(x)
        numpy.testing.assert_allclose(
            result[0][0][0].as_array(),
            (diagonal[0][0][0] * x[0][0][0]).as_array())
        numpy.testing.assert_allclose(result[1].as_array(),
                                      (diagonal[1] * x[1]).as_array())
        self.assertTrue(operator.dot_test(operator))

    def test_matches_applying_the_leaves_one_at_a_time(self):
        # An independent statement of what the operator should do, built only
        # from container arithmetic, so it does not reuse the operator at all.
        result = self.operator.direct(self.x)
        expected = [
            d * v
            for d, v in zip(leaves(self.diagonal), leaves(self.x))]
        for actual, wanted in zip(leaves(result), expected):
            numpy.testing.assert_allclose(actual.as_array(), wanted.as_array())


if __name__ == '__main__':
    unittest.main()
