#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
from utils import initialise_tests
import numpy as np
from cil.framework import ImageGeometry, AcquisitionGeometry, VectorGeometry
from cil.framework import ImageData, AcquisitionData, Partitioner
from cil.framework import BlockDataContainer, BlockGeometry
import functools

from cil.optimisation.operators import GradientOperator, IdentityOperator, BlockOperator
from testclass import CCPiTestClass as BDCUnittest

from cil.utilities import dataexample
import numpy

initialise_tests()

class TestBlockDataContainer(BDCUnittest):
    def skiptest_BlockDataContainerShape(self):
        ig0 = ImageGeometry(12,42,55,32)
        ig1 = ImageGeometry(12,42,55,32)

        data0 = ImageData(geometry=ig0)
        data1 = ImageData(geometry=ig1) + 1

        data2 = ImageData(geometry=ig0) + 2
        data3 = ImageData(geometry=ig1) + 3

        cp0 = BlockDataContainer(data0,data1)
        cp1 = BlockDataContainer(data2,data3)
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp0.T.shape == transpose_shape)


    def skiptest_BlockDataContainerShapeArithmetic(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,4)

        data0 = ImageData(geometry=ig0)
        data1 = ImageData(geometry=ig1) + 1

        data2 = ImageData(geometry=ig0) + 2
        data3 = ImageData(geometry=ig1) + 3

        cp0 = BlockDataContainer(data0,data1)
        #cp1 = BlockDataContainer(data2,data3)
        cp1 = cp0 + 1
        self.assertTrue(cp1.shape == cp0.shape)
        cp1 = cp0.T + 1

        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T - 1
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = (cp0.T + 1)*2
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = (cp0.T + 1)/2
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T.power(2.2)
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T.maximum(3)
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T.abs()
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T.sign()
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T.sqrt()
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)

        cp1 = cp0.T.conjugate()
        transpose_shape = (cp0.shape[1], cp0.shape[0])
        self.assertTrue(cp1.shape == transpose_shape)


    def test_BlockDataContainer(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        # data0 = ImageData(geometry=ig0)
        # data1 = ImageData(geometry=ig1) + 1
        data0 = ig0.allocate(0.)
        data1 = ig1.allocate(1.)

        # data2 = ImageData(geometry=ig0) + 2
        # data3 = ImageData(geometry=ig1) + 3
        data2 = ig0.allocate(2.)
        data3 = ig1.allocate(3.)

        cp0 = BlockDataContainer(data0,data1)
        cp1 = BlockDataContainer(data2,data3)

        cp2 = BlockDataContainer(data0+1, data2+1)
        d = cp2 + data0
        self.assertEqual(d.get_item(0).as_array()[0][0][0], 1)
        try:
            d = cp2 + data1
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)
        d = cp2 - data0
        self.assertEqual(d.get_item(0).as_array()[0][0][0], 1)
        try:
            d = cp2 - data1
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)
        d = cp2 * data2
        self.assertEqual(d.get_item(0).as_array()[0][0][0], 2)
        try:
            d = cp2 * data1
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)

        a = [ (el, ot) for el,ot in zip(cp0.containers,cp1.containers)]
        #cp2 = BlockDataContainer(*a)
        cp2 = cp0.add(cp1)
        self.assertEqual (cp2.get_item(0).as_array()[0][0][0] , 2.)
        self.assertEqual (cp2.get_item(1).as_array()[0][0][0] , 4.)

        cp2 = cp0 + cp1
        self.assertTrue (cp2.get_item(0).as_array()[0][0][0] == 2.)
        self.assertTrue (cp2.get_item(1).as_array()[0][0][0] == 4.)
        cp2 = cp0 + 1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 1. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = cp0 + [1 ,2]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 1. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 3., decimal = 5)
        cp2 += cp1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , +3. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +6., decimal = 5)

        cp2 += 1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , +4. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +7., decimal = 5)

        cp2 += [-2,-1]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 2. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 6., decimal = 5)


        cp2 = cp0.subtract(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == -2.)
        assert (cp2.get_item(1).as_array()[0][0][0] == -2.)
        cp2 = cp0 - cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == -2.)
        assert (cp2.get_item(1).as_array()[0][0][0] == -2.)

        cp2 = cp0 - 1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -1. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0, decimal = 5)
        cp2 = cp0 - [1 ,2]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -1. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -1., decimal = 5)

        cp2 -= cp1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -3. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -4., decimal = 5)

        cp2 -= 1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -4. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -5., decimal = 5)

        cp2 -= [-2,-1]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -2. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -4., decimal = 5)


        cp2 = cp0.multiply(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        assert (cp2.get_item(1).as_array()[0][0][0] == 3.)
        cp2 = cp0 * cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        assert (cp2.get_item(1).as_array()[0][0][0] == 3.)

        cp2 = cp0 * 2
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2, decimal = 5)
        cp2 = 2 * cp0
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2, decimal = 5)
        cp2 = cp0 * [3 ,2]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = cp0 * np.asarray([3 ,2])
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)

        cp2 = [3,2] * cp0
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = np.asarray([3,2]) * cp0
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)

        try:
            cp2 = [3,2,3] * cp0
            #np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
            #np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)
        cp2 *= cp1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0 , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +6., decimal = 5)

        cp2 *= 1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +6., decimal = 5)

        cp2 *= [-2,-1]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -6., decimal = 5)

        try:
            cp2 *= [2,3,5]
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)

        cp2 = cp0.divide(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1./3., decimal=4)
        cp2 = cp0/cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1./3., decimal=4)

        cp2 = cp0 / 2
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        cp2 = cp0 / [3 ,2]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        cp2 = cp0 / np.asarray([3 ,2])
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        cp3 = np.asarray([3 ,2]) / (cp0+1)
        np.testing.assert_almost_equal(cp3.get_item(0).as_array()[0][0][0] , 3. , decimal=5)
        np.testing.assert_almost_equal(cp3.get_item(1).as_array()[0][0][0] , 1, decimal = 5)

        cp2 += 1
        cp2 /= cp1
        # TODO fix inplace division

        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 1./2 , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 1.5/3., decimal = 5)

        cp2 /= 1
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0.5 , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)

        cp2 /= [-2,-1]
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -0.5/2. , decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -0.5, decimal = 5)
        ####

        cp2 = cp0.power(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1., decimal=4)
        cp2 = cp0**cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1., decimal=4)

        cp2 = cp0 ** 2
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0., decimal=5)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 1., decimal = 5)

        cp2 = cp0.maximum(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == cp1.get_item(0).as_array()[0][0][0])
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], cp2.get_item(1).as_array()[0][0][0], decimal=4)


        cp2 = cp0.abs()
        np.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0], 0., decimal=4)
        np.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1., decimal=4)

        cp2 = cp0.subtract(cp1)
        s = cp2.sign()
        np.testing.assert_almost_equal(s.get_item(0).as_array()[0][0][0], -1., decimal=4)
        np.testing.assert_almost_equal(s.get_item(1).as_array()[0][0][0], -1., decimal=4)

        cp2 = cp0.add(cp1)
        s = cp2.sqrt()
        np.testing.assert_almost_equal(s.get_item(0).as_array()[0][0][0], np.sqrt(2), decimal=4)
        np.testing.assert_almost_equal(s.get_item(1).as_array()[0][0][0], np.sqrt(4), decimal=4)

        s = cp0.sum()
        size = functools.reduce(lambda x,y: x*y, data1.shape, 1)
        np.testing.assert_almost_equal(s, 0 + size, decimal=4)
        s0 = 1
        s1 = 1
        for i in cp0.get_item(0).shape:
            s0 *= i
        for i in cp0.get_item(1).shape:
            s1 *= i

        #np.testing.assert_almost_equal(s[1], cp0.get_item(0,0).as_array()[0][0][0]*s0 +cp0.get_item(1,0).as_array()[0][0][0]*s1, decimal=4)
    def test_Nested_BlockDataContainer(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,4)

        # data0 = ImageData(geometry=ig0)
        # data1 = ImageData(geometry=ig1) + 1

        # data2 = ImageData(geometry=ig0) + 2
        # data3 = ImageData(geometry=ig1) + 3
        data0 = ig0.allocate(0.)
        data1 = ig1.allocate(1.)

        data2 = ig0.allocate(2.)
        data3 = ig1.allocate(3.)

        cp0 = BlockDataContainer(data0,data1)
        cp1 = BlockDataContainer(data2,data3)

        nbdc = BlockDataContainer(cp0, cp1)
        nbdc2 = nbdc + 2
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 2. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 3. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 5. , decimal=5)

        nbdc2 = 2 + nbdc
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 2. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 3. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 5. , decimal=5)


        nbdc2 = nbdc * 2
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 2. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 6. , decimal=5)

        nbdc2 = 2 * nbdc
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 2. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 6. , decimal=5)

        nbdc2 = nbdc / 2
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , .5 , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 1. , decimal=5)
        np.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 3./2 , decimal=5)

        c5 = nbdc.get_item(0).power(2).sum()
        c5a = nbdc.power(2).sum()

        cp0 = BlockDataContainer(data0,data2)
        a = cp0 * data2
        b = data2 * cp0
        self.assertBlockDataContainerEqual(a,b)


    def test_NestedBlockDataContainer2(self):
        M, N = 2, 3
        ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N)
        ag = ig
        u = ig.allocate(1)
        op1 = GradientOperator(ig)
        op2 = IdentityOperator(ig, ag)

        operator = BlockOperator(op1, op2, shape=(2,1))

        d1 = op1.direct(u)
        d2 = op2.direct(u)

        d = operator.direct(u)

        dd = operator.domain_geometry()
        ww = operator.range_geometry()

        c1 = d + d

        c2 = 2*d

        c3 = d / (d+0.0001)


        c5 = d.get_item(0).power(2).sum()


    def test_BlockDataContainer_fill(self):

        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ImageData(geometry=ig0)
        data1 = ImageData(geometry=ig1) + 1

        data2 = ImageData(geometry=ig0) + 2
        data3 = ImageData(geometry=ig1) + 3

        cp0 = BlockDataContainer(data0,data1)
        #cp1 = BlockDataContainer(data2,data3)

        cp2 = BlockDataContainer(data0+1, data1+1)

        data0.fill(data2)
        np.testing.assert_array_equal(data0.as_array(), data2.as_array())
        data0 = ImageData(geometry=ig0)

        cp0.fill(cp2)
        self.assertBlockDataContainerEqual(cp0, cp2)


    def test_NestedBlockDataContainer(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(0)
        data2 = ig0.allocate(1)

        cp0 = BlockDataContainer(data0,data2)
        #cp1 = BlockDataContainer(data2,data3)

        nested = BlockDataContainer(cp0, data2, data2)
        out = BlockDataContainer(BlockDataContainer(data0 , data0), data0, data0)
        nested.divide(data2,out=out)
        self.assertBlockDataContainerEqual(out, nested)




    def test_sapyb(self):
        # test axpby between BlockDataContainers
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        data1 = ig0.allocate(2)
        data3 = ig0.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(data1,data3)

        out = cp0 * 0. - 10

        cp0.sapyb(3,cp1, -2,out, num_threads=4)

        # operation should be [  3 * -1 + (-2) * 2 , 3 * 1 + (-2) * 3 ]
        # output should be [ -7 , -3 ]
        res0 = ig0.allocate(-7)
        res2 = ig0.allocate(-3)
        res = BlockDataContainer(res0, res2)

        self.assertBlockDataContainerEqual(out, res)

    def test_sapyb2(self):
        # test axpby with BlockDataContainer and DataContainer
        ig0 = ImageGeometry(2,3,4)
        # ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        data1 = ig0.allocate(2)
        # data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        # cp1 = BlockDataContainer(data1,data3)

        out = cp0 * 0. - 10

        cp0.sapyb(3,data1,-2,out)

        # operation should be [  3 * -1 + (-2) * 2 , 3 * 1 + (-2) * 2 ]
        # output should be [ -7 , -1 ]
        res0 = ig0.allocate(-7)
        res2 = ig0.allocate(-1)
        res = BlockDataContainer(res0, res2)

        self.assertBlockDataContainerEqual(out, res)


    def test_sapyb3(self):
        # test axpby with nested BlockDataContainer
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        # data1 = ig0.allocate(2)
        data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(cp0 *0. +  [2, -2], data3)

        out = cp1 * 0.
        cp2 = out + [1,3]

        cp2.sapyb(3,cp1, -2 ,out)

        # output should be [ [ -1 , 7 ] , 3]
        res0 = ig0.allocate(-1)
        res2 = ig0.allocate(7)
        res3 = ig1.allocate(3)
        res = BlockDataContainer(BlockDataContainer(res0, res2), res3)

        self.assertBlockDataContainerEqual(out, res)

    def test_sapyb4(self):
        # test axpby with nested BlockDataContainer
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        # data1 = ig0.allocate(2)
        data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(cp0 *0. +  [2, -2], data3)

        out = cp1 * 0.
        cp2 = out + [1,3]

        cp2.sapyb(3, cp1, -2, out, num_threads=4)

        # output should be [ [ -1 , 7 ] , 3]
        res0 = ig0.allocate(-1)
        res2 = ig0.allocate(7)
        res3 = ig1.allocate(3)
        res = BlockDataContainer(BlockDataContainer(res0, res2), res3)

        self.assertBlockDataContainerEqual(out, res)


class TestOutParameter(BDCUnittest):
    def setUp(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig1.allocate(1)

        # data1 = ig0.allocate(2)
        # data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        self.ig0 = ig0
        self.ig1 = ig1
        self.cp0 = cp0

    def test_binary_add(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 0

        cp0.add(1 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate(0), self.ig1.allocate(2))
        self.assertBlockDataContainerEqual(cp1, res)


    def test_binary_subtract(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 0

        cp0.subtract(1 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate(-1-1), self.ig1.allocate(1-1))
        self.assertBlockDataContainerEqual(cp1, res)


    def test_binary_multiply(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 0

        cp0.multiply(2 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate(-1*2), self.ig1.allocate(1*2))
        self.assertBlockDataContainerAlmostEqual(cp1, res)
    def test_binary_divide(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 0

        cp0.divide(2 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate(-1/2), self.ig1.allocate(1/2))
        self.assertBlockDataContainerAlmostEqual(cp1, res)
    def test_binary_power(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 0

        cp0.power(2 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate((-1)**2), self.ig1.allocate((1)**2))
        self.assertBlockDataContainerAlmostEqual(cp1, res)
    def test_binary_maximum(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 10

        cp0.maximum(0 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate(0), self.ig1.allocate(1))
        self.assertBlockDataContainerAlmostEqual(cp1, res)
    def test_binary_minimum(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0 * 10

        cp0.minimum(0 , out = cp1)
        res = BlockDataContainer(self.ig0.allocate(-1), self.ig1.allocate(0))
        self.assertBlockDataContainerAlmostEqual(cp1, res)

    def test_unary_abs(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp0.abs(out = cp0)
        res = BlockDataContainer(self.ig0.allocate(1), self.ig1.allocate(1))
        self.assertBlockDataContainerAlmostEqual(res, cp0)
    def test_unary_sign(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0.sign()
        res = BlockDataContainer(self.ig0.allocate(-1), self.ig1.allocate(1))
        self.assertBlockDataContainerAlmostEqual(res, cp1)
    def test_unary_sign2(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp0.sign(out=cp0)
        res = BlockDataContainer(self.ig0.allocate(-1), self.ig1.allocate(1))
        self.assertBlockDataContainerAlmostEqual(res, cp0)
    def test_unary_sqrt(self):
        # test axpby with nested BlockDataContainer
        data0 = self.ig0.allocate(4)
        data2 = self.ig1.allocate(8)

        # data1 = ig0.allocate(2)
        # data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = cp0.sqrt()
        res = BlockDataContainer(self.ig0.allocate(np.sqrt(4)), self.ig1.allocate(np.sqrt(8)))
        self.assertBlockDataContainerAlmostEqual(res, cp1)
    def test_unary_sqrt2(self):
        # test axpby with nested BlockDataContainer
        data0 = self.ig0.allocate(4)
        data2 = self.ig1.allocate(8)

        # data1 = ig0.allocate(2)
        # data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp0.sqrt(out=cp0)
        res = BlockDataContainer(self.ig0.allocate(np.sqrt(4)), self.ig1.allocate(np.sqrt(8)))
        self.assertBlockDataContainerAlmostEqual(res, cp0)

    def test_unary_conjugate(self):
        # test axpby with nested BlockDataContainer
        data0 = self.ig0.allocate(4+3j, dtype=np.complex64)
        data2 = self.ig1.allocate(1-1j, dtype=np.complex64)

        # data1 = ig0.allocate(2)
        # data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = cp0.conjugate()
        res = BlockDataContainer(self.ig0.allocate(4-3j, dtype=np.complex64), self.ig1.allocate(1+1j, dtype=np.complex64))
        self.assertBlockDataContainerAlmostEqual(res, cp1)
    def test_unary_conjugate2(self):
        # test axpby with nested BlockDataContainer
        data0 = self.ig0.allocate(4+3j, dtype=np.complex64)
        data2 = self.ig1.allocate(1-1j, dtype=np.complex64)

        # data1 = ig0.allocate(2)
        # data3 = ig1.allocate(3)

        cp0 = BlockDataContainer(data0,data2)
        cp0.conjugate(out=cp0)
        res = BlockDataContainer(self.ig0.allocate(4-3j, dtype=np.complex64), self.ig1.allocate(1+1j, dtype=np.complex64))
        self.assertBlockDataContainerAlmostEqual(res, cp0)

    def test_unary_abs1(self):
        # test axpby with nested BlockDataContainer
        cp0 = self.cp0
        cp1 = cp0.abs()
        res = BlockDataContainer(self.ig0.allocate(1), self.ig1.allocate(1))
        self.assertBlockDataContainerAlmostEqual(res, cp1)

    def test_sapyb_a_blockdc(self):
        # test axpby between BlockDataContainers, with a as a blockdatacontainer

        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        data1 = ig0.allocate(2)
        data3 = ig0.allocate(3)

        a1 = ig0.allocate(3)
        a2 = ig0.allocate(2)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(data1,data3)
        a = BlockDataContainer(a1,a2)

        out = cp0 * 0. - 10

        # cp0.axpby(a,-2,cp1,out, num_threads=4)
        cp0.sapyb(a, cp1,-2,out, num_threads=1)

        # operation should be [  3 * -1 + (-2) * 2 , 2 * 1 + (-2) * 3 ]
        # output should be [ -7 , -4 ]
        res0 = ig0.allocate(-7)
        res2 = ig0.allocate(-4)
        res = BlockDataContainer(res0, res2)

        self.assertBlockDataContainerEqual(out, res)


    def test_sapyb_b_blockdc(self):
        # test axpby between BlockDataContainers, with b as a blockdatacontainer

        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        data1 = ig0.allocate(2)
        data3 = ig0.allocate(3)

        b1 = ig0.allocate(-2)
        b2 = ig0.allocate(-3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(data1,data3)
        b = BlockDataContainer(b1,b2)

        out = cp0 * 0. - 10

        cp0.sapyb(3,cp1, b,out, num_threads=4)

        # operation should be [  3 * -1 + (-2) * 2 , 3 * 1 + (-3) * 3 ]
        # output should be [ -7 , -3 ]
        res0 = ig0.allocate(-7)
        res2 = ig0.allocate(-6)
        res = BlockDataContainer(res0, res2)

        self.assertBlockDataContainerEqual(out, res)

    def test_sapyb_ab_blockdc(self):
        # test axpby between BlockDataContainers, with a and b as a blockdatacontainer

        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        data1 = ig0.allocate(2)
        data3 = ig0.allocate(3)

        a1 = ig0.allocate(3)
        a2 = ig0.allocate(2)

        b1 = ig0.allocate(-2)
        b2 = ig0.allocate(-3)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(data1,data3)
        a = BlockDataContainer(a1,a2)
        b = BlockDataContainer(b1,b2)

        out = cp0 * 0. - 10

        cp0.sapyb(a,cp1,b,out, num_threads=4)

        # operation should be [  3 * -1 + (-2) * 2 , 2 * 1 + (-3) * 3 ]
        # output should be [ -7 , -7 ]
        res0 = ig0.allocate(-7)
        res2 = ig0.allocate(-7)
        res = BlockDataContainer(res0, res2)

        self.assertBlockDataContainerEqual(out, res)


    def test_sapyb_ab_blockdc_y_dc(self):
        # test axpby between BlockDataContainers, with a and b as a blockdatacontainer, and y as a dc

        ig0 = ImageGeometry(2,3,4)

        data0 = ig0.allocate(-1)
        data2 = ig0.allocate(1)

        data1 = ig0.allocate(2)

        a1 = ig0.allocate(3)
        a2 = ig0.allocate(2)

        b1 = ig0.allocate(-2)
        b2 = ig0.allocate(-3)

        cp0 = BlockDataContainer(data0,data2)

        a = BlockDataContainer(a1,a2)
        b = BlockDataContainer(b1,b2)

        out = cp0 * 0. - 10

        cp0.sapyb(a,data1,b,out, num_threads=4)

        # operation should be [  3 * -1 + (-2) * 2 , 2 * 1 + (-3) * 2 ]
        # output should be [ -7 , -4 ]
        res0 = ig0.allocate(-7)
        res2 = ig0.allocate(-4)
        res = BlockDataContainer(res0, res2)

        self.assertBlockDataContainerEqual(out, res)

    def test_iterator(self):
        ig0 = VectorGeometry(5)
        data0=ig0.allocate(0)
        data1=ig0.allocate(1)
        data2=ig0.allocate(2)
        container=BlockDataContainer(data0,data1,data2)
        a=[]
        for data in container:
            a.append(list(data.array))
        self.assertListEqual(a, [[0.,0.,0.,0.,0.],[1.,1.,1.,1.,1.],[2,2,2,2,2]])

        a=[] #check it works a second time!
        for data in container:
            a.append(list(data.array))
        self.assertListEqual(a, [[0.,0.,0.,0.,0.],[1.,1.,1.,1.,1.],[2,2,2,2,2]])


class TestBlockGeometry(unittest.TestCase):
    def setUp(self):
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(0,360, 9))

        self.ig = BlockGeometry( *[ AG.copy() for i in range(9) ] )

    def test_block_geometry(self):
        for ig in self.ig:
            assert type(ig) == AcquisitionGeometry


class TestAcquisitionDataPartition(unittest.TestCase):
    def setUp(self):
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(9))

        data = AG.allocate(None)
        for i in range(AG.num_projections):
            data.array[i] = i
        self.data = data

    def test_partition(self):
        # data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        # self.data = data.get_slice(vertical='centre')
        # split in num_batches
        num_batches = 4

        #Testing sequential partitioning
        data = self.data.partition(num_batches, 'sequential')
        idxs = self.data._partition_indices(num_batches, indices=self.data.geometry.num_projections, stagger=False)
        assert len(data.containers) == num_batches
        self.assertDataIsTheSame(data, idxs)

        #Testing staggered partitioning
        data = self.data.partition(num_batches, Partitioner.STAGGERED)
        idxs = self.data._partition_indices(num_batches, indices=self.data.geometry.num_projections, stagger=True)
        assert len(data.containers) == num_batches
        self.assertDataIsTheSame(data, idxs)

    def test_partition_diff_num_batches(self):

        #Check what happens when the number of batches is equal to the number of projection angles
        num_batches=9
        data = self.data.partition(num_batches, 'sequential')
        idxs = self.data._partition_indices(num_batches, indices=self.data.geometry.num_projections, stagger=False)
        assert len(data.containers) == num_batches
        self.assertDataIsTheSame(data, idxs, msg='Failed when num_batches=number of projections')

        #Check what happens when the number of batches is one, the whole set of projection angles
        num_batches=1
        data = self.data.partition(num_batches, 'sequential')
        idxs = self.data._partition_indices(num_batches, indices=self.data.geometry.num_projections, stagger=False)
        assert len(data.containers) == num_batches
        self.assertDataIsTheSame(data, idxs, msg="Failed when num_batches=1")

        #Check what happens when the number of batches is zero
        num_batches=0
        with self.assertRaises(ZeroDivisionError):
            data = self.data.partition(num_batches, 'sequential')

       #Check what happens when the number of batches is greater than the number of projection angles
        num_batches=10
        with self.assertRaises(ValueError):
            data = self.data.partition(num_batches, 'sequential')






    def assertDataIsTheSame(self, data, idxs, msg=None):
        # let's check that the data is the same
        k = 0
        wrong = 0
        for i, el in enumerate(data):
            if len(el.shape)>1:
                j_range=el.shape[0]
            else:
                j_range=1
            for j in range(j_range):
                idx = idxs[i][j]
                try:
                    np.testing.assert_array_equal(el.as_array()[j], self.data.as_array()[idx], err_msg=msg)
                except AssertionError:
                    wrong += 1
                k += 1

        assert wrong == 0

class TestBlockDataContainerGeometry(BDCUnittest):
    def test_allocate(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(0)
        data2 = ig0.allocate(1)

        cp0 = BlockDataContainer(data0,data2)

        assert cp0.geometry == BlockGeometry(ig0, ig1)

        with self.assertRaises(ValueError):
            ig2 = ImageGeometry(2,3)
            cp1 = BlockDataContainer(data0, data2, ig2.allocate(0))
            assert cp0.geometry == BlockGeometry(ig0, ig1)
    
    def test_pnorm2(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(0)
        data2 = ig1.allocate(1)

        cp0 = BlockDataContainer(data0,data2)

        with self.assertRaises(ValueError):
            cp0.pnorm(2)

        cp0 = BlockDataContainer(data2,data2)
        self.assertAlmostEqual(cp0.pnorm(2), np.sqrt(2*3*5))


        assert cp0.pnorm(2) == np.sqrt(0**2 + 1**2)

    def test_equals(self):
        ig0 = ImageGeometry(2,3,4)
        ig1 = ImageGeometry(2,3,5)

        data0 = ig0.allocate(0)
        data2 = ig0.allocate(1)

        cp0 = BlockDataContainer(data0,data2)
        cp1 = BlockDataContainer(data0,data2)

        assert cp0.geometry == cp1.geometry

        cp1 = BlockDataContainer(data2,data2,data2)

        assert cp0 != cp1