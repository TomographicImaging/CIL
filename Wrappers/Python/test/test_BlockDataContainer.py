# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest
import numpy
from ccpi.framework import ImageGeometry, AcquisitionGeometry
from ccpi.framework import ImageData, AcquisitionData
from ccpi.framework import BlockDataContainer, DataContainer
import functools

from ccpi.optimisation.operators import Gradient, Identity, BlockOperator

class TestBlockDataContainer(unittest.TestCase):
    def skiptest_BlockDataContainerShape(self):
        print ("test block data container")
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
        print ("test block data container")
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
        print ("test block data container")
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
            print (ve)
            self.assertTrue(True)
        d = cp2 - data0
        self.assertEqual(d.get_item(0).as_array()[0][0][0], 1) 
        try:
            d = cp2 - data1
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)
        d = cp2 * data2
        self.assertEqual(d.get_item(0).as_array()[0][0][0], 2) 
        try:
            d = cp2 * data1
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)
            
        a = [ (el, ot) for el,ot in zip(cp0.containers,cp1.containers)]
        print  (a[0][0].shape)
        #cp2 = BlockDataContainer(*a)
        cp2 = cp0.add(cp1)
        self.assertEqual (cp2.get_item(0).as_array()[0][0][0] , 2.)
        self.assertEqual (cp2.get_item(1).as_array()[0][0][0] , 4.)
        
        cp2 = cp0 + cp1 
        self.assertTrue (cp2.get_item(0).as_array()[0][0][0] == 2.)
        self.assertTrue (cp2.get_item(1).as_array()[0][0][0] == 4.)
        cp2 = cp0 + 1 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = cp0 + [1 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 3., decimal = 5)
        cp2 += cp1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , +3. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +6., decimal = 5)
        
        cp2 += 1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , +4. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +7., decimal = 5)
        
        cp2 += [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 2. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 6., decimal = 5)
        
        
        cp2 = cp0.subtract(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == -2.)
        assert (cp2.get_item(1).as_array()[0][0][0] == -2.)
        cp2 = cp0 - cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == -2.)
        assert (cp2.get_item(1).as_array()[0][0][0] == -2.)
        
        cp2 = cp0 - 1 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0, decimal = 5)
        cp2 = cp0 - [1 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -1., decimal = 5)
        
        cp2 -= cp1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -3. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -4., decimal = 5)
        
        cp2 -= 1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -4. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -5., decimal = 5)
        
        cp2 -= [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -2. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -4., decimal = 5)
        
        
        cp2 = cp0.multiply(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        assert (cp2.get_item(1).as_array()[0][0][0] == 3.)
        cp2 = cp0 * cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        assert (cp2.get_item(1).as_array()[0][0][0] == 3.)
        
        cp2 = cp0 * 2 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2, decimal = 5)
        cp2 = 2 * cp0  
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2, decimal = 5)
        cp2 = cp0 * [3 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = cp0 * numpy.asarray([3 ,2])
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        
        cp2 = [3,2] * cp0 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = numpy.asarray([3,2]) * cp0 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
        
        try:
            cp2 = [3,2,3] * cp0 
            #numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
            #numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 2., decimal = 5)
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)
        cp2 *= cp1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0 , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +6., decimal = 5)
        
        cp2 *= 1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , +6., decimal = 5)
        
        cp2 *= [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -6., decimal = 5)
        
        try:
            cp2 *= [2,3,5]
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)
        
        cp2 = cp0.divide(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1./3., decimal=4)
        cp2 = cp0/cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1./3., decimal=4)
        
        cp2 = cp0 / 2 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        cp2 = cp0 / [3 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        cp2 = cp0 / numpy.asarray([3 ,2])
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        cp3 = numpy.asarray([3 ,2]) / (cp0+1)
        numpy.testing.assert_almost_equal(cp3.get_item(0).as_array()[0][0][0] , 3. , decimal=5)
        numpy.testing.assert_almost_equal(cp3.get_item(1).as_array()[0][0][0] , 1, decimal = 5)
        
        cp2 += 1
        cp2 /= cp1
        # TODO fix inplace division
         
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 1./2 , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 1.5/3., decimal = 5)
        
        cp2 /= 1
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0.5 , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 0.5, decimal = 5)
        
        cp2 /= [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , -0.5/2. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , -0.5, decimal = 5)
        ####
        
        cp2 = cp0.power(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1., decimal=4)
        cp2 = cp0**cp1
        assert (cp2.get_item(0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1., decimal=4)
        
        cp2 = cp0 ** 2 
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0] , 0., decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0] , 1., decimal = 5)
        
        cp2 = cp0.maximum(cp1)
        assert (cp2.get_item(0).as_array()[0][0][0] == cp1.get_item(0).as_array()[0][0][0])
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], cp2.get_item(1).as_array()[0][0][0], decimal=4)
        
        
        cp2 = cp0.abs()
        numpy.testing.assert_almost_equal(cp2.get_item(0).as_array()[0][0][0], 0., decimal=4)
        numpy.testing.assert_almost_equal(cp2.get_item(1).as_array()[0][0][0], 1., decimal=4)
        
        cp2 = cp0.subtract(cp1)
        s = cp2.sign()
        numpy.testing.assert_almost_equal(s.get_item(0).as_array()[0][0][0], -1., decimal=4)
        numpy.testing.assert_almost_equal(s.get_item(1).as_array()[0][0][0], -1., decimal=4)
        
        cp2 = cp0.add(cp1)
        s = cp2.sqrt()
        numpy.testing.assert_almost_equal(s.get_item(0).as_array()[0][0][0], numpy.sqrt(2), decimal=4)
        numpy.testing.assert_almost_equal(s.get_item(1).as_array()[0][0][0], numpy.sqrt(4), decimal=4)
        
        s = cp0.sum()
        size = functools.reduce(lambda x,y: x*y, data1.shape, 1)
        print ("size" , size)
        numpy.testing.assert_almost_equal(s, 0 + size, decimal=4)
        s0 = 1
        s1 = 1
        for i in cp0.get_item(0).shape:
            s0 *= i
        for i in cp0.get_item(1).shape:
            s1 *= i
            
        #numpy.testing.assert_almost_equal(s[1], cp0.get_item(0,0).as_array()[0][0][0]*s0 +cp0.get_item(1,0).as_array()[0][0][0]*s1, decimal=4)
    def test_Nested_BlockDataContainer(self):
        print ("test_Nested_BlockDataContainer")
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
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 2. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 3. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 5. , decimal=5)

        nbdc2 = 2 + nbdc
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 2. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 3. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 5. , decimal=5)


        nbdc2 = nbdc * 2 
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 2. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 6. , decimal=5)

        nbdc2 = 2 * nbdc
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , 2. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 4. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 6. , decimal=5)

        nbdc2 = nbdc / 2 
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(0).get_item(1).as_array()[0][0][0] , .5 , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(0).as_array()[0][0][0] , 1. , decimal=5)
        numpy.testing.assert_almost_equal(nbdc2.get_item(1).get_item(1).as_array()[0][0][0] , 3./2 , decimal=5)

        c5 = nbdc.get_item(0).power(2).sum()
        c5a = nbdc.power(2).sum()
        print ("sum", c5a, c5)
        
        cp0 = BlockDataContainer(data0,data2)
        a = cp0 * data2
        b = data2 * cp0
        self.assertBlockDataContainerEqual(a,b)
        

        print ("test_Nested_BlockDataContainer OK")
    def stest_NestedBlockDataContainer2(self):
        M, N = 2, 3
        ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        ag = ig
        u = ig.allocate(1)
        op1 = Gradient(ig)
        op2 = Identity(ig, ag)

        operator = BlockOperator(op1, op2, shape=(2,1)) 

        d1 = op1.direct(u)
        d2 = op2.direct(u)

        d = operator.direct(u)

        dd = operator.domain_geometry()
        ww = operator.range_geometry()

        print(d.get_item(0).get_item(0).as_array())
        print(d.get_item(0).get_item(1).as_array())
        print(d.get_item(1).as_array())

        c1 = d + d

        c2 = 2*d

        c3 = d / (d+0.0001)


        c5 = d.get_item(0).power(2).sum()

    def test_BlockDataContainer_fill(self):
        print ("test block data container")
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
        self.assertNumpyArrayEqual(data0.as_array(), data2.as_array())
        data0 = ImageData(geometry=ig0)

        for el,ot in zip(cp0, cp2):
            print (el.shape, ot.shape)
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


    def assertBlockDataContainerEqual(self, container1, container2):
        print ("assert Block Data Container Equal")
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if issubclass(container1.get_item(col).__class__, DataContainer):
                print ("Checking col ", col)
                self.assertNumpyArrayEqual(
                    container1.get_item(col).as_array(), 
                    container2.get_item(col).as_array()
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),container2.get_item(col))

    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)    


