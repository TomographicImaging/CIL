# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:08:23 2019

@author: ofn77899
"""

import unittest
import numpy
#from ccpi.plugins.ops import CCPiProjectorSimple
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.optimisation.ops import TomoIdentity
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.framework import ImageGeometry, AcquisitionGeometry
from ccpi.framework import ImageData, AcquisitionData
#from ccpi.optimisation.algorithms import GradientDescent
from ccpi.framework import BlockDataContainer
#from ccpi.optimisation.Algorithms import CGLS
import functools

class TestBlockDataContainer(unittest.TestCase):
    def skiptest_BlockDataContainerShape(self):
        print ("test block data container")
        ig0 = ImageGeometry(2,3,4)
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
        ig1 = ImageGeometry(12,42,55,32)

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
        ig1 = ImageGeometry(12,42,55,32)
        
        data0 = ImageData(geometry=ig0)
        data1 = ImageData(geometry=ig1) + 1
        
        data2 = ImageData(geometry=ig0) + 2
        data3 = ImageData(geometry=ig1) + 3
        
        cp0 = BlockDataContainer(data0,data1)
        cp1 = BlockDataContainer(data2,data3)
    #    
        a = [ (el, ot) for el,ot in zip(cp0.containers,cp1.containers)]
        print  (a[0][0].shape)
        #cp2 = BlockDataContainer(*a)
        cp2 = cp0.add(cp1)
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 2.)
        assert (cp2.get_item(1,0).as_array()[0][0][0] == 4.)
        
        cp2 = cp0 + cp1 
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 2.)
        assert (cp2.get_item(1,0).as_array()[0][0][0] == 4.)
        cp2 = cp0 + 1 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = cp0 + [1 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 3., decimal = 5)
        cp2 += cp1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , +3. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +6., decimal = 5)
        
        cp2 += 1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , +4. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +7., decimal = 5)
        
        cp2 += [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 2. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 6., decimal = 5)
        
        
        cp2 = cp0.subtract(cp1)
        assert (cp2.get_item(0,0).as_array()[0][0][0] == -2.)
        assert (cp2.get_item(1,0).as_array()[0][0][0] == -2.)
        cp2 = cp0 - cp1
        assert (cp2.get_item(0,0).as_array()[0][0][0] == -2.)
        assert (cp2.get_item(1,0).as_array()[0][0][0] == -2.)
        
        cp2 = cp0 - 1 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0, decimal = 5)
        cp2 = cp0 - [1 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -1. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -1., decimal = 5)
        
        cp2 -= cp1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -3. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -4., decimal = 5)
        
        cp2 -= 1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -4. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -5., decimal = 5)
        
        cp2 -= [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -2. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -4., decimal = 5)
        
        
        cp2 = cp0.multiply(cp1)
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
        assert (cp2.get_item(1,0).as_array()[0][0][0] == 3.)
        cp2 = cp0 * cp1
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
        assert (cp2.get_item(1,0).as_array()[0][0][0] == 3.)
        
        cp2 = cp0 * 2 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2, decimal = 5)
        cp2 = 2 * cp0  
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2, decimal = 5)
        cp2 = cp0 * [3 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = cp0 * numpy.asarray([3 ,2])
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
        
        cp2 = [3,2] * cp0 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = numpy.asarray([3,2]) * cp0 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
        cp2 = [3,2,3] * cp0 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
        
        cp2 *= cp1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0 , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +6., decimal = 5)
        
        cp2 *= 1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +6., decimal = 5)
        
        cp2 *= [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -6., decimal = 5)
        
        
        cp2 = cp0.divide(cp1)
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1./3., decimal=4)
        cp2 = cp0/cp1
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1./3., decimal=4)
        
        cp2 = cp0 / 2 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
        cp2 = cp0 / [3 ,2]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
        cp2 = cp0 / numpy.asarray([3 ,2])
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
        cp3 = numpy.asarray([3 ,2]) / (cp0+1)
        numpy.testing.assert_almost_equal(cp3.get_item(0,0).as_array()[0][0][0] , 3. , decimal=5)
        numpy.testing.assert_almost_equal(cp3.get_item(1,0).as_array()[0][0][0] , 1, decimal = 5)
        
        cp2 += 1
        cp2 /= cp1
        # TODO fix inplace division
         
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 1./2 , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 1.5/3., decimal = 5)
        
        cp2 /= 1
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0.5 , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
        
        cp2 /= [-2,-1]
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -0.5/2. , decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -0.5, decimal = 5)
        ####
        
        cp2 = cp0.power(cp1)
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1., decimal=4)
        cp2 = cp0**cp1
        assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1., decimal=4)
        
        cp2 = cp0 ** 2 
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0., decimal=5)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 1., decimal = 5)
        
        cp2 = cp0.maximum(cp1)
        assert (cp2.get_item(0,0).as_array()[0][0][0] == cp1.get_item(0,0).as_array()[0][0][0])
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], cp2.get_item(1,0).as_array()[0][0][0], decimal=4)
        
        
        cp2 = cp0.abs()
        numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0], 0., decimal=4)
        numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1., decimal=4)
        
        cp2 = cp0.subtract(cp1)
        s = cp2.sign()
        numpy.testing.assert_almost_equal(s.get_item(0,0).as_array()[0][0][0], -1., decimal=4)
        numpy.testing.assert_almost_equal(s.get_item(1,0).as_array()[0][0][0], -1., decimal=4)
        
        cp2 = cp0.add(cp1)
        s = cp2.sqrt()
        numpy.testing.assert_almost_equal(s.get_item(0,0).as_array()[0][0][0], numpy.sqrt(2), decimal=4)
        numpy.testing.assert_almost_equal(s.get_item(1,0).as_array()[0][0][0], numpy.sqrt(4), decimal=4)
        
        s = cp0.sum()
        size = functools.reduce(lambda x,y: x*y, data1.shape, 1)
        print ("size" , size)
        numpy.testing.assert_almost_equal(s, 0 + size, decimal=4)
        s0 = 1
        s1 = 1
        for i in cp0.get_item(0,0).shape:
            s0 *= i
        for i in cp0.get_item(1,0).shape:
            s1 *= i
            
        #numpy.testing.assert_almost_equal(s[1], cp0.get_item(0,0).as_array()[0][0][0]*s0 +cp0.get_item(1,0).as_array()[0][0][0]*s1, decimal=4)
    