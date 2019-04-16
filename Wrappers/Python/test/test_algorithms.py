# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:11:36 2019

@author: ofn77899
"""

import unittest
import numpy
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from ccpi.optimisation.ops import TomoIdentity
from ccpi.optimisation.funcs import Norm2sq
from ccpi.optimisation.algorithms import GradientDescent
from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.algorithms import FISTA
from ccpi.optimisation.algorithms import FBPD




class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        #wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        #self.filename = '24737_fd.nxs'
        # we use TomoIdentity as the operator and solve the simple least squares 
        # problem for a random-valued ImageData or AcquisitionData b?  
        # Then we know the minimiser is b itself
        
        # || I x -b ||^2
        
        # create an ImageGeometry
        ig = ImageGeometry(12,13,14)
        pass

    def tearDown(self):
        #os.remove(self.filename)
        pass
    
    def test_GradientDescent(self):
        print ("Test GradientDescent")
        ig = ImageGeometry(12,13,14)
        x_init = ImageData(geometry=ig)
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        
        identity = TomoIdentity(geometry=ig)
        
        norm2sq = Norm2sq(identity, b)
        
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              rate=0.3)
        alg.max_iteration = 20
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
    def test_CGLS(self):
        print ("Test CGLS")
        ig = ImageGeometry(124,153,154)
        x_init = ImageData(geometry=ig)
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        
        identity = TomoIdentity(geometry=ig)
        
        alg = CGLS(x_init=x_init, operator=identity, data=b)
        alg.max_iteration = 1
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        
    def test_FISTA(self):
        print ("Test FISTA")
        ig = ImageGeometry(127,139,149)
        x_init = ImageData(geometry=ig)
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        x_init = ImageData(geometry=ig)
        x_init.fill(numpy.random.random(x_init.shape))
        
        identity = TomoIdentity(geometry=ig)
        
        norm2sq = Norm2sq(identity, b)
        norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        opt = {'tol': 1e-4, 'memopt':False}
        alg = FISTA(x_init=x_init, f=norm2sq, g=None, opt=opt)
        alg.max_iteration = 2
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
               

    
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
        
    
        


if __name__ == '__main__':
    unittest.main()
 