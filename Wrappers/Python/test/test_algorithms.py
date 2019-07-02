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
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from ccpi.optimisation.operators import Identity
from ccpi.optimisation.functions import Norm2Sq, ZeroFunction, \
   L2NormSquared, FunctionOperatorComposition
from ccpi.optimisation.algorithms import GradientDescent
from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.algorithms import FISTA




class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        #wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        #self.filename = '24737_fd.nxs'
        # we use Identity as the operator and solve the simple least squares 
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
        
        identity = Identity(ig)
        
        norm2sq = Norm2Sq(identity, b)
        rate = 0.3
        rate = norm2sq.L / 3.
        
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              rate=rate)
        alg.max_iteration = 20
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
    def test_CGLS(self):
        print ("Test CGLS")
        ig = ImageGeometry(124,153,154)
        x_init = ImageData(geometry=ig)
        x_init = ig.allocate()
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        b = ig.allocate('random')
        identity = Identity(ig)
        
        alg = CGLS(x_init=x_init, operator=identity, data=b)
        alg.max_iteration = 200
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        
    def test_FISTA(self):
        print ("Test FISTA")
        ig = ImageGeometry(127,139,149)
        x_init = ImageData(geometry=ig)
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        x_init = ig.allocate(ImageGeometry.RANDOM)
        identity = Identity(ig)
        
	#### it seems FISTA does not work with Nowm2Sq
        # norm2sq = Norm2Sq(identity, b)
        # norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        norm2sq = FunctionOperatorComposition(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt':False}
        print ("initial objective", norm2sq(x_init))
        alg = FISTA(x_init=x_init, f=norm2sq, g=ZeroFunction())
        alg.max_iteration = 2
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
 
