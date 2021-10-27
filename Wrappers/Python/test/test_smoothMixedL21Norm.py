# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.framework import ImageGeometry, BlockGeometry
from cil.optimisation.functions import SmoothMixedL21Norm, MixedL21Norm
import unittest
import numpy 
                    
class TestFunction(unittest.TestCase):
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

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
            print("expected " , second)
            print("actual " , first)

        self.assertTrue(res)
        
    def test_smoothL21Norm(self):
        
        ig = ImageGeometry(4, 5)
        bg = BlockGeometry(ig, ig)
        
        epsilon = 0.5
        
        
        
        f1 = SmoothMixedL21Norm(epsilon)    
        x = bg.allocate('random', seed=10)
        
        
        print("Check call for smooth MixedL21Norm")
        
        # check call
        res1 = f1(x)        
        res2 = (x.pnorm(2)**2 + epsilon**2).sqrt().sum()

    
        # alternative        
        tmp1 = x.copy()
        tmp1.containers += (epsilon,)        
        res3 = tmp1.pnorm(2).sum()
                        
        numpy.testing.assert_almost_equal(res1, res2, decimal=5) 
        numpy.testing.assert_almost_equal(res1, res3, decimal=5) 
        
        print("Check gradient for smooth MixedL21Norm ... OK ")        
        
        res1 = f1.gradient(x)
        res2 = x.divide((x.pnorm(2)**2 + epsilon**2).sqrt())
        numpy.testing.assert_array_almost_equal(res1.get_item(0).as_array(), 
                                                res2.get_item(0).as_array()) 
        
        numpy.testing.assert_array_almost_equal(res1.get_item(1).as_array(), 
                                                res2.get_item(1).as_array()) 
        
        # check with MixedL21Norm, when epsilon close to 0
        
        print("Check as epsilon goes to 0 ... OK") 
        
        f1 = SmoothMixedL21Norm(1e-12)   
        f2 = MixedL21Norm()
        
        res1 = f1(x)
        res2 = f2(x)
        numpy.testing.assert_almost_equal(f1(x), f2(x)) 
        
         


        