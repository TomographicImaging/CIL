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
from __future__ import division

from ccpi.optimisation.functions import Function, L1Norm, ScaledFunction, \
                                        LeastSquares, L2NormSquared, \
                                        KullbackLeibler, ZeroFunction, \
                                        ConstantFunction, TranslateFunction, \
                                        MixedL21Norm, FunctionOperatorComposition
from ccpi.optimisation.operators import Identity, Gradient                              
from ccpi.framework import ImageGeometry, BlockGeometry

import unittest
import numpy
from numbers import Number


''' Here we test SumFunction class for different function

L2Norm, L1Norm, KullbackLeibler, ZeroFunction, ConstantFunction, Scalar

for call method
for gradient method



'''



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
        
    def test_TranslateFunction(self):   
        
        # Test TranslationFunction
    
        ig = ImageGeometry(4,4)
        tmp = ig.allocate('random')
        b = ig.allocate('random')
        scalar = 0.4
        tau = 0.05
                                
        list1 = [ L2NormSquared(), scalar * L2NormSquared(), scalar * L2NormSquared(b=b), 
                 L1Norm(), scalar * L1Norm(), scalar * L1Norm(b=b)]
        
        list1_shift = [ L2NormSquared().centered_at(ig.allocate()), scalar * L2NormSquared().centered_at(ig.allocate()), scalar * L2NormSquared().centered_at(b),
                        L1Norm().centered_at(ig.allocate()), scalar * L1Norm().centered_at(ig.allocate()), scalar * L1Norm().centered_at(b)]        
        
        out_gradient1 = ig.allocate()
        out_gradient2 = ig.allocate()
        
        out_proximal1 = ig.allocate()
        out_proximal2 = ig.allocate()     
        
        out_proximal_conj1 = ig.allocate()
        out_proximal_conj2 = ig.allocate()             
        
        for func, func_shift in zip(list1, list1_shift):

            # check call
            res1 = func(tmp)
            res2 = func_shift(tmp)
            self.assertNumpyArrayAlmostEqual(res1, res2)
            
            try:
                # check gradient
                res1_gradient = func.gradient(tmp)
                res2_gradient = func_shift.gradient(tmp)
                self.assertNumpyArrayAlmostEqual(res1_gradient.as_array(), res2_gradient.as_array()) 
                
                # check gradient out
                func.gradient(tmp, out = out_gradient1)
                func_shift.gradient(tmp, out = out_gradient2)
                self.assertNumpyArrayAlmostEqual(out_gradient1.as_array(), out_gradient2.as_array())
                
            except NotImplementedError:
                print('Function is not differentiable')
                             
            # check proximal     
            func.proximal(tmp, tau, out = out_proximal1)
            func_shift.proximal(tmp, tau, out = out_proximal2)
            self.assertNumpyArrayAlmostEqual(out_proximal1.as_array(), out_proximal2.as_array())  

            # check proximal conjugate
            func.proximal_conjugate(tmp, tau, out = out_proximal_conj1)
            func_shift.proximal_conjugate(tmp, tau, out = out_proximal_conj2)
            self.assertNumpyArrayAlmostEqual(out_proximal_conj1.as_array(), out_proximal_conj1.as_array())               
            
                
    def test_TranslateFunction_MixedL21Norm(self):    
        
        ig = ImageGeometry(4,4)
        
        Grad = Gradient(ig)
        
        b = Grad.range_geometry().allocate('random_int')
                
        alpha = 0.4
        fun = alpha * TranslateFunction(MixedL21Norm(), b)
        
        tmp_x = Grad.range_geometry().allocate('random_int')
        
        res1 = fun(tmp_x)        
        res2 = alpha * (tmp_x - b).pnorm(2).sum()        
        self.assertAlmostEqual(res1, res2)
        print("Check call...OK")
        
        # TODO
        print("Missing convex conj, prox, prox_conj")
        
                                     
            
if __name__ == '__main__':
 
    t = TestFunction()
    t.test_TranslateFunction() 
    t.test_TranslateFunction_MixedL21Norm()

