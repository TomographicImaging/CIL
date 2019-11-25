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

from ccpi.optimisation.functions import Function, L1Norm, ScaledFunction, \
                                        LeastSquares, L2NormSquared, \
                                        KullbackLeibler, ZeroFunction, ConstantFunction, TranslateFunction
from ccpi.optimisation.operators import Identity                                        
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
        tmp = ig.allocate('random_int')
        b = ig.allocate('random_int')
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
                                
            
if __name__ == '__main__':
#    
    t = TestFunction()
    t.test_TranslateFunction()    


#    ig = ImageGeometry(4,4)
#    tmp = ig.allocate('random_int')
#    b = ig.allocate('random_int')
#    scalar = 0.4
#            
##    f = scalar * L2NormSquared().centered_at(b)
##    print(f.function.function)
#    list1 = [ L2NormSquared(), scalar * L2NormSquared(), scalar * L2NormSquared(b=b)]                            
#        
##        for func in list_functions:
##            
###            if isinstance(func, ScaledFunction):
###                func_tmp = func.function
###            else:
###                func_tmp = func                
###                                
###            if func_tmp.b is None:
###                tmp_data = ig.allocate()
###            else:
###                tmp_data = b 
##            
##            func_tmp = func
##            tmp_data = ig.allocate()
##                                        
##            res1 = func_tmp(tmp)        
##            res2 = func_tmp.centered_at(tmp_data)(tmp)
##            
##            self.assertNumpyArrayAlmostEqual(res1, res2)
        
        

        
        
#        
#        for i in list_functions:
#            
#            print('Test Translation for Function {} '.format(type(i).__name__))
#            
#            if isinstance(i, L2NormSquared):
#                
#                f = L2NormSquared(b = b)    
#                g = TranslateFunction(L2NormSquared(), b)
#                
#            elif isinstance(i, L1Norm):
#                
#                f = L1Norm(b = b)    
#                g = TranslateFunction(L1Norm(), b)
#                
#            elif isinstance(i, ScaledFunction):
#    
#                if isinstance(i.function, L2NormSquared):
#                    f = scalar * L2NormSquared(b = b)    
#                    g = scalar * TranslateFunction(L2NormSquared(), b)
#                    
#                if isinstance(i.function, L1Norm):
#                    f = scalar * L1Norm(b = b)    
#                    g = scalar * TranslateFunction(L1Norm(), b)                
#                            
#            # check call
#            res1 = f(tmp)
#            res2 = g(tmp)    
#            numpy.testing.assert_equal(res1, res2)
#            
#            # check gradient
#                
#            if not isinstance(i, L1Norm):
#            
#                res1 = f.gradient(tmp)
#                res2 = g.gradient(tmp) 
#                numpy.testing.assert_equal(res1.as_array(), res2.as_array()) 
#            
#                # check gradient out
#                res3 = ig.allocate()
#                res4 = ig.allocate()
#                f.gradient(tmp, out = res3)
#                g.gradient(tmp, out = res4)
#                numpy.testing.assert_equal(res3.as_array(), res4.as_array())
#            
#            # check convex conjugate
#            res1 = f.convex_conjugate(tmp)
#            res2 = g.convex_conjugate(tmp)
#            numpy.testing.assert_equal(res1, res2) 
#            
#            # check proximal    
#            tau = 0.5
#            res1 = f.proximal(tmp, tau)
#            res2 = g.proximal(tmp, tau)
#            numpy.testing.assert_equal(res1.as_array(), res2.as_array()) 
#            
#            # check proximal out           
#            res3 = ig.allocate()
#            res4 = ig.allocate()
#            f.proximal(tmp, tau, out = res3)
#            g.proximal(tmp, tau, out = res4)
#            numpy.testing.assert_array_almost_equal(res3.as_array(), res4.as_array(),decimal = decimal)     
#            
#            # check proximal conjugate  
#            tau = 0.4
#            res1 = f.proximal_conjugate(tmp, tau)
#            res2 = g.proximal_conjugate(tmp, tau)
#            numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(),decimal = decimal)  
#                                
#            # check proximal out           
#            res3 = ig.allocate()
#            res4 = ig.allocate()
#            f.proximal_conjugate(tmp, tau, out = res3)
#            g.proximal_conjugate(tmp, tau, out = res4)
#            numpy.testing.assert_array_almost_equal(res3.as_array(), res4.as_array(),decimal = decimal)          
#            
#            
#            f = L2NormSquared() + 1
#            print(f(tmp))
#            
#                
       
#            
#        
#    #    tau = 0.5    
#    #    f = L2NormSquared(b=b) 
#    #    g = TranslateFunction(f, b)
#    #    res1 = f.proximal_conjugate(tmp, tau)    
#    #    res2 = tmp - tau * f.proximal(tmp/tau, 1/tau)
#    #    res3 = g.proximal_conjugate(tmp, tau)
#        
#    #    print(res1.as_array())
#    #    print(res3.as_array())
#    #    numpy.testing.assert_equal(res1.as_array(), res2.as_array()) 
#    #    numpy.testing.assert_equal(res1.as_array(), res3.as_array()) 
#    
#                    
#    
