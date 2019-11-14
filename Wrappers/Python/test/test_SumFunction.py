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

from ccpi.optimisation.functions import L1Norm, ScaledFunction, \
                                        LeastSquares, MixedL21Norm, L2NormSquared, \
                                        KullbackLeibler, ZeroFunction, ConstantFunction
from ccpi.optimisation.operators import Identity                                        
from ccpi.framework import ImageGeometry, BlockGeometry

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
        
    def test_SumFunction(self):
        
        M, N, K = 3,4,5
        ig = ImageGeometry(M, N, K)
        bg = BlockGeometry(ig, ig)
        
        tmp = ig.allocate('random_int')
        bg_tmp = bg.allocate('random_int')
        b = ig.allocate('random_int')
        
        operator = Identity(ig)

        scalar = 0.25
        f1 = L2NormSquared()
        f2 = L1Norm()
        f3 = scalar * L2NormSquared()
        f4 = scalar * L1Norm()
        f5 = scalar * L2NormSquared(b=b)
        f6 = scalar * L1Norm(b=b)  
        f7 = ZeroFunction()
        f8 = 5 *  ConstantFunction(10)             
        f9 = LeastSquares(operator, b, c=scalar)
        f10 = 0.5 * MixedL21Norm()
        
        list1 = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
                
        print('###################  Check sum of two functions ################## \n')
        
        for func in list1:
               
            
            # check sum of two functions   
            
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__

            sumf = f1 + func           
            self.assertNumpyArrayAlmostEqual(sumf(tmp), f1(tmp) + func(tmp) )
            print('{} = ( {} + {} ) is OK'.format(type(sumf).__name__, type(f1).__name__, type_fun))
            
            sumf1 = func + f1 
            self.assertNumpyArrayAlmostEqual(sumf1(tmp), func(tmp) + f1(tmp))
            print('Checking commutative')
            print('{} + ( {} + {} ) is OK\n'.format(type(sumf1).__name__, type_fun, type(f1).__name__))
            
        print('###################  Lispchitz constant ################## \n')
        
        for func in list1:
            
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__            
               
            
            # check Lispchitz sum of two functions   
            
            sumf = f1 + func   
            
            try:
                sumf.L==f1.L + func.L
            except TypeError:
                print('Function {} has L = None'.format(type_fun))
                
        print('\n###################  Check Gradient ################## \n')   
              
              
        for func in list1:
              
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__
          
            sumf = f1 + func
            # check gradient          
            try:
                self.assertNumpyArrayAlmostEqual(sumf.gradient(tmp).as_array(), (f1.gradient(tmp) + func.gradient(tmp)).as_array())
            except NotImplementedError:
                print("{} is not differentiable".format(type_fun))
                
        print('\n###################  Check Gradient Out ################## \n')  
         
        out_left = ig.allocate()
        out_right1 = ig.allocate()
        out_right2 = ig.allocate()  
            
        for func in list1:               
                
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__
            
            sumf = f1 + func
                                    
            # check gradient out    
            try:
                print('Check {} + {}\n'.format(type(f1).__name__, type_fun))
                sumf.gradient(tmp, out = out_left)
                f1.gradient(tmp, out = out_right1)
                func.gradient(tmp, out = out_right2)    
                self.assertNumpyArrayAlmostEqual(out_left.as_array(), out_right1.as_array() + out_right2.as_array())
            except NotImplementedError:
                print("{} is not differentiable".format(type_fun))  

