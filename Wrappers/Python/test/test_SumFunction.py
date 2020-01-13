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
                                        LeastSquares, L2NormSquared, \
                                        KullbackLeibler, ZeroFunction, ConstantFunction
from ccpi.optimisation.operators import Identity                                        
from ccpi.framework import ImageGeometry

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
        
    def test_SumFunction(self):
        
        M, N, K = 3,4,5
        ig = ImageGeometry(M, N, K)
        
        tmp = ig.allocate('random_int')
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
        f10 = 0.5*KullbackLeibler(b=b)
        f11 = KullbackLeibler(b=b)
        f12 = 10
        
#        f10 = 0.5 * MixedL21Norm()
#        f11 = IndicatorBox(lower=0)
        
        list1 = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
                
        print('###################  Check sum of two functions ################## \n')
        
        for func in list1:
               
            
            # check sum of two functions   
            
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__
                
            if isinstance(func, Number):
                tmp_fun_eval = func
            else:
                tmp_fun_eval = func(tmp)                
                             
            sumf = f1 + func           
            self.assertNumpyArrayAlmostEqual(sumf(tmp), f1(tmp) + tmp_fun_eval )
            print('{} = ( {} + {} ) is OK'.format(type(sumf).__name__, type(f1).__name__, type_fun))
            
            sumf1 = func + f1 
            self.assertNumpyArrayAlmostEqual(sumf1(tmp), tmp_fun_eval + f1(tmp))
            print('Checking commutative')
            print('{} + ( {} + {} ) is OK\n'.format(type(sumf1).__name__, type_fun, type(f1).__name__))
            
        print('###################  Check Lispchitz constant ################## \n')
        
        for func in list1:
            
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__            
               
            
            # check Lispchitz sum of two functions  
            
            if isinstance(func, Number):
                tmp_fun_L = 0
            else:
                tmp_fun_L = func.L           
            
            sumf = f1 + func   
            
            try:
                sumf.L==f1.L + tmp_fun_L
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
                if isinstance(func, Number):
                    tmp_fun_gradient = 0
                else:
                    tmp_fun_gradient = func.gradient(tmp)   
                    
                self.assertNumpyArrayAlmostEqual(sumf.gradient(tmp).as_array(), (f1.gradient(tmp) + tmp_fun_gradient).as_array())
            except NotImplementedError:
                print("{} is not differentiable".format(type_fun))
                
        print('\n###################  Check Gradient Out ################## \n')  
         
        out_left = ig.allocate()
        out_right1 = ig.allocate()
        out_right2 = ig.allocate()  
            
        for i, func in enumerate(list1):               
                
            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__
            
            sumf = f1 + func
            
                                    
            # check gradient out    
            try:
                
                
                if isinstance(func, Number):
                    tmp_fun_gradient_out = 0
                else:
                    func.gradient(tmp, out = out_right2) 
                    tmp_fun_gradient_out = out_right2.as_array()
                    
                #print('Check {} + {}\n'.format(type(f1).__name__, type_fun))
                sumf.gradient(tmp, out = out_left)
                f1.gradient(tmp, out = out_right1)   
                self.assertNumpyArrayAlmostEqual(out_left.as_array(), out_right1.as_array() + tmp_fun_gradient_out)
            except NotImplementedError:
                print("{} is not differentiable".format(type_fun))  
    def test_ConstantFunction(self):

        k = ConstantFunction(constant=1)
        ig = ImageGeometry(1,2,3)
        x = ig.allocate(2)

        grad = k.gradient(x)
        out = ig.allocate(-1)

        k.gradient(x, out=out)
        #out.fill(-3)
        
        self.assertNumpyArrayEqual(numpy.zeros(x.shape), grad.as_array())
        
        self.assertNumpyArrayEqual(out.as_array(), grad.as_array())
                
    def test_SumFunctionScalar(self):      
        
        M, N, K = 3,4,5
        ig = ImageGeometry(M, N, K)
        
        tmp = ig.allocate('random_int')
        b = ig.allocate('random_int')
        
        scalar = 0.25
        f1 =  scalar * L2NormSquared(b=b)
        f2 = 5
           
        f = f1 + f2
        print(f)
        
        g = f2 + f1
        print(g)

        tau = 0.03
        
        # check call method       
        res1 = f(tmp)
        res2 = f1(tmp) + f2
        self.assertAlmostEqual(res1, res2)
        
        # check gradient
        res1 = f.gradient(tmp)
        res2 = f1.gradient(tmp)
        self.assertNumpyArrayAlmostEqual(res1.as_array(), res2.as_array())
        
        # check gradient with out
        out1 = tmp*0
        out2 = tmp*0
        f.gradient(tmp, out=out1)
        f1.gradient(tmp, out=out2)
        self.assertNumpyArrayAlmostEqual(out1.as_array(), out2.as_array())
        

        res1 = f.proximal(tmp, tau)
        res2 = f1.proximal(tmp, tau)
        
        # proximal of sum of function with scalar = proximal of function
        self.assertNumpyArrayAlmostEqual(res1.as_array(), res2.as_array())
        
        # proximal  with out of sum of function with scalar = proximal of function
        res1_out = ig.allocate()
        res2_out = ig.allocate()         
        f.proximal(tmp, tau, out = res1_out)
        f1.proximal(tmp, tau, out = res2_out)             
        self.assertNumpyArrayAlmostEqual(res1_out.as_array(), res2_out.as_array())

        res1 = f.proximal_conjugate(tmp, tau)
        res2 = f1.proximal_conjugate(tmp, tau)
        
        # proximal of sum of function with scalar = proximal of function
        self.assertNumpyArrayAlmostEqual(res1.as_array(), res2.as_array())
        
        # proximal  with out of sum of function with scalar = proximal of function
        res1_out = ig.allocate()
        res2_out = ig.allocate()         
        f.proximal_conjugate(tmp, tau, out = res1_out)
        f1.proximal_conjugate(tmp, tau, out = res2_out)             
        self.assertNumpyArrayAlmostEqual(res1_out.as_array(), res2_out.as_array())        
            
        
def test_ConstantFunction(self):      
        
        M, N, K = 3,4,5
        ig = ImageGeometry(M, N, K)
        
        tmp = ig.allocate('random_int')
        
        constant = 10
        f = ConstantFunction(constant)
        
        # check call
        res1 = f(tmp)
        self.assertAlmostEqual(res1, constant)
        
        # check gradient with and without out
        res1 = f.gradient(tmp)
        out = ig.allocate()
        self.assertNumpyArrayAlmostEqual(res1.as_array(), out)
        
        out1 = ig.allocate()
        f.gradient(tmp, out=out1)
        self.assertNumpyArrayAlmostEqual(res1.as_array(), out1)
        
        # check convex conjugate        
        res1 = f.convex_conjugate(tmp)
        res2 = tmp.maximum(0).sum()
        self.assertNumpyArrayAlmostEqual(res1.as_array(), res2.as_array())
        
        # check proximal 
        tau = 0.4
        res1 = f.proximal(tmp, tau)
        self.assertNumpyArrayAlmostEqual(res1.as_array(), tmp.as_array()) 
            
        
        
        # check call
        
        
        
     
        
                
#if __name__ == '__main__':
#    
#    t = TestFunction()
#    t.test_SumFunction()
#    t.test_SumFunctionScalar()

                

