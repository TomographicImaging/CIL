# -*- coding: utf-8 -*-
#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
from cil.optimisation.functions import L1Norm, ScaledFunction, SumFunction,\
                                        LeastSquares, L2NormSquared, \
                                        KullbackLeibler, ZeroFunction, ConstantFunction
from cil.optimisation.operators import IdentityOperator                                        
from cil.framework import ImageGeometry, DataContainer
import numpy
import numpy as np
from numbers import Number
from testclass import CCPiTestClass

initialise_tests()

class TestFunction(CCPiTestClass):
            
        
    def setUp(self):   
        M, N, K = 3,4,5
        self.ig = ImageGeometry(M, N, K)
        
        self.x = self.ig.allocate('random', seed=1)
        self.b   = self.ig.allocate('random', seed=2)
        self.eta = self.ig.allocate(0.1)
        
        self.operator = IdentityOperator(self.ig)

        scalar = 0.25
        self.f1 = L2NormSquared()
        self.f2 = L1Norm()
        self.f3 = scalar * L2NormSquared()
        self.f4 = scalar * L1Norm()
        self.f5 = scalar * L2NormSquared(b=self.b)
        self.f6 = scalar * L1Norm(b=self.b)  
        self.f7 = ZeroFunction()
        self.f8 = 5 *  ConstantFunction(10)             
        self.f9 = LeastSquares(self.operator, self.b, c=scalar)
        self.f10 = 0.5*KullbackLeibler(b=self.b,eta = self.eta)
        self.f11 = KullbackLeibler(b=self.b, eta =self.eta)
        self.f12 = 10   

        self.list1 = [self.f1, self.f2, self.f3, self.f4, self.f5, \
                    self.f6, self.f7, self.f8, self.f9, self.f10, self.f11, self.f12]                  
        

    def test_SumFunction_call(self):
        for func in self.list1:

            if isinstance(func, ScaledFunction):
                type_fun = ' scalar * ' + type(func.function).__name__
            else:    
                type_fun = type(func).__name__            
                           
            # check sum of two functions   
                
            if isinstance(func, Number):
                tmp_fun_eval = func
            else:
                tmp_fun_eval = func(self.x)                
                             
            sumf = self.f1 + func        
            np.testing.assert_allclose( sumf(self.x), self.f1(self.x) + tmp_fun_eval ) 

            # check if the sum is SumFunction 
            self.assertIsInstance(sumf, SumFunction)
                        
            sumf1 = func + self.f1 
            np.testing.assert_allclose(sumf1(self.x), tmp_fun_eval + self.f1(self.x))

            # check if the sum is SumFunction 
            self.assertIsInstance(sumf1, SumFunction)                                


    def test_SumFunction_Lipschitz(self):        
        for func in self.list1:
                           
            try:
                # check Lispchitz sum of two functions  
                if isinstance(func, Number):
                    tmp_fun_L = 0
                else:
                    tmp_fun_L = func.L           
                
                sumf = self.f1 + func   
                
                try:
                    sumf.L==self.f1.L + tmp_fun_L
                except TypeError:
                    # print('Function {} has L = None'.format(type_fun))
                    pass
            except ValueError as nie:
                # print (func.__class__.__name__, nie)
                pass
                

    def test_SumFunction_gradient(self):              
        for func in self.list1:
            sumf = self.f1 + func
            # check gradient          
            try:
                if isinstance(func, Number):
                    tmp_fun_gradient = 0
                else:
                    tmp_fun_gradient = func.gradient(self.x)   
                    
                self.assertNumpyArrayAlmostEqual(sumf.gradient(self.x).as_array(), (self.f1.gradient(self.x) + tmp_fun_gradient).as_array())
            except NotImplementedError:
                # print("{} is not differentiable".format(type_fun))
                pass
                                
        # check gradient out 
        out_left = self.ig.allocate()
        out_right1 = self.ig.allocate()
        out_right2 = self.ig.allocate()  
            
        for func in self.list1:               
            sumf = self.f1 + func
            try:
                if isinstance(func, Number):
                    tmp_fun_gradient_out = 0
                else:
                    func.gradient(self.x, out = out_right2) 
                    tmp_fun_gradient_out = out_right2.as_array()
                    
                #print('Check {} + {}\n'.format(type(f1).__name__, type_fun))
                sumf.gradient(self.x, out = out_left)
                self.f1.gradient(self.x, out = out_right1)   
                self.assertNumpyArrayAlmostEqual(out_left.as_array(), out_right1.as_array() + tmp_fun_gradient_out)

            except NotImplementedError:
                # print("{} is not differentiable".format(type_fun))  
                pass


    def test_SumFunction_inputs(self):
        try:
            f = SumFunction(self.list1[0])
            self.assertFalse(True, "passed only one function and SumFunction accepted it!")
        except ValueError as ve:
            self.assertTrue(True, "Correctly failed" + str(ve))


    def test_SumFunction_more_inputs(self):
        # Test Lipshchitz value with more than 2 functions
        list2 = [LeastSquares(self.operator, self.b, c=0.25), 
                 LeastSquares(self.operator, self.b, c=4), 
                 LeastSquares(self.operator, self.b, c=5)]
        
        F = SumFunction(*list2)
        L = 0.
        for f in list2:
            L += f.L
                    
        self.assertAlmostEqual(L , F.L)

        # assert Lmax property
        self.assertAlmostEqual(max(f.L for f in list2) , F.Lmax)

        ## test value of the gradient
        out =  list2[0].gradient(self.x)
        out += list2[1].gradient(self.x)
        out += list2[2].gradient(self.x)

        # gradient without out
        out2 = F.gradient(self.x)
        np.testing.assert_allclose(out.as_array(), out2.as_array())
        
        # gradient with out
        out3 = self.x * 0.
        F.gradient(self.x, out=out3)
        np.testing.assert_allclose(out.as_array(), out3.as_array())

        # check call method
        val = F(self.x)
        val2 = 0.
        for f in F.functions:
            val2 += f(self.x)
        np.testing.assert_almost_equal(val, val2)

        # adding one more function (3 in total)
        scalar = 2.5
        F2 = F + ConstantFunction(scalar)

        # test __add__ method
        assert len(F2.functions) == len(F.functions) + 1

        # test if the sum remains a SumFunction
        self.assertIsInstance(F2, SumFunction)

        # check call 
        np.testing.assert_almost_equal(F2(self.x), F(self.x) + scalar)

        # adding one more function (4 in total)
        F3 = F + F2
    
        np.testing.assert_almost_equal(F2(self.x)+F(self.x), F3(self.x))
        self.assertEqual(len(F3.functions) , len(F2.functions) + len(F.functions))

       # test if the sum remains a SumFunction
        self.assertIsInstance(F3, SumFunction)        


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
        numpy.random.seed(1)
        M, N, K = 3,4,5
        ig = ImageGeometry(M, N, K)
        
        tmp = ig.allocate('random')
        b = ig.allocate('random')
        
        scalar = 0.25
        f1 =  scalar * L2NormSquared(b=b)
        f2 = 5
           
        f = f1 + f2
        
        g = f2 + f1

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
            
        