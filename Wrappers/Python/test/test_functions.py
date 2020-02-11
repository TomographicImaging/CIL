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

from __future__ import absolute_import

import numpy as np

from ccpi.framework import DataContainer, ImageData, ImageGeometry, \
    VectorGeometry, VectorData, BlockDataContainer
from ccpi.optimisation.operators import Identity, LinearOperatorMatrix, BlockOperator
from ccpi.optimisation.functions import Function, KullbackLeibler
from numbers import Number
from ccpi.optimisation.operators import Gradient

from ccpi.optimisation.functions import Function, KullbackLeibler, L2NormSquared,\
                                         L1Norm, MixedL21Norm, LeastSquares, \
                                         ZeroFunction, FunctionOperatorComposition,\
                                         Rosenbrock, IndicatorBox

import unittest
import numpy
import scipy.special

                    
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
    def test_Function(self):
    
    
        N = 3
        ig = ImageGeometry(N,N)
        ag = ig       
        op1 = Gradient(ig)
        op2 = Identity(ig, ag)

        # Form Composite Operator
        operator = BlockOperator(op1, op2 , shape=(2,1) )
        
        # Create functions
        noisy_data = ag.allocate(ImageGeometry.RANDOM_INT)
        
        d = ag.allocate(ImageGeometry.RANDOM_INT)
        alpha = 0.5
        # scaled function
        g = alpha * L2NormSquared(b=noisy_data)
        
        # Compare call of g
        a2 = alpha*(d - noisy_data).power(2).sum()
        #print(a2, g(d)) 
        self.assertEqual(a2, g(d))
        
        # Compare convex conjugate of g
        a3 = 0.5 * d.squared_norm() + d.dot(noisy_data)
        self.assertEqual(a3, g.convex_conjugate(d))
        #print( a3, g.convex_conjugate(d))

        #test proximal conjugate

    
    def test_L2NormSquared(self):
        # TESTS for L2 and scalar * L2
        print ("Test L2NormSquared")
    
        M, N, K = 2,3,5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
        u = ig.allocate(ImageGeometry.RANDOM_INT)
        b = ig.allocate(ImageGeometry.RANDOM_INT) 
        
        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = 2 * u
        numpy.testing.assert_array_almost_equal(a1.as_array(), a2.as_array(), decimal=4)
        numpy.testing.assert_equal(f(u), u.squared_norm())

        # check grad/call with data
        f1 = L2NormSquared(b=b)
        b1 = f1.gradient(u)
        b2 = 2 * (u-b)
            
        numpy.testing.assert_array_almost_equal(b1.as_array(), b2.as_array(), decimal=4)
        numpy.testing.assert_equal(f1(u), (u-b).squared_norm())
        
        #check convex conjuagate no data
        c1 = f.convex_conjugate(u)
        c2 = 1/4. * u.squared_norm()
        numpy.testing.assert_equal(c1, c2)
        
        #check convex conjugate with data
        d1 = f1.convex_conjugate(u)
        d2 = (1./4.) * u.squared_norm() + (u*b).sum()
        numpy.testing.assert_equal(d1, d2)  
        
        # check proximal no data
        tau = 5
        e1 = f.proximal(u, tau)
        e2 = u/(1+2*tau)
        numpy.testing.assert_array_almost_equal(e1.as_array(), e2.as_array(), decimal=4)
        
        # check proximal with data
        tau = 5
        h1 = f1.proximal(u, tau)
        h2 = (u-b)/(1+2*tau) + b
        numpy.testing.assert_array_almost_equal(h1.as_array(), h2.as_array(), decimal=4)    
        
        # check proximal conjugate no data
        tau = 0.2
        k1 = f.proximal_conjugate(u, tau)
        k2 = u/(1 + tau/2 )
        numpy.testing.assert_array_almost_equal(k1.as_array(), k2.as_array(), decimal=4) 
        
        # check proximal conjugate with data
        l1 = f1.proximal_conjugate(u, tau)
        l2 = (u - tau * b)/(1 + tau/2 )
        numpy.testing.assert_array_almost_equal(l1.as_array(), l2.as_array(), decimal=4)     

        # check scaled function properties

        # scalar 
        scalar = 100
        f_scaled_no_data = scalar * L2NormSquared()
        f_scaled_data = scalar * L2NormSquared(b=b)
        
        # call
        numpy.testing.assert_equal(f_scaled_no_data(u), scalar*f(u))
        numpy.testing.assert_equal(f_scaled_data(u), scalar*f1(u))
        
        # grad
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.gradient(u).as_array(), scalar*f.gradient(u).as_array(), decimal=4)
        numpy.testing.assert_array_almost_equal(f_scaled_data.gradient(u).as_array(), scalar*f1.gradient(u).as_array(), decimal=4)
        
        # conj
        numpy.testing.assert_almost_equal(f_scaled_no_data.convex_conjugate(u), \
                                f.convex_conjugate(u/scalar) * scalar, decimal=4)
        
        numpy.testing.assert_almost_equal(f_scaled_data.convex_conjugate(u), \
                                scalar * f1.convex_conjugate(u/scalar), decimal=4)
        
        # proximal
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal(u, tau).as_array(), \
                                                f.proximal(u, tau*scalar).as_array())
        
        
        numpy.testing.assert_array_almost_equal(f_scaled_data.proximal(u, tau).as_array(), \
                                                f1.proximal(u, tau*scalar).as_array())
                                
        
        # proximal conjugate
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal_conjugate(u, tau).as_array(), \
                                                (u/(1 + tau/(2*scalar) )).as_array(), decimal=4)
        
        numpy.testing.assert_array_almost_equal(f_scaled_data.proximal_conjugate(u, tau).as_array(), \
                                                ((u - tau * b)/(1 + tau/(2*scalar) )).as_array(), decimal=4)    
        
    def test_L2NormSquaredOut(self):
        # TESTS for L2 and scalar * L2
    
        M, N, K = 2,3,5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
        u = ig.allocate(ImageGeometry.RANDOM_INT)
        b = ig.allocate(ImageGeometry.RANDOM_INT) 
        
        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = a1 * 0.
        f.gradient(u, out=a2)
        numpy.testing.assert_array_almost_equal(a1.as_array(), a2.as_array(), decimal=4)
        #numpy.testing.assert_equal(f(u), u.squared_norm())

        # check grad/call with data
        f1 = L2NormSquared(b=b)
        b1 = f1.gradient(u)
        b2 = b1 * 0.
        f1.gradient(u, out=b2)
            
        numpy.testing.assert_array_almost_equal(b1.as_array(), b2.as_array(), decimal=4)
        #numpy.testing.assert_equal(f1(u), (u-b).squared_norm())
        
        # check proximal no data
        tau = 5
        e1 = f.proximal(u, tau)
        e2 = e1 * 0.
        f.proximal(u, tau, out=e2)
        numpy.testing.assert_array_almost_equal(e1.as_array(), e2.as_array(), decimal=4)
        
        # check proximal with data
        tau = 5
        h1 = f1.proximal(u, tau)
        h2 = h1 * 0.
        f1.proximal(u, tau, out=h2)
        numpy.testing.assert_array_almost_equal(h1.as_array(), h2.as_array(), decimal=4)    
        
        # check proximal conjugate no data
        tau = 0.2
        k1 = f.proximal_conjugate(u, tau)
        k2 = k1 * 0.
        f.proximal_conjugate(u, tau, out=k2)

        numpy.testing.assert_array_almost_equal(k1.as_array(), k2.as_array(), decimal=4) 
        
        # check proximal conjugate with data
        l1 = f1.proximal_conjugate(u, tau)
        l2 = l1 * 0.
        f1.proximal_conjugate(u, tau, out=l2)
        numpy.testing.assert_array_almost_equal(l1.as_array(), l2.as_array(), decimal=4)     

        # check scaled function properties

        # scalar 
        scalar = 100
        f_scaled_no_data = scalar * L2NormSquared()
        f_scaled_data = scalar * L2NormSquared(b=b)
        
        # grad
        w = f_scaled_no_data.gradient(u)
        ww = w * 0
        f_scaled_no_data.gradient(u, out=ww)

        numpy.testing.assert_array_almost_equal(w.as_array(), 
            ww.as_array(), decimal=4)

        # numpy.testing.assert_array_almost_equal(f_scaled_data.gradient(u).as_array(), scalar*f1.gradient(u).as_array(), decimal=4)
        
        # # conj
        # numpy.testing.assert_almost_equal(f_scaled_no_data.convex_conjugate(u), \
        #                         f.convex_conjugate(u/scalar) * scalar, decimal=4)
        
        # numpy.testing.assert_almost_equal(f_scaled_data.convex_conjugate(u), \
        #                         scalar * f1.convex_conjugate(u/scalar), decimal=4)
        
        # # proximal
        w = f_scaled_no_data.proximal(u, tau)
        ww = w * 0
        f_scaled_no_data.proximal(u, tau, out=ww)
        numpy.testing.assert_array_almost_equal(w.as_array(), \
                                                ww.as_array())
        
        
        # numpy.testing.assert_array_almost_equal(f_scaled_data.proximal(u, tau).as_array(), \
        #                                         f1.proximal(u, tau*scalar).as_array())
                                
        
        # proximal conjugate
        w = f_scaled_no_data.proximal_conjugate(u, tau)
        ww = w * 0
        f_scaled_no_data.proximal_conjugate(u, tau, out=ww)
        numpy.testing.assert_array_almost_equal(w.as_array(), \
                                                ww.as_array(), decimal=4)
        
        # numpy.testing.assert_array_almost_equal(f_scaled_data.proximal_conjugate(u, tau).as_array(), \
        #                                         ((u - tau * b)/(1 + tau/(2*scalar) )).as_array(), decimal=4)    
           
    def test_Norm2sq_as_FunctionOperatorComposition(self):
        
        print('Test for FunctionOperatorComposition')         
            
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        b = ig.allocate('random_int')
        
        print('Check call with Identity operator... OK\n')
        operator = 3 * Identity(ig)
            
        u = ig.allocate('random_int', seed = 50)
        
        func1 = FunctionOperatorComposition(0.5 * L2NormSquared(b = b), operator)
        func2 = LeastSquares(operator, b, 0.5)
            
        self.assertNumpyArrayAlmostEqual(func1(u), func2(u))
        
        
        print('Check gradient with Identity operator... OK\n')
        
        tmp1 = ig.allocate()
        tmp2 = ig.allocate()
        res_gradient1 = func1.gradient(u)
        res_gradient2 = func2.gradient(u)    
        func1.gradient(u, out = tmp1)
        func2.gradient(u, out = tmp2)
            
        self.assertNumpyArrayAlmostEqual(tmp1.as_array(), tmp2.as_array())
        self.assertNumpyArrayAlmostEqual(res_gradient1.as_array(), res_gradient2.as_array())
        
        print('Check call with LinearOperatorMatrix... OK\n')  
        mat = np.random.randn(M, N)
        operator = LinearOperatorMatrix(mat)   
        vg = VectorGeometry(N)
        b = vg.allocate('random_int')    
        u = vg.allocate('random_int')
          
        func1 = FunctionOperatorComposition(0.5 * L2NormSquared(b = b), operator)
        func2 = LeastSquares(operator, b, 0.5)
         
        self.assertNumpyArrayAlmostEqual(func1(u), func2(u))   
        
        self.assertNumpyArrayAlmostEqual(func1.L, func2.L)
            
    def test_mixedL12Norm(self):
        M, N, K = 2,3,5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        u1 = ig.allocate('random_int')
        u2 = ig.allocate('random_int')
        
        U = BlockDataContainer(u1, u2, shape=(2,1))
        
        # Define no scale and scaled
        f_no_scaled = MixedL21Norm() 
        f_scaled = 1 * MixedL21Norm()  
        
        # call
        
        a1 = f_no_scaled(U)
        a2 = f_scaled(U)
        self.assertNumpyArrayAlmostEqual(a1,a2)
        
        
        tmp = [ el**2 for el in U.containers ]
        self.assertBlockDataContainerEqual(BlockDataContainer(*tmp),
                                           U.power(2))
        
        z1 = f_no_scaled.proximal_conjugate(U, 1)
        u3 = ig.allocate('random_int')
        u4 = ig.allocate('random_int')
        
        z3 = BlockDataContainer(u3, u4, shape=(2,1))
        
        
        f_no_scaled.proximal_conjugate(U, 1, out=z3)
        self.assertBlockDataContainerEqual(z3,z1)

    def test_KullbackLeibler(self):
        print ("test_KullbackLeibler")
        
        M, N, K =  2, 3, 4
        ig = ImageGeometry(N, M, K)
        
        u1 = ig.allocate('random_int', seed = 500)    
        g1 = ig.allocate('random_int', seed = 100)
        b1 = ig.allocate('random_int', seed = 1000)
        
        # with no data
        try:
            f = KullbackLeibler()   
        except ValueError:
            print('Give data b=...\n')
            
        print('With negative data, no background\n')   
        try:        
            f = KullbackLeibler(b=-1*g1)
        except ValueError:
            print('We have negative data\n') 
            
        f = KullbackLeibler(b=g1)        
            
        print('Check KullbackLeibler(x,x)=0\n') 
        self.assertNumpyArrayAlmostEqual(0.0, f(g1))
                
        print('Check gradient .... is OK \n')
        res_gradient = f.gradient(u1)
        res_gradient_out = u1.geometry.allocate()
        f.gradient(u1, out = res_gradient_out) 
        self.assertNumpyArrayAlmostEqual(res_gradient.as_array(), \
                                                res_gradient_out.as_array(),decimal = 4)  
        
        print('Check proximal ... is OK\n')        
        tau = 400.4
        res_proximal = f.proximal(u1, tau)
        res_proximal_out = u1.geometry.allocate()   
        f.proximal(u1, tau, out = res_proximal_out)
        self.assertNumpyArrayAlmostEqual(res_proximal.as_array(), \
                                                res_proximal_out.as_array(), decimal =5)  
        
        print('Check conjugate ... is OK\n')  
        
        if (1 - u1.as_array()).all():
            print('If 1-x<=0, Convex conjugate returns 0.0')
            
        self.assertNumpyArrayAlmostEqual(0.0, f.convex_conjugate(u1))   


        print('Check KullbackLeibler with background\n')      
        eta = b1
        
        f1 = KullbackLeibler(b=g1, eta=b1) 
            
        tmp_sum = (u1 + eta).as_array()
        ind = tmp_sum >= 0
        tmp = scipy.special.kl_div(f1.b.as_array()[ind], tmp_sum[ind])                 
        self.assertNumpyArrayAlmostEqual(f1(u1), numpy.sum(tmp) )          
        
        res_proximal_conj_out = u1.geometry.allocate()
        proxc = f.proximal_conjugate(u1,tau)
        f.proximal_conjugate(u1, tau, out=res_proximal_conj_out)
        print(res_proximal_conj_out.as_array())
        print(proxc.as_array())
        numpy.testing.assert_array_almost_equal(proxc.as_array(), res_proximal_conj_out.as_array())

    def test_Rosenbrock(self):
        f = Rosenbrock (alpha = 1, beta=100)
        x = VectorData(numpy.asarray([1,1]))
        assert f(x) == 0.
        numpy.testing.assert_array_almost_equal( f.gradient(x).as_array(), numpy.zeros(shape=(2,), dtype=numpy.float32))
    def test_IndicatorBox(self):
        ig = ImageGeometry(10,10)
        im = ig.allocate(-1)
        ib = IndicatorBox(lower=0)
        a = ib(im)
        numpy.testing.assert_equal(a, numpy.inf)
        ib = IndicatorBox(lower=-2)
        a = ib(im)
        numpy.testing.assert_array_equal(0, a)
        ib = IndicatorBox(lower=-5, upper=-2)
        a = ib(im)
        numpy.testing.assert_equal(a, numpy.inf)

if __name__ == '__main__':
    
    d = TestFunction()
    d.test_KullbackLeibler()
