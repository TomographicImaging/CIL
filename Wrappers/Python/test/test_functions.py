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

from cil.optimisation.functions.Function import ScaledFunction
import numpy as np

from cil.framework import DataContainer, ImageGeometry, \
    VectorGeometry, VectorData, BlockDataContainer
from cil.optimisation.operators import IdentityOperator, MatrixOperator, CompositionOperator, DiagonalOperator, BlockOperator
from cil.optimisation.functions import Function, KullbackLeibler, ConstantFunction, TranslateFunction
from cil.optimisation.operators import GradientOperator

from cil.optimisation.functions import Function, KullbackLeibler, WeightedL2NormSquared, L2NormSquared,\
                                         L1Norm, MixedL21Norm, LeastSquares, \
                                         ZeroFunction, OperatorCompositionFunction,\
                                         Rosenbrock, IndicatorBox, TotalVariation       
from cil.optimisation.functions import BlockFunction                              

import unittest
import numpy
import scipy.special

from cil.framework import ImageGeometry
from cil.optimisation.functions import TranslateFunction
from timeit import default_timer as timer

import numpy as np                         
from cil.utilities import dataexample
from cil.utilities import noise
import os
import sys
try:
    from ccpi.filters import regularisers
    has_reg_toolkit = True
except ImportError as ie:
    has_reg_toolkit = False
if has_reg_toolkit:
    from cil.plugins.ccpi_regularisation.functions import FGP_TV

print ("Has Reg Toolkit", has_reg_toolkit)

try:
    import tomophantom
    from tomophantom import TomoP3D
    has_tomophantom = True
    from cil.plugins import TomoPhantom
except ImportError as ie:
    has_tomophantom = False

from cil.utilities.quality_measures import mae
has_numba = True
try:
    import numba
except ImportError as ie:
    has_numba = False

has_numba = True
try:
    import numba
except ImportError as ie:
    has_numba = False


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
    def assertBlockDataContainerAlmostEqual(self, container1, container2, decimal=7):
        print ("assert Block Data Container Equal")
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if issubclass(container1.get_item(col).__class__, DataContainer):
                print ("Checking col ", col)
                self.assertNumpyArrayAlmostEqual(
                    container1.get_item(col).as_array(), 
                    container2.get_item(col).as_array(), 
                    decimal = decimal
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),
                container2.get_item(col), decimal=decimal)
    
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
    
        numpy.random.seed(10)
        N = 3
        ig = ImageGeometry(N,N)
        ag = ig       
        op1 = GradientOperator(ig)
        op2 = IdentityOperator(ig, ag)

        # Form Composite Operator
        operator = BlockOperator(op1, op2 , shape=(2,1) )
        
        # Create functions
        noisy_data = ag.allocate(ImageGeometry.RANDOM)
        
        d = ag.allocate(ImageGeometry.RANDOM)
        alpha = 0.5

        # scaled function
        g = alpha * L2NormSquared(b=noisy_data)
        
        # Compare call of g
        a2 = alpha*(d - noisy_data).power(2).sum()

        self.assertEqual(a2, g(d))
        
        # Compare convex conjugate of g
        a3 = 0.5 * d.squared_norm() + d.dot(noisy_data)
        self.assertAlmostEqual(a3, g.convex_conjugate(d), places=7)
    
    def test_L2NormSquared(self):
        # TESTS for L2 and scalar * L2
        print ("Test L2NormSquared")
        numpy.random.seed(1)
        M, N, K = 2,3,5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
        u = ig.allocate(ImageGeometry.RANDOM)
        b = ig.allocate(ImageGeometry.RANDOM) 
        
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
        numpy.testing.assert_almost_equal(d1, d2, decimal=6)  
        
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
        u = ig.allocate(ImageGeometry.RANDOM, seed=1)
        b = ig.allocate(ImageGeometry.RANDOM, seed=2) 
        
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
           
    def test_Norm2sq_as_OperatorCompositionFunction(self):
        
        print('Test for OperatorCompositionFunction')         
            
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        #numpy.random.seed(1)
        b = ig.allocate('random', seed=1)
        
        print('Check call with IdentityOperator operator... OK\n')
        operator = 3 * IdentityOperator(ig)
            
        u = ig.allocate('random', seed = 50)
        f = 0.5 * L2NormSquared(b = b)
        func1 = OperatorCompositionFunction(f, operator)
        func2 = LeastSquares(operator, b, 0.5)
        print("f.L {}".format(f.L))
        print("0.5*f.L {}".format((0.5*f).L))
        print("type func1 {}".format(type(func1)))
        print("func1.L {}".format(func1.L))
        print("func2.L {}".format(func2.L))
        print("operator.norm() {}".format(operator.norm()))
  
        numpy.testing.assert_almost_equal(func1(u), func2(u))
        
        
        print('Check gradient with IdentityOperator operator... OK\n')
        
        tmp1 = ig.allocate()
        tmp2 = ig.allocate()
        res_gradient1 = func1.gradient(u)
        res_gradient2 = func2.gradient(u)    
        func1.gradient(u, out = tmp1)
        func2.gradient(u, out = tmp2)
            
        self.assertNumpyArrayAlmostEqual(res_gradient1.as_array(), res_gradient2.as_array())
        self.assertNumpyArrayAlmostEqual(tmp1.as_array(), tmp2.as_array())
       
        
        print('Check call with MatrixOperator... OK\n')  
        mat = np.random.randn(M, N)
        operator = MatrixOperator(mat)   
        vg = VectorGeometry(N)
        b = vg.allocate('random')    
        u = vg.allocate('random')
          
        func1 = OperatorCompositionFunction(0.5 * L2NormSquared(b = b), operator)
        func2 = LeastSquares(operator, b, 0.5)
         
        self.assertNumpyArrayAlmostEqual(func1(u), func2(u))       
        numpy.testing.assert_almost_equal(func1.L, func2.L)         
            
    def test_mixedL12Norm(self):
        numpy.random.seed(1)
        M, N, K = 2,3,5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        u1 = ig.allocate('random')
        u2 = ig.allocate('random')
        
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
        u3 = ig.allocate('random')
        u4 = ig.allocate('random')
        
        z3 = BlockDataContainer(u3, u4, shape=(2,1))
        
        
        f_no_scaled.proximal_conjugate(U, 1, out=z3)
        self.assertBlockDataContainerAlmostEqual(z3,z1, decimal=5)

    def test_KullbackLeibler(self):
        print ("test_KullbackLeibler")
        #numpy.random.seed(1)
        M, N, K =  2, 3, 4
        ig = ImageGeometry(N, M, K)
        
        u1 = ig.allocate('random', seed = 500)    
        g1 = ig.allocate('random', seed = 100)
        b1 = ig.allocate('random', seed = 1000)
        
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
        u2 = u1 * 0 + 2.
        self.assertNumpyArrayAlmostEqual(0.0, f.convex_conjugate(u2))   


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
        
    def tests_for_L2NormSq_and_weighted(self):
        numpy.random.seed(1)
        M, N, K = 2,3,1
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
        u = ig.allocate('random')
        b = ig.allocate('random') 
        
        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = 2 * u
        numpy.testing.assert_array_almost_equal(a1.as_array(), a2.as_array(), decimal=4)
        numpy.testing.assert_equal(f(u), u.squared_norm())
    
        # check grad/call with data
        
#        igggg = ImageGeometry(4,4)
        f1 = L2NormSquared(b=b)
        b1 = f1.gradient(u)
        b2 = 2 * (u-b)
            
        numpy.testing.assert_array_almost_equal(b1.as_array(), b2.as_array(), decimal=4)
        numpy.testing.assert_equal(f1(u), ((u-b)).squared_norm())
        
        #check convex conjuagate no data
        c1 = f.convex_conjugate(u)
        c2 = 1/4 * u.squared_norm()
        numpy.testing.assert_equal(c1, c2)
        
        #check convex conjuagate with data
        d1 = f1.convex_conjugate(u)
        d2 = (1/4) * u.squared_norm() + u.dot(b)
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
        
        
        
        print( " ####### check without out ######### " )
              
              
        u_out_no_out = ig.allocate('random_int')         
        res_no_out = f_scaled_data.proximal_conjugate(u_out_no_out, 0.5)          
        print(res_no_out.as_array())
        
        print( " ####### check with out ######### " ) 
              
        res_out = ig.allocate()        
        f_scaled_data.proximal_conjugate(u_out_no_out, 0.5, out = res_out)
        
        print(res_out.as_array())   
    
        numpy.testing.assert_array_almost_equal(res_no_out.as_array(), \
                                                res_out.as_array(), decimal=4)  
        
        
        
        ig1 = ImageGeometry(2,3)
        
        tau = 0.1
        
        u = ig1.allocate('random')
        b = ig1.allocate('random')
        
        scalar = 0.5
        f_scaled = scalar * L2NormSquared(b=b)
        f_noscaled = L2NormSquared(b=b)
        
        
        res1 = f_scaled.proximal(u, tau)
        res2 = f_noscaled.proximal(u, tau*scalar)
        
    #    res2 = (u + tau*b)/(1+tau)
        
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)
        
                    
        print("Checks for WeightedL2NormSquared")

        
        # Tests for weighted L2NormSquared
        ig = ImageGeometry(voxel_num_x = 3, voxel_num_y = 3)
        weight = ig.allocate('random')
        
        f = WeightedL2NormSquared(weight=weight)                                              
        x = ig.allocate(0.4)
        
        res1 = f(x)
        res2 = (weight * (x**2)).sum()
        numpy.testing.assert_almost_equal(res1, res2, decimal=4)
        
        print("Call of WeightedL2NormSquared is ... ok")
        
        # gradient for weighted L2NormSquared    
        res1 = f.gradient(x)
        res2 = 2*weight*x
        out = ig.allocate()
        f.gradient(x, out = out)
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                out.as_array(), decimal=4)  
        numpy.testing.assert_array_almost_equal(res2.as_array(), \
                                                out.as_array(), decimal=4)  
        
        print("GradientOperator of WeightedL2NormSquared is ... ok")    
        
        # convex conjugate for weighted L2NormSquared       
        res1 = f.convex_conjugate(x)
        res2 = 1/4 * (x/weight.sqrt()).squared_norm()
        numpy.testing.assert_array_almost_equal(res1, \
                                                res2, decimal=4)   
        
        print("Convex conjugate of WeightedL2NormSquared is ... ok")        
        
        # proximal for weighted L2NormSquared       
        tau = 0.3
        out = ig.allocate()
        res1 = f.proximal(x, tau)
        f.proximal(x, tau, out = out)
        res2 = x/(1+2*tau*weight)
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)  
        
        print("Proximal of WeightedL2NormSquared is ... ok")  
        
        
        tau = 0.3
        out = ig.allocate()
        res1 = f.proximal_conjugate(x, tau)   
        res2 = x/(1 + tau/(2*weight))    
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)  
        
        print("Proximal conjugate of WeightedL2NormSquared is ... ok")  
    
    
        b = ig.allocate('random')
        f1 = TranslateFunction(WeightedL2NormSquared(weight=weight), b) 
        f2 = WeightedL2NormSquared(weight = weight, b=b)
        res1 = f1(x)
        res2 = f2(x)
        numpy.testing.assert_almost_equal(res1, res2, decimal=4)
        
        print("Call of WeightedL2NormSquared vs TranslateFunction is ... ok") 
        
        f1 = WeightedL2NormSquared(b=b)
        f2 = L2NormSquared(b=b)
        
        numpy.testing.assert_almost_equal(f1.L, f2.L, decimal=4)
        numpy.testing.assert_almost_equal(f1.L, 2, decimal=4)
        numpy.testing.assert_almost_equal(f2.L, 2, decimal=4)
        
        print("Check Lip constants ... ok")          
        


    def tests_for_LS_weightedLS(self):
                        
        ig = ImageGeometry(40,30)
        
        numpy.random.seed(1)

        A = IdentityOperator(ig)
        b = ig.allocate('random')
        x = ig.allocate('random')
        c = numpy.float64(0.3)
        
        weight = ig.allocate('random') 
        
        D = DiagonalOperator(weight)
        norm_weight = numpy.float64(D.norm())
        print("norm_weight", norm_weight)
        
        f1 = LeastSquares(A, b, c, weight) 
        f2 = LeastSquares(A, b, c)
        
        print("Check LS vs wLS")        
        
        # check Lipshitz    
        numpy.testing.assert_almost_equal(f2.L, 2 * c * (A.norm()**2))   
        print ("unwrapped", 2. * c * norm_weight * (A.norm()**2))
        print ("f1.L", f1.L)
        numpy.testing.assert_almost_equal(f1.L, numpy.float64(2.) * c * norm_weight * (A.norm()**2)) 
        print("Lipschitz is ... OK")
            
        # check call with weight                   
        res1 = c * (A.direct(x)-b).dot(weight * (A.direct(x) - b))
        res2 = f1(x)    
        numpy.testing.assert_almost_equal(res1, res2)
        print("Call is ... OK")        
        
        # check call without weight                  
        #res1 = c * (A.direct(x)-b).dot((A.direct(x) - b))
        res1 = c * (A.direct(x)-b).squared_norm()
        res2 = f2(x)    
        numpy.testing.assert_almost_equal(res1, res2) 
        print("Call without weight is ... OK")        
        
        # check gradient with weight             
        out = ig.allocate(None)
        res1 = f1.gradient(x)
        #out = f1.gradient(x)
        f1.gradient(x, out=out)
        res2 = 2 * c * A.adjoint(weight*(A.direct(x)-b))
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array())
        numpy.testing.assert_array_almost_equal(out.as_array(), res2.as_array())
        print("GradientOperator is ... OK")          
        
        # check gradient without weight             
        out = ig.allocate()
        res1 = f2.gradient(x)
        f2.gradient(x, out = out)
        res2 = 2*c*A.adjoint(A.direct(x)-b)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array())
        numpy.testing.assert_array_almost_equal(out.as_array(), res2.as_array())
        
        print("GradientOperator without weight is ... OK")   


        print("Compare Least Squares with DiagonalOperator + CompositionOperator")
        
        ig2 = ImageGeometry(100,100,100)
        A = IdentityOperator(ig2)
        b = ig2.allocate('random')
        x = ig2.allocate('random')
        c = 0.3
        
        weight = ig2.allocate('random')     
        
        weight_operator = DiagonalOperator(weight.sqrt())
        tmp_A = CompositionOperator(weight_operator, A)
        tmp_b = weight_operator.direct(b)
    
        f1 = LeastSquares(tmp_A, tmp_b, c)    
        f2 = LeastSquares(A, b, c, weight)
        
        t0 = timer()
        res1 = f1(x)
        t1 = timer()
        print("Time with Composition+Diagonal is {}".format(t1-t0))
        
        t2 = timer()
        res2 = f2(x)
        t3 = timer()
        print("Time with LS + weight is {}".format(t3-t2))
        
        numpy.testing.assert_almost_equal(res1, res2, decimal=2)          

    def test_Lipschitz(self):
        print('Test for OperatorCompositionFunction')         
            
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        b = ig.allocate('random', seed=1)
        
        print('Check call with IdentityOperator operator... OK\n')
        operator = 3 * IdentityOperator(ig)
            
        u = ig.allocate('random_int', seed = 50)
        func2 = LeastSquares(operator, b, 0.5)
        assert func2.L != 2
        print (func2.L)
        func2.L = 2
        assert func2.L == 2
    def test_Lipschitz2(self):
        print('Test for test_Lipschitz2')         
            
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        b = ig.allocate('random', seed=1)
        
        print('Check call with IdentityOperator operator... OK\n')
        operator = 3 * IdentityOperator(ig)
            
        u = ig.allocate('random_int', seed = 50)
        func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        f3 = func1 + func2
        assert f3.L != 2
        print (func2.L)
        func2.L = 2
        assert func2.L == 2
    def test_Lipschitz3(self):
        print('Test for test_Lipschitz3')         
            
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        b = ig.allocate('random', seed=1)
        
        print('Check call with IdentityOperator operator... OK\n')
        operator = 3 * IdentityOperator(ig)
            
        u = ig.allocate('random_int', seed = 50)
        # func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        f3 = TranslateFunction(func1, 3)
        assert f3.L != 2
        print (f3.L)
        f3.L = 2
        assert f3.L == 2
    def test_Lipschitz4(self):
        print('Test for test_Lipschitz4')         
            
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        b = ig.allocate('random', seed=1)
        
        print('Check call with IdentityOperator operator... OK\n')
        operator = 3 * IdentityOperator(ig)
            
        u = ig.allocate('random_int', seed = 50)
        # func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        f3 = func1 + 3
        assert f3.L == 0
        print ("OK")
        print (f3.L)
        f3.L = 2
        assert f3.L == 2
        print ("OK")
        assert func1.L == 0
        print ("OK")
        try:
            func1.L = 2
            assert False
        except AttributeError as ve:
            assert True
        print ("OK")
        f2 = LeastSquares(operator, b, 0.5)
        f4 = 2 * f2
        assert f4.L == 2 * f2.L
        
        print ("OK")
        f4.L = 10
        assert f4.L != 2 * f2.L  
        print ("OK")

        f4 = -2 * f2
        assert f4.L == 2 * f2.L


class TestTotalVariation(unittest.TestCase):

    def setUp(self) -> None:
        self.tv = TotalVariation()
        self.alpha = 0.15
        self.tv_scaled = self.alpha * TotalVariation()
        self.tv_iso = TotalVariation()
        self.tv_aniso = TotalVariation(isotropic=False)
        self.ig_real = ImageGeometry(3,4)   
        self.grad = GradientOperator(self.ig_real)  
        
    def test_regularisation_parameter(self):
        np.testing.assert_almost_equal(self.tv.regularisation_parameter, 1.)

    def test_regularisation_parameter2(self):
        np.testing.assert_almost_equal(self.tv_scaled.regularisation_parameter, self.alpha)
    
    def test_rmul(self):
        assert isinstance(self.tv_scaled, TotalVariation)
    
    def test_regularisation_parameter3(self):
        try:
            self.tv.regularisation_parameter = 'string'
            assert False
        except TypeError as te:
            assert True
    def test_rmul2(self):
        alpha = 'string'
        try:
            tv = alpha * TotalVariation()
            assert False
        except TypeError as te:
            assert True

    def test_call_real_isotropic(self):

        x_real = self.ig_real.allocate('random', seed=4)  
        

        res1 = self.tv_iso(x_real)
        res2 = self.grad.direct(x_real).pnorm(2).sum()
        np.testing.assert_equal(res1, res2)  

    def test_call_real_anisotropic(self):

        x_real = self.ig_real.allocate('random', seed=4) 
        
        res1 = self.tv_aniso(x_real)
        res2 = self.grad.direct(x_real).pnorm(1).sum()
        np.testing.assert_equal(res1, res2)                
    
    @unittest.skipUnless(has_reg_toolkit, "Regularisation Toolkit not present")
    def test_compare_regularisation_toolkit(self):
    
        # print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV no tolerance (2D)")

        data = dataexample.SHAPES.get()
        ig = data.geometry
        ag = ig

        np.random.seed(0)
        # Create noisy data. 
        n1 = np.random.normal(0, 0.1, size = ig.shape)
        noisy_data = ig.allocate()
        noisy_data.fill(n1+data.as_array())
        
        alpha = 0.1
        iters = 100
            
        # CIL_FGP_TV no tolerance
        g_CIL = alpha * TotalVariation(iters, tolerance=None, lower = 0, info = True)
        t0 = timer()
        res1 = g_CIL.proximal(noisy_data, 1.)
        t1 = timer()
        # print(t1-t0)
        
        r_alpha = alpha
        r_iterations = iters
        r_tolerance = 1e-9
        r_iso = True
        r_nonneg = True
        g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations, tolerance=r_tolerance, 
             isotropic=r_iso, nonnegativity=r_nonneg, device='cpu')
        
        t2 = timer()
        res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
        t3 = timer()
        # print(t3-t1)
        
        
        # np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal = 4)
        np.testing.assert_allclose(res1.as_array(), res2.as_array(), atol=2.5e-3)

        ###################################################################
        ###################################################################
        ###################################################################
        ###################################################################    
        
        # print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with iterations.")
        iters = 408
        # CIL_FGP_TV no tolerance
        g_CIL = alpha * TotalVariation(iters, tolerance=1e-9, lower = 0.)
        t0 = timer()
        res1 = g_CIL.proximal(noisy_data, 1.)
        t1 = timer()
        # print(t1-t0)
        
        r_alpha = alpha
        r_iterations = iters
        r_tolerance = 1e-9
        r_iso = True
        r_nonneg = True
        g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations, tolerance=r_tolerance, 
             isotropic=r_iso, nonnegativity=r_nonneg, device='cpu')

        t2 = timer()
        res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
        t3 = timer()
        # print(t3-t2)
        
        
        # print(mae(res1, res2))
        np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=3)    
        
        ###################################################################
        ###################################################################
        ###################################################################
        ###################################################################
    @unittest.skipUnless(has_tomophantom and has_reg_toolkit, "Missing Tomophantom or Regularisation-Toolkit")
    def test_compare_regularisation_toolkit_tomophantom(self):
    
        # print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV no tolerance (3D)") 
            
        # print ("Building 3D phantom using TomoPhantom software")
        model = 13 # select a model number from the library
        N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
        #This will generate a N_size x N_size x N_size phantom (3D)
        
        ig = ImageGeometry(N_size, N_size, N_size)
        data = TomoPhantom.get_ImageData(num_model=model, geometry=ig)

        noisy_data = noise.gaussian(data, seed=10)
        
        alpha = 0.1
        iters = 100
        
        # print("Use tau as an array of ones")
        # CIL_TotalVariation no tolerance
        g_CIL = alpha * TotalVariation(iters, tolerance=None, info=True)
        # res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
        t0 = timer()   
        res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
        t1 = timer()
        # print(t1-t0)

        # CCPi Regularisation toolkit high tolerance
        
        r_alpha = alpha
        r_iterations = iters
        r_tolerance = 1e-9
        r_iso = True
        r_nonneg = True
        g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations, tolerance=r_tolerance, 
             isotropic=r_iso, nonnegativity=r_nonneg, device='cpu')


        t2 = timer()
        res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
        t3 = timer()
        # print (t3-t2)
        
        np.testing.assert_allclose(res1.as_array(), res2.as_array(), atol=7.5e-2)


class TestKullbackLeiblerNumba(unittest.TestCase):
    def setUp(self):
        print ("test_KullbackLeibler numba")
        #numpy.random.seed(1)
        M, N, K =  2, 3, 4
        ig = ImageGeometry(N, M)
        
        u1 = ig.allocate('random', seed = 500)
        u1 = ig.allocate(0.2)  
        #g1 = ig.allocate('random', seed = 100)
        g1 = ig.allocate(1)

        b1 = ig.allocate('random', seed = 1000)
        eta = ig.allocate(1e-3)

        mask = ig.allocate(1)

        mask.fill(0, horizontal_x=0)

        mask_c = ig.allocate(0)
        mask_c.fill(1, horizontal_x=0)

        # print ("mask\n", mask.as_array())
        # print ("mask_c\n", mask_c.as_array())

        f = KullbackLeibler(b=g1, use_numba=True, eta=eta)
        f_np = KullbackLeibler(b=g1, use_numba=False, eta=eta)


        # mask is on vartical=0
        # separate the u1 vertical=0
        f_mask = KullbackLeibler(b=g1.copy(), use_numba=True, mask=mask.copy(), eta=eta.copy())
        f_mask_c = KullbackLeibler(b=g1.copy(), use_numba=True, mask=mask_c.copy(), eta=eta.copy())
        f_on_mask = KullbackLeibler(b=g1.subset(horizontal_x=0), use_numba=True, eta=eta.subset(horizontal_x=0))
        u1_on_mask = u1.subset(horizontal_x=0)


        tau = 400.4
        self.tau = tau
        self.u1 = u1
        self.g1 = g1
        self.b1 = b1
        self.eta = eta
        self.f = f
        self.f_np = f_np
        self.mask = mask
        self.mask_c = mask_c
        self.f_mask = f_mask
        self.f_mask_c = f_mask_c
        self.f_on_mask = f_on_mask
        self.u1_on_mask = u1_on_mask
        
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_call(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f(u1), f_np(u1),  rtol=1e-5)
    
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_call_mask(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1
        g1 = self.g1
        mask = self.mask

        u1_on_mask = self.u1_on_mask
        f_on_mask = self.f_on_mask
        f_mask = self.f_mask
        f_mask_c = self.f_mask_c
        
        numpy.testing.assert_allclose(f_mask(u1) + f_mask_c(u1), f(u1),  rtol=1e-5)

    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_proximal(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.proximal(u1,tau=tau).as_array(), 
                                      f_np.proximal(u1,tau=tau).as_array(), rtol=7e-3)
        numpy.testing.assert_array_almost_equal(f.proximal(u1,tau=tau).as_array(), 
        f_np.proximal(u1,tau=tau).as_array(), decimal=4)
        
    
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_proximal_arr(self):
        f = self.f
        f_np = self.f_np
        tau = self.u1.copy()
        tau.fill(self.tau)
        u1 = self.u1
        a = f.proximal(u1,tau=self.tau)
        b = f.proximal(u1,tau=tau)
        numpy.testing.assert_allclose(f.proximal(u1,tau=self.tau).as_array(), 
                                      f.proximal(u1,tau=tau).as_array(), rtol=7e-3)
        numpy.testing.assert_array_almost_equal(f.proximal(u1,tau=self.tau).as_array(), 
                                                f.proximal(u1,tau=tau).as_array(), decimal=4)
        
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_gradient(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.gradient(u1).as_array(), f_np.gradient(u1).as_array(), rtol=1e-3)
        
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_convex_conjugate(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.convex_conjugate(u1), f_np.convex_conjugate(u1), rtol=1e-3)
        
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_proximal_conjugate_arr(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.proximal_conjugate(u1,tau=tau).as_array(), 
                        f_np.proximal_conjugate(u1,tau=tau).as_array(), rtol=1e-3)
    
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_convex_conjugate_mask(self):
        f = self.f
        tau = self.tau
        u1 = self.u1

        mask = self.mask
        f_mask = self.f_mask
        f_mask_c = self.f_mask_c
        f_on_mask = self.f_on_mask
        u1_on_mask = self.u1_on_mask

        print (f.convex_conjugate(u1))
        print (f_mask.convex_conjugate(u1))
        print (f_mask_c.convex_conjugate(u1))

        numpy.testing.assert_allclose(
            f.convex_conjugate(u1), 
            f_mask.convex_conjugate(u1) + f_mask_c.convex_conjugate(u1) ,\
                 rtol=1e-3)

    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_KullbackLeibler_numba_proximal_conjugate_mask(self):
        f = self.f
        f_mask = self.f_mask
        f_mask_c = self.f_mask_c
        x = self.u1
        m = self.mask
        m_c = self.mask_c
        tau = self.tau

        out = x * 0
        out_c = x * 0
        f_mask_c.proximal_conjugate(x,tau=tau, out=out_c)
        f_mask.proximal_conjugate(x,tau=tau, out=out)
        numpy.testing.assert_allclose(f.proximal_conjugate(x,tau=tau).as_array(), 
                                      (out + out_c).as_array(), rtol=7e-3)
        # print ("f.prox_conj\n"       , f.proximal_conjugate(x,tau=tau).as_array())
        # print ("f_mask.prox_conj\n"  , out.as_array())
        # print ("f_mask_c.prox_conj\n", out_c.as_array())
        b = f_mask_c.proximal_conjugate(x,tau=tau)
        a = f_mask.proximal_conjugate(x,tau=tau)
        numpy.testing.assert_allclose(f.proximal_conjugate(x,tau=tau).as_array(), 
                                      (f_mask.proximal_conjugate(x,tau=tau) +\
                                      f_mask_c.proximal_conjugate(x, tau=tau)) .as_array(), rtol=7e-3)
        
    def tearDown(self):
        pass


class TestLeastSquares(unittest.TestCase):
    def setUp(self) -> None:
        ig = ImageGeometry(10,2)
        A = IdentityOperator(ig)
        self.A = A
        self.ig = ig
        return super().setUp()


    def test_rmul(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        assert constant * ls.c == twicels.c

    def test_rmul_with_call(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        np.testing.assert_almost_equal( constant * ls(x) , twicels(x))
    def test_rmul_with_Lipschitz(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        np.testing.assert_almost_equal( constant * ls.L , twicels.L)

    def test_rmul_with_gradient(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        y1 = ls.gradient(x)
        y2 = twicels.gradient(x)
        np.testing.assert_array_almost_equal( constant * y1.as_array(), y2.as_array())

        ls.gradient(x, out=y2)
        twicels.gradient(x, out=y1)
        np.testing.assert_array_almost_equal( constant * y2.as_array(), y1.as_array())
        
        
# tests for OperatorCompositionFunction
class TestOperatorCompositionFunctionWithWrongInterfaceFunction(unittest.TestCase):
    def setUp(self):
        ig = ImageGeometry(2,2)
        I = IdentityOperator(ig)
        
        x = ig.allocate(0)
        class NotAFunction(object):
            pass
        ocf = OperatorCompositionFunction(NotAFunction(), I)
        self.pars = (ig, I, x, ocf)
    def tearDown(self):
        pass

    def test_call(self):
        ig , I, x, ocf = self.pars
        
        try:
            ocf(x)
            self.assertFalse(True)
        except TypeError as te:
            self.assertFalse(False)
    def test_L(self):
        ig , I, x, ocf = self.pars
        try:
            ocf.L
            self.assertTrue(False)
        except AttributeError as ae:
            self.assertTrue(True)
    def test_gradient(self):
        ig , I, x, ocf = self.pars
        try:
            ocf.gradient(x)
            self.assertTrue(False)
        except AttributeError as ae:
            self.assertTrue(True)
    def test_proximal(self):
        ig , I, x, ocf = self.pars
        try:
            ocf.proximal(x, tau=1)
            self.assertTrue(False)
        except NotImplementedError as ae:
            self.assertTrue(True)
    def test_proximal_conjugate(self):
        ig , I, x, ocf = self.pars
        try:
            ocf.proximal_conjugate(x, tau=1)
            self.assertTrue(False)
        except NotImplementedError as ae:
            self.assertTrue(True)
    def test_convex_conjugate(self):
        ig , I, x, ocf = self.pars
        try:
            ocf.convex_conjugate(x)
            self.assertTrue(False)
        except NotImplementedError as ae:
            self.assertTrue(True)
    def test_proximal_conjugate(self):
        ig , I, x, ocf = self.pars
        try:
            ocf.proximal_conjugate(x, tau=1)
            self.assertTrue(False)
        except NotImplementedError as ae:
            self.assertTrue(True)
    
class TestOperatorCompositionFunctionWithWrongInterfaceFunctionAddScalar(TestOperatorCompositionFunctionWithWrongInterfaceFunction):
    def setUp(self):
        ig = ImageGeometry(2,2)
        I = IdentityOperator(ig)
        
        x = ig.allocate(0)
        class NotAFunction(object):
            pass
        ocf = OperatorCompositionFunction(NotAFunction(), I) + 1
        self.pars = (ig, I, x, ocf)
class TestOperatorCompositionFunctionWithWrongInterfaceFunctionMultiplyScalar(TestOperatorCompositionFunctionWithWrongInterfaceFunction):
    def setUp(self):
        ig = ImageGeometry(2,2)
        I = IdentityOperator(ig)
        
        x = ig.allocate(0)
        class NotAFunction(object):
            pass
        ocf = OperatorCompositionFunction(NotAFunction(), I) * 2.
        self.pars = (ig, I, x, ocf)
class TestOperatorCompositionFunctionWithWrongInterfaceFunctionAddFunction(TestOperatorCompositionFunctionWithWrongInterfaceFunction):
    def setUp(self):
        ig = ImageGeometry(2,2)
        I = IdentityOperator(ig)
        
        x = ig.allocate(0)
        class NotAFunction(object):
            pass
        ocf = OperatorCompositionFunction(NotAFunction(), I) + IndicatorBox()
        self.pars = (ig, I, x, ocf)

class TestOperatorCompositionFunctionWithWrongInterfaceOperator(TestOperatorCompositionFunctionWithWrongInterfaceFunction):
    def setUp(self):
        ig = ImageGeometry(2,2)
        F = IndicatorBox()
        
        x = ig.allocate(0)
        class NotAnOperator(object):
            pass
        nao = NotAnOperator()
        ocf = OperatorCompositionFunction(F, nao)
        self.pars = (ig, nao, x, ocf)
    def tearDown(self):
        pass
    def test_call(self):
        ig , I, x, ocf = self.pars
        
        try:
            ocf(x)
            self.assertFalse(True)
        except AttributeError as te:
            self.assertFalse(False)
class TestOperatorCompositionFunctionWithWrongInterfaceOperatorScaled(TestOperatorCompositionFunctionWithWrongInterfaceOperator):
    def setUp(self):
        ig = ImageGeometry(2,2)
        F = IndicatorBox()
        
        x = ig.allocate(0)
        class NotAnOperator(object):
            def __mul__(self, value):
                return self
        nao = NotAnOperator() * 2
        ocf = OperatorCompositionFunction(F, nao)
        self.pars = (ig, nao, x, ocf)

class TestBlockFunction(unittest.TestCase):
    def setUp(self):
        # M, N = 50, 50
        # ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        # b = ig.allocate('random', seed=1)
        
        # print('Check call with IdentityOperator operator... OK\n')
        # operator = 3 * IdentityOperator(ig)
            
        # u = ig.allocate('random_int', seed = 50)
        # func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        func2 = ConstantFunction(-1.0)
        self.funcs = [ func1 , func2 ]
        # self.ig = ig

    def tearDown(self) -> None:
        return super().tearDown()

    def test_iterator(self):
        bf = BlockFunction(*self.funcs)
        for el in bf:
            assert isinstance(el, ConstantFunction)

    def test_rmul_with_scalar_return(self):
        bf = BlockFunction(*self.funcs)
        bf2 = 2*bf
        assert isinstance(bf2, BlockFunction)

    def test_getitem(self):
        bf = BlockFunction(*self.funcs)
        assert isinstance(bf[0], ConstantFunction)
        assert isinstance(bf[1], ConstantFunction)
        
        

    def test_rmul_with_scalar1(self):
        bf0 = BlockFunction(*self.funcs)
        bf = 2*bf0

        print (bf[0].constant, bf0[0].constant)
        print (bf[1].constant, bf0[1].constant)

        for i in range(2):
            assert bf[i].constant == 2*bf0[i].constant
    def test_rmul_with_scalar2(self):
        bf0 = BlockFunction(L1Norm())
        bf = 2*bf0
        assert isinstance(bf[0], ScaledFunction)