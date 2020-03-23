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
                                         L1Norm, MixedL21Norm, weighted_L2NormSquared,  LeastSquares, \
                                         ZeroFunction, FunctionOperatorComposition,\
                                         Rosenbrock, IndicatorBox

import unittest
import numpy
import scipy.special

                    
class TestFunctionL2NormSq(unittest.TestCase):
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


    def tests_for_L2NormSq_and_weighted(self):

        M, N, K = 2,3,1
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
        u = ig.allocate('random_int')
        b = ig.allocate('random_int') 
        
        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = 2 * u
        numpy.testing.assert_array_almost_equal(a1.as_array(), a2.as_array(), decimal=4)
        numpy.testing.assert_equal(f(u), u.squared_norm())
    
        # check grad/call with data
        
        igggg = ImageGeometry(4,4)
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
        
        u = ig1.allocate('random_int')
        b = ig1.allocate('random_int')
        
        scalar = 0.5
        f_scaled = scalar * L2NormSquared(b=b)
        f_noscaled = L2NormSquared(b=b)
        
        
        res1 = f_scaled.proximal(u, tau)
        res2 = f_noscaled.proximal(u, tau*scalar)
        
    #    res2 = (u + tau*b)/(1+tau)
        
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)
        
        
        # Tests for weighted L2NormSquared
        ig = ImageGeometry(voxel_num_x = 3, voxel_num_y = 3)
        weight = ig.allocate('random_int')
        
        f = weighted_L2NormSquared(weight=weight)                                              
        x = ig.allocate(0.4)
        
        res1 = f(x)
        res2 = (weight * (x**2)).sum()
        numpy.testing.assert_almost_equal(res1, res2, decimal=4)
        
        # gradient for weighted L2NormSquared    
        res1 = f.gradient(x)
        res2 = 2*weight*x
        out = ig.allocate()
        f.gradient(x, out = out)
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                out.as_array(), decimal=4)  
        numpy.testing.assert_array_almost_equal(res2.as_array(), \
                                                out.as_array(), decimal=4)  
        
        # convex conjugate for weighted L2NormSquared       
        res1 = f.convex_conjugate(x)
        res2 = 1/4 * (x/weight.sqrt()).squared_norm()
        numpy.testing.assert_array_almost_equal(res1, \
                                                res2, decimal=4)   
        
        # proximal for weighted L2NormSquared       
        tau = 0.3
        out = ig.allocate()
        res1 = f.proximal(x, tau)
        f.proximal(x, tau, out = out)
        res2 = x/(1+2*tau*weight)
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)  
        
        
        
        

