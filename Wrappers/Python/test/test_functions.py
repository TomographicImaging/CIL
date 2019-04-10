#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:24:37 2019

@author: evangelos
"""


import numpy as np
#from ccpi.optimisation.funcs import Function
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 
from ccpi.optimisation.operators import  Identity
from ccpi.optimisation.operators import BlockOperator
from ccpi.framework import BlockDataContainer
from numbers import Number
from ccpi.optimisation.operators import Gradient

#from ccpi.optimisation.functions import SimpleL2NormSq
from ccpi.optimisation.functions import L2NormSquared
#from ccpi.optimisation.functions import SimpleL1Norm
from ccpi.optimisation.functions import L1Norm

from ccpi.optimisation.funcs import Norm2sq
# from ccpi.optimisation.functions.L2NormSquared import SimpleL2NormSq, L2NormSq
# from ccpi.optimisation.functions.L1Norm import SimpleL1Norm, L1Norm
#from ccpi.optimisation.functions import mixed_L12Norm
from ccpi.optimisation.functions import ZeroFun

from ccpi.optimisation.functions import FunctionOperatorComposition
import unittest  
import numpy      

#    

                    
class TestFunction(unittest.TestCase):
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
    
    def test_L2NormSquared(self):
        # TESTS for L2 and scalar * L2
    
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
        
        #check convex conjuagate with data
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
        
        
    def test_Norm2sq_as_FunctionOperatorComposition(self):
        M, N, K = 2,3,5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
        u = ig.allocate(ImageGeometry.RANDOM_INT)
        b = ig.allocate(ImageGeometry.RANDOM_INT) 
        
        A = 0.5 * Identity(ig)
        old_chisq = Norm2sq(A, b, 1.0)
        new_chisq = FunctionOperatorComposition(A, L2NormSquared(b=b))

        yold = old_chisq(u)
        ynew = new_chisq(u)
        self.assertEqual(yold, ynew)

        yold = old_chisq.gradient(u)
        ynew = new_chisq.gradient(u)
        numpy.testing.assert_array_equal(yold.as_array(), ynew.as_array())
        

    
#    
#    f1 = L2NormSq(alpha=1, b=noisy_data)
#    print(f1(noisy_data))
#    
#    f2 =  L2NormSq(alpha=5, b=noisy_data).composition_with(op2)
#    print(f2(noisy_data))
#    
#    print(f1.gradient(noisy_data).as_array())
#    print(f2.gradient(noisy_data).as_array())
##    
#    print(f1.proximal(noisy_data,1).as_array())
#    print(f2.proximal(noisy_data,1).as_array())
#    
#    
#    f3 = mixed_L12Norm(alpha = 1).composition_with(op1)
#    print(f3(noisy_data))
#            
#    print(ImageData(op1.direct(noisy_data).power(2).sum(axis=0)).sqrt().sum())
#    
#    print( 5*(op2.direct(d) - noisy_data).power(2).sum(), f2(d))
#    
#    from functions import mixed_L12Norm as mixed_L12Norm_old
#    
#    print(mixed_L12Norm_old(op1,None,alpha)(noisy_data))
    
    
    #        

        
