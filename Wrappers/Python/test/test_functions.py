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

from ccpi.optimisation.functions import SimpleL2NormSq
from ccpi.optimisation.functions import L2NormSq
from ccpi.optimisation.functions import SimpleL1Norm
from ccpi.optimisation.functions import L1Norm
# from ccpi.optimisation.functions.L2NormSquared import SimpleL2NormSq, L2NormSq
# from ccpi.optimisation.functions.L1Norm import SimpleL1Norm, L1Norm
from ccpi.optimisation.functions import mixed_L12Norm
from ccpi.optimisation.functions import ZeroFun

from ccpi.optimisation.functions import FunctionOperatorComposition
import unittest        

#    

                    
class TestFunction(unittest.TestCase):
    def test_Function(self):
    
    
        N = 3
        ig = (N,N)
        ag = ig       
        op1 = Gradient(ig)
        op2 = Identity(ig, ag)

        # Form Composite Operator
        operator = BlockOperator((2,1), op1, op2 ) 
        
        # Create functions
        noisy_data = ImageData(np.random.randint(10, size=ag))
        
        d = ImageData(np.random.randint(10, size=ag))
        
        f = mixed_L12Norm(alpha = 1).composition_with(op1)
        g = L2NormSq(alpha=0.5, b=noisy_data)
        
        # Compare call of f
        a1 = ImageData(op1.direct(d).power(2).sum(axis=0)).sqrt().sum()
        #print(a1, f(d))
        self.assertEqual (a1, f(d))
        
        # Compare call of g
        a2 = g.alpha*(d - noisy_data).power(2).sum()
        #print(a2, g(d)) 
        self.assertEqual(a2, g(d))
        
        # Compare convex conjugate of g
        a3 = 0.5 * d.power(2).sum() + (d*noisy_data).sum()
        self.assertEqual(a3, g.convex_conjugate(d))
        #print( a3, g.convex_conjugate(d))
    
    
    
    
    
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

        
