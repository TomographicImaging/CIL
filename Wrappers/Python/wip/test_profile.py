#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:57:46 2019

@author: evangelos
"""

# profile direct, adjoint, gradient

from ccpi.framework import ImageGeometry
from ccpi.optimisation.operators import Gradient, BlockOperator, Identity
from ccpi.optimisation.functions import MixedL21Norm, L2NormSquared, BlockFunction
import numpy

N, M, K = 2, 3, 2

ig = ImageGeometry(N, M)
b = ig.allocate('random_int')

G = Gradient(ig)
Id = Identity(ig)

#operator = BlockOperator(G, Id)
operator = G

f1 = MixedL21Norm()
f2 = L2NormSquared(b = b)

f = BlockFunction( f1, f2)


x_old = operator.domain_geometry().allocate()
y_old = operator.range_geometry().allocate('random_int') 
  

xbar = operator.domain_geometry().allocate('random_int')

x_tmp = x_old.copy()
x = x_old.copy()
    
y_tmp = operator.range_geometry().allocate()
y = y_old.copy()

y1 = y.copy()

sigma = 20

for i in range(100):
    
    operator.direct(xbar, out = y_tmp)  
    y_tmp *= sigma
    y_tmp += y_old

    
    y_tmp1 =  sigma * operator.direct(xbar) + y_old
    
    print(i)    
    print(" y_old :", y_old[0].as_array(), "\n")
    print(" y_tmp[0] :", y_tmp[0].as_array(),"\n")
    print(" y_tmp1[0] :", y_tmp1[0].as_array())
    
    
    numpy.testing.assert_array_equal(y_tmp[0].as_array(), \
                                            y_tmp1[0].as_array())
    
    numpy.testing.assert_array_equal(y_tmp[1].as_array(), \
                                            y_tmp1[1].as_array()) 
    
    
    y1 = f.proximal_conjugate(y_tmp1, sigma)
    f.proximal_conjugate(y_tmp, sigma, y)
    

    
    
 
    

    
    
    
    
    