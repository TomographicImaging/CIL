#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 01:08:35 2019

@author: evangelos
"""

from numbers import Number
import numpy
from ccpi.optimisation.functions import Function
from ccpi.framework import ImageGeometry, ImageData
from ScaledFunction import ScaledFunction


M, N = 200,300
ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
u = ig.allocate('random')
b = ig.allocate('random') 
    
alpha = 0.5
    
f = L2NormSquared()
g = alpha * L2NormSquared()
    
f1 = L2NormSquared(b=b)
g1 = alpha * L2NormSquared(b=b)
    
a1 = alpha * f(u)
a2 = g(u)
    
numpy.testing.assert_equal(a1, a2)
    