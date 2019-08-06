#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:39:34 2019

@author: evangelos
"""

from ccpi.framework import BlockGeometry, ImageGeometry
from ccpi.optimisation.functions import MixedL21Norm
import numpy

N, M = 10, 20

ig = ImageGeometry(M, N)

BG = BlockGeometry(ig, ig)

U = BG.allocate('random')

f_no_scaled = MixedL21Norm() 
f_scaled = 0.5 * MixedL21Norm()  

# call
a1 = f_no_scaled(U)
a2 = f_scaled(U)    
print(a1)
print(2*a2)
print(numpy.sum(numpy.sqrt(U[0].as_array()**2 + U[1].as_array()**2)))





#%%