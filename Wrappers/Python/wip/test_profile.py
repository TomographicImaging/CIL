#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:57:46 2019

@author: evangelos
"""

# profile direct, adjoint, gradient

from ccpi.framework import ImageGeometry
from ccpi.optimisation.operators import Gradient

N, M = 500, 500

ig = ImageGeometry(N, M)

G = Gradient(ig)

u = G.domain_geometry().allocate('random_int')
w = G.range_geometry().allocate('random_int')

for i in range(500):
    
    res = G.adjoint(w)
    