#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:57:46 2019

@author: evangelos
"""

# profile direct, adjoint, gradient

from ccpi.framework import ImageGeometry
from ccpi.optimisation.operators import Gradient, BlockOperator, Identity

N, M, K = 200, 300, 100

ig = ImageGeometry(N, M, K)

G = Gradient(ig)
Id = Identity(ig)

u = G.domain_geometry().allocate('random_int')
w = G.range_geometry().allocate('random_int')


res = G.range_geometry().allocate()
res1 = G.domain_geometry().allocate()
#
#
#LHS = (G.direct(u)*w).sum()
#RHS = (u * G.adjoint(w)).sum()
#
#print(G.norm())
#print(LHS, RHS)
#
##%%%re
##
#G.direct(u, out=res)
#G.adjoint(w, out=res1)
##
#LHS1 = (res * w).sum()
#RHS1 = (u * res1).sum()
##
#print(LHS1, RHS1)

B = BlockOperator(2*G, 3*Id)
uB = B.domain_geometry().allocate('random_int')
resB = B.range_geometry().allocate()

#z2 = B.direct(uB)
#B.direct(uB, out = resB)

#%%

for i in range(100):
#    
#    z2 = B.direct(uB)
#    
    B.direct(uB, out = resB)
    
#    z1 = G.adjoint(w)
#    z = G.direct(u)
    
#    G.adjoint(w, out=res1)
    
#    G.direct(u, out=res)
    
    