#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:58:59 2018

@author: evangelos
"""

from ccpi.optimisation.ops import Identity, FiniteDiff2D
from ccpi.optimisation.funcs import Norm2sq, ZeroFun, Norm1, TV2D, Norm2
import numpy as np
from ccpi.optimisation.ops import Identity, FiniteDiff2D
import numpy
from ccpi.framework import DataContainer
from numpy import linalg as LA

z = np.random.rand(10,1)
x = DataContainer(z).as_array()

print(x)

#%%
g = Norm2()

print(g(x))

xx = numpy.sum(numpy.sqrt(numpy.sum(numpy.square(x.as_array()), None, keepdims=True)))

print(xx)


#%%


#ys = sp.lil_matrix(np.array([]))
#
#for i in range(0,3):
#    xs = np.random.randint(10,size=(3,3))
#    ys = sp.hstack([ys, xs]) if ys.size else xs
#
#print(ys)


d1 = np.array([])
d2 = sp.lil_matrix((3,4))

z = sp.hstack( [d1, d2] ) if d1.size else d2

#%%