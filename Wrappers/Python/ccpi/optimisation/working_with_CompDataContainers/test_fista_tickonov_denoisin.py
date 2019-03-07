#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData 
import numpy as np                           

from numpy import inf
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize


from Algorithms.PDHG import PDHG
from Operators.CompositeOperator_DataContainer import CompositeOperator, CompositeDataContainer
from Operators.IdentityOperator import *
from Operators.GradientOperator import Gradient

from Functions.FunctionComposition import FunctionComposition_new
from Functions.mixed_L12Norm import mixed_L12Norm
from Functions.L2NormSquared import L2NormSq
from Functions.ZeroFun import ZeroFun

from skimage.util import random_noise

#%%###############################################################################
# Create phantom for TV

N = 100
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = (N,N)
ag = ig


Id = Identity(ig, ag)

# Create noisy data. Add Gaussian noise
n1 = random_noise(phantom, mode='gaussian', seed=10)
noisy_data = ImageData(n1)
alpha = 2

op1 = Gradient(ig)
f = L2NormSq(0.5, b = noisy_data)
g0 = FunctionComposition_new(op1, L2NormSq(0.1))

opt = { 'iter': 1000}
x_init = ImageData(np.zeros((N,N)))



#x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)
#
#plt.imshow(x_fista1.as_array())
#plt.show()

#%%
