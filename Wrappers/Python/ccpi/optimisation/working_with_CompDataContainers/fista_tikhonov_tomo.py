#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:03:12 2019

@author: evangelos
"""


from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.algs import FISTA
import numpy as np                           

from numpy import inf
import matplotlib.pyplot as plt
from cvxpy import *

from Algorithms import PDHG
from Operators import CompositeOperator, Identity, Gradient, \
                     SymmetrizedGradient, CompositeDataContainer, ZeroOp, AstraProjectorSimple
from Functions import ZeroFun, L2NormSq, mixed_L12Norm, L1Norm, \
                      FunctionOperatorComposition, BlockFunction

from skimage.util import random_noise


#%%###############################################################################
# Create phantom for TV tomography

N = 100
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = 100
angles = np.linspace(0,np.pi,100)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(data)

plt.imshow(sin.as_array())
plt.title('Sinogram')
plt.colorbar()
plt.show()

# Add Gaussian noise to the sinogram data
np.random.seed(10)
n1 = np.random.random(sin.shape)
noisy_data = ImageData(sin.as_array() + 0.25*n1)

plt.imshow(noisy_data.as_array())
plt.title('Noisy Sinogram')
plt.colorbar()
plt.show()

alpha = 80
f = FunctionOperatorComposition(Aop, L2NormSq(0.5, b=sin))
g0 = L2NormSq(alpha)
#g0 = L1Norm(alpha)

opt = { 'iter': 400}

x_init = ImageData(np.zeros((N,N)))

x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)
plt.imshow(x_fista1.as_array())
plt.colorbar()
plt.show()