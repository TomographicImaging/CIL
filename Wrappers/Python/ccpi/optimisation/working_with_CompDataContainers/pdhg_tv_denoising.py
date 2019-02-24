#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:31:47 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.ops import PowerMethodNonsquare


import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG
from operators import CompositeOperator, Identity
from GradientOperators import Gradient
from functions import L1Norm, ZeroFun, L2NormSq

#%%###############################################################################
# Create phantom for TV

N = 200
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = (N,N)
ag = ig

# Create noisy data. Add Gaussian noise
noisy_data = ImageData(random_noise(data,'gaussian', mean = 0, var = 0.01))
alpha = 1

# Create operators
op1 = Gradient(ig)
op2 = Identity(ig, ag)

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 

# Create functions
f = [ L1Norm(op1,alpha), \
      L2NormSq(op2, noisy_data, c = 0.5, memopt = False) ]
g = ZeroFun()

# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

# Number of iterations
opt = {'niter':1000}

# Run algorithm
res, total_time, its = PDHG(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)

plt.imshow(res.as_array())
plt.colorbar()
plt.show()
##
plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.show()
#
plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'Recon')
plt.legend()