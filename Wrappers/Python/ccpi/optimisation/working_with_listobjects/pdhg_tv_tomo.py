#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:19:46 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.astra.ops import AstraProjectorSimple


import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG_testGeneric
from operators import CompositeOperator, Identity, AstraProjectorSimple
from GradientOperators import Gradient
from functions import L1Norm, ZeroFun, L2NormSq

#%% # Create phantom

N = 100
ig = ImageGeometry(voxel_num_x = N, voxel_num_y=N)

data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
data = ImageData(data)

detectors = N
angles = np.linspace(0,np.pi,80)
SourceOrig = 100
OrigDetec = 0

# parallel
ag = AcquisitionGeometry('parallel','2D',angles,detectors)

#cone geometry
#ag = AcquisitionGeometry('cone','2D',angles,detectors,sourcecenter=SourceOrig, centerTodetector=OrigDetec)

# Create ccpi-astra projectir
Aop = AstraProjectorSimple(ig, ag, 'cpu')

#%%

# create sinogram and noisy sinogram
sin = Aop.direct(data)

np.random.seed(1)

noise = 'gaussian'

if noise == 'gaussian':
    noisy_data = AcquisitionData(sin.as_array() + np.random.normal(0, 2, sin.shape))
elif noise == 'poisson':
    scale = 0.5
    noisy_data = AcquisitionData(scale * np.random.poisson(sin.as_array()/scale))
    
# simple backprojection
backproj = Aop.adjoint(noisy_data)

plt.imshow(data.as_array())
plt.title('Phantom image')
plt.show()

plt.imshow(noisy_data.array)
plt.title('Simulated data')
plt.show()

plt.imshow(backproj.array)
plt.title('Backprojected data')
plt.show()

#%%
alpha = 100 

# Create operators
op1 = Gradient((ig.voxel_num_x,ig.voxel_num_y))
op2 = Aop

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 

f = [ L1Norm(op1,alpha), \
      L2NormSq(op2, noisy_data, c = 0.5, memopt = False) ]
g = ZeroFun()

#Aop.norm = Aop.get_max_sing_val()
normK = operator.norm()
#normK = compute_opNorm(operator)

# Primal & dual stepsizes
sigma = 10
tau = 1/(sigma*normK**2)

opt = {'niter':1000}
res, total_time, its = PDHG_testGeneric(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)

# Show result
plt.imshow(res.as_array(), cmap = 'viridis')
plt.title('Reconstruction with PDHG')
plt.colorbar()
plt.show()


#%%