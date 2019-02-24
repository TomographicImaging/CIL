#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:36 2019

@author: evangelos
"""

import time

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.funcs import Norm2sq, Norm1, IndicatorBox
from ccpi.optimisation.ops import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare,TomoIdentity

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from my_changes import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

import sys
sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/wip/cvx_scripts/')
from cvx_functions import *

from algorithms import *
from operators import *
from regularisers import *

#%% Create Phantom
N = 100

# Create phantom
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

#data = scipy.misc.ascent()
#data = data/np.max(data)

#################  color      #################################################
#data = scipy.misc.face()
#data = data/np.max(data)
#N, M, K = data.shape
#ig = ImageGeometry(voxel_num_z=K, voxel_num_x = N, voxel_num_y = M)
#noisy_data = ImageData(data + 0.25 * np.random.random_sample((N,M,K)), geometry=ig)
#operator = form_Operator(gradient(ig), TomoIdentity(ig))
#alpha = 0.5
#f = [TV(alpha), Norm2sq_new(TomoIdentity(ig), noisy_data, c = 0.5, memopt = False)]
#plt.imshow(noisy_data.as_array())
#plt.show()
#g = ZeroFun()


#%%

N, M = data.shape
np.random.seed(10)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = M)

# Variable for cvx
u_cvx = Variable((N, N))

noise = 'gaussian' # poisson, s&p (salt & pepper)

if noise == 'gaussian':
    alpha = 1
    noisy_data = ImageData(data + 0.25 * np.random.random_sample((N,N)), geometry=ig)#ImageData(random_noise(data,'gaussian', mean = 0, var = 0.01), geometry = ig)
    fidelity = 0.5 * sum_squares(u_cvx - noisy_data.as_array())
    constraints = []
    solver = MOSEK
    operator = form_Operator(gradient(ig), TomoIdentity(ig))
    f = [TV(alpha), Norm2sq_new(TomoIdentity(ig), noisy_data, c = 0.5, memopt = False)]
    g = ZeroFun()
#    operator = form_Operator(gradient(ig))
#    f = [TV(alpha)]
#    g = Norm2sq_new(TomoIdentity(ig), noisy_data, c = 0.5, memopt = False)
elif noise == 'poisson': 
    scale = 0.05
    noisy_data = ImageData(scale * np.random.poisson(data/scale), geometry = ig)
    fidelity = sum( u_cvx - multiply(noisy_data.as_array(), log(u_cvx)) )    
    constraints = [u_cvx>=1e-12]
    solver = SCS # this is not accurate compared to MOSEK
    operator = form_Operator(gradient(ig), TomoIdentity(ig))
    f = [TV(alpha),  KL_diverg(TomoIdentity(ig), noisy_data, c = 1)]
    g = ZeroFun()
elif noise == 's&p':
    alpha = 1
    noisy_data = ImageData(random_noise(data, 's&p', amount = 0.1), geometry = ig)
    constraints = []
    fidelity = pnorm(u_cvx-noisy_data.as_array(),1)
    solver = MOSEK
    operator = form_Operator(gradient(ig), TomoIdentity(ig))
    f = [TV(alpha),  L1Norm(TomoIdentity(ig), noisy_data, c =1)]
    g = ZeroFun()
    

#%% Solve with cvx


# total variation regulariser    
#regulariser = alpha * TV_cvx(u_cvx)    
#
#obj =  Minimize( regulariser +  fidelity)
#prob = Problem(obj, constraints)
#
## Choose solver (SCS is fast but less accurate than MOSEK)
#res = prob.solve(verbose = True, solver = solver)
#
#print()
#print('Objective value is {} '.format(obj.value))

#%% Solve with pdhg
    
normK = compute_opNorm(operator)
# Primal & dual stepsizes
sigma = 20
tau = 1/(sigma*normK**2)
#sigma = 1.0/normK
#tau = 1.0/normK


#%%

ag = ig
opt = {'niter':1000, 'show_iter':100, 'stop_crit': cmp_L2norm,\
       'tol':1e-5}
res, total_time, its = PDHG(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)

plt.imshow(res.as_array())
plt.show()

#%%
## Show results
#
#plt.subplots(2, 2, figsize=(10,10))
#diff_img = np.abs(res.as_array() - u_cvx.value)
#
#plt.subplot(2, 2, 1)
#plt.imshow(noisy_data.as_array(), cmap='viridis', aspect='auto')
#plt.colorbar()
#plt.title('Noisy ( ' + noise + ' )')
#
#plt.subplot(2, 2, 2)
#plt.imshow(u_cvx.value, cmap='viridis', aspect='auto')
#plt.colorbar()
#plt.title('TV - denoising (CVX)')
#
#plt.subplot(2, 2, 3)
#plt.imshow(res.as_array(), cmap='viridis', aspect='auto')
#plt.colorbar()
#plt.title('TV - denoising (PDHG)')
#
#plt.subplot(2, 2, 4)
#plt.imshow(diff_img, cmap='viridis', aspect='auto')
#plt.colorbar()
#plt.title('Absolute Diff')


