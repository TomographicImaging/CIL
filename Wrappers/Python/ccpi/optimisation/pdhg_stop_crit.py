#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:16:43 2019

@author: evangelos
"""

import time

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.funcs import Norm2sq, Norm1, IndicatorBox, ZeroFun
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

#%% Create Phantom
N = 100

# Create phantom
#data = np.zeros((N,N))
#data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
#data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
#ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

data = scipy.misc.ascent()
data = data/np.max(data)
data = resize(data,[100,100])
#
N, M = data.shape
np.random.seed(10)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = M)

# reg_parameter
alpha = 0.5

# Variable for cvx
u_cvx = Variable((N, N))

noise = 's&p' # poisson, s&p (salt & pepper)

if noise == 'gaussian':
    noisy_data = ImageData(random_noise(data,'gaussian', mean = 0, var = 0.01), geometry = ig)
    fidelity = 0.5 * sum_squares(u_cvx - noisy_data.as_array())
    constraints = []
    solver = MOSEK
    operator = form_Operator(gradient(ig), TomoIdentity(ig))
    f = [TV(alpha), Norm2sq_new(TomoIdentity(ig), noisy_data, c = 0.5, memopt = False)]
    g = ZeroFun()
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
    noisy_data = ImageData(random_noise(data, 's&p', amount = 0.1), geometry = ig)
    constraints = []
    fidelity = pnorm(u_cvx-noisy_data.as_array(),1)
    solver = MOSEK
    operator = form_Operator(gradient(ig), TomoIdentity(ig))
    f = [TV(alpha),  L1Norm(TomoIdentity(ig), noisy_data, c =1)]
    g = ZeroFun()
    

#%% Solve with cvx


## total variation regulariser    
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

#sigma = 0.9/normK
#tau = 0.9/normK

ag = ig
opt = {'niter': 2000, 'show_iter': 50} 

res, total_time = PDHG(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)


# Show results

plt.imshow(res.as_array())
plt.show()
 
 

#%%

#f, ax = plt.subplots(2, 2, figsize=(10,10))
#f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                    wspace=0.02, hspace=0.02)
#cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
#cbar = f.colorbar(im, cax=cb_ax)
#
#ax[0,0].imshow(noisy_data.as_array(), cmap = 'viridis')
#ax[0,0].set_title('Noisy ( ' + noise + ' )')
#ax[0,0].set_axis_off()
#
#ax[0,1].imshow(u_cvx.value,cmap = 'viridis')
#ax[0,1].set_title('TV - denoising (CVX)')
#ax[0,1].set_axis_off()
#
#ax[1,1].imshow(res.as_array(),cmap = 'viridis')
#ax[1,1].set_title('TV - denoising (PDHG)')
#ax[1,1].set_axis_off()
#
#diff_img = np.abs(res.as_array() - u_cvx.value)
#ax[1,0].imshow(diff_img,cmap = 'viridis')
#ax[1,0].set_title('Absolute Diff')
#ax[1,0].set_axis_off()

