#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:41:05 2019

@author: evangelos
"""
import numpy as np
from skimage.util import random_noise

import matplotlib.pyplot as plt
from cvx_functions import *
from cvxpy import *

# Create a phantom

N = 100
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
x = data

#N = 100
#
#x = np.zeros((N,N))
#
#x1 = np.linspace(0, 30, N)
#x2 = np.linspace(30, 0., N)
#xv, yv = np.meshgrid(x1, x2)
#
#xv[25:74, 25:51] = 25
#xv[25:74, 51:74] = 5
#
#x = xv
## Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
#x = x/x.max()

# Select noise
noise = 'gaussian' # poisson, s&p (salt & pepper)

#Construct problem    
u = Variable((N, N))


if noise == 'gaussian':
#    noisy_image = random_noise(x.shape,'gaussian', mean = 0, var = 0.01)
    np.random.seed(10)
    z = np.random.rand(N,N)
    noisy_image = x + 0.25 * z
    alpha = 10
    constraints = []
    fidelity = 0.5 * sum_squares(u - noisy_image)
    solver = MOSEK
elif noise == 'poisson': 
    scale = 0.03
    noisy_image = scale * np.random.poisson(x/scale)
    alpha = 0.5 
    fidelity = sum( u - multiply(noisy_image, log(u)) )    
    constraints = [u>=1e-12]
    solver = SCS
elif noise == 's&p':
    noisy_image = random_noise(x, 's&p', amount = 0.2)
    alpha = 2
    constraints = []
    fidelity = pnorm(u-noisy_image,1)
    solver = MOSEK
 
# total variation regulariser    
regulariser = alpha * TV_cvx(u)    

obj =  Minimize( regulariser +  fidelity)
prob = Problem(obj, constraints)

# Choose solver (SCS is fast but less accurate than MOSEK)
res = prob.solve(verbose = True, solver = solver)

print()
print('Objective value is {} '.format(obj.value))

# Show result
plt.gray()
f, ax = plt.subplots(1, 2, figsize=(13,13))

ax[0].imshow(noisy_image)
ax[0].set_title('Noisy ( ' + noise + ' )')

ax[1].imshow(u.value)
ax[1].set_title('TV - denoising ')



