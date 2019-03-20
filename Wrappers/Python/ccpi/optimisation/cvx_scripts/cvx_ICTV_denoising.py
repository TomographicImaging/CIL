#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:44:54 2018

@author: evangelos
"""
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt
from cvx_functions import *
from cvxpy import *

# Create a phantom 
N = 100

x = np.zeros((N,N))

x1 = np.linspace(0, 30, N)
x2 = np.linspace(30, 0., N)
xv, yv = np.meshgrid(x1, x2)

xv[25:74, 25:51] = 25
xv[25:74, 51:74] = 5
x = xv
# Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
x = x/x.max()

# Select noise
noise = 'gaussian' # poisson, gaussian, s&p

#Construct problem
u = Variable((N, N))
v = Variable((N, N))

# disc Step size
discStep = np.ones(len(u.shape))


if noise == 'gaussian':
    noisy_image = random_noise(x,'gaussian', mean = 0, var = 0.01)
    alpha0, alpha1 = 0.2, 1
    constraints = []
    fidelity = 0.5 * sum_squares(u + v - noisy_image)
    solver = MOSEK
elif noise == 'poisson': 
    scale = 0.03
    noisy_image = scale * np.random.poisson(x/scale)
    alpha0, alpha1 = 1, 5
    fidelity = sum( u + v - multiply(noisy_image, log(u + v)) )    
    constraints = [u>=1e-12, v>=1e-12]
    solver = SCS
elif noise == 's&p':
    noisy_image = random_noise(x, 's&p', amount = 0.2)
    alpha0, alpha1 = 1, 3
    constraints = []
    fidelity = pnorm(u + v - noisy_image,1)
    solver = MOSEK

# Create infimal convolution total variation
regulariser = ictv(u, v, alpha0, alpha1)  

obj =  Minimize( regulariser +  fidelity)
prob = Problem(obj, constraints)

res = prob.solve(verbose = True, solver = solver)

print()
print('Objective value is {} '.format(obj.value))

# Show result
plt.gray()
f, ax = plt.subplots(2, 2, figsize=(10,10))

ax[0,1].imshow(u.value + v.value)
ax[0,1].set_title('ICTV - denoising ')

ax[0,0].imshow(noisy_image)
ax[0,0].set_title('Noisy ( ' + noise + ' )')

ax[1,0].imshow(u.value)
ax[1,0].set_title('Decomposition: u')

ax[1,1].imshow(v.value)
ax[1,1].set_title('Decomposition: v')

