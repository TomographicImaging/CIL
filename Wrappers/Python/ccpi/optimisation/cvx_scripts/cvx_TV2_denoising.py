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
xv[25:74, 25:74] = yv[25:74, 25:74].T
x = xv
# Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
x = x/x.max()

# Select noise
noise = 's&p' # poisson, s&p

#Construct problem
u = Variable((N, N))

# disc Step size
discStep = np.ones(len(u.shape))

if noise == 'gaussian':
    noisy_image = random_noise(x,'gaussian', mean = 0, var = 0.01)
    alpha =  0.075
    constraints = []
    fidelity = 0.5 * sum_squares(u - noisy_image)
    solver = MOSEK
elif noise == 'poisson': 
    scale = 0.03
    noisy_image = scale * np.random.poisson(x/scale)
    alpha = 0.2
    fidelity = sum( u - multiply(noisy_image, log(u)) )    
    constraints = [u>=1e-12]
    solver = SCS
elif noise == 's&p':
    noisy_image = random_noise(x, 's&p', amount = 0.2)
    alpha = 0.4
    constraints = []
    fidelity = pnorm(u - noisy_image,1)
    solver = MOSEK
    
# second order total variation regulariser 
regulariser = alpha * tv2(u)  

obj =  Minimize( regulariser +  fidelity )
prob = Problem(obj, constraints)

res = prob.solve(verbose = True, solver = solver)

print()
print('Objective value is {} '.format(obj.value))


# Show result
plt.gray()
f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[1].imshow(u.value)
ax[1].set_title('TV2 - denoising ')

ax[0].imshow(noisy_image)
ax[0].set_title('Noisy ( ' + noise + ' )')


