#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:44:54 2018

@author: evangelos
"""

# Confirm solutions usign CVX template
#  1) ROF denoising, ( TV - L2 )
# CVXPY has many solvers, but for more accuracy we prefer MOSEK. Need academic licence.
# Otherwise, we can use SCS which is fast but less accurate.


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from cvx_functions import *
from cvxpy import *

#%% Create ground truth data (toy phantom)

N = 100

x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
plt.colorbar()
plt.show()

# Add noise ( Noisy data )
np.random.seed(1)
noisy = x + 0.15*np.random.randn(N, N)

fig = plt.figure()
im = plt.imshow(noisy)
plt.title('Noisy image')
fig.colorbar(im)
plt.show()


#%%
###############################################################################
    # ROF denoising ( objective = TV + L2, with identity operator )
###############################################################################
    
# set regularising parameter
alpha_tvDen = 0.3

# Define the problem
u_tvDen = Variable((N, N))
obj_tvDen =  Minimize(0.5 * sum_squares(u_tvDen - noisy) +  alpha_tvDen * tv_fun(u_tvDen) )
prob_tvDen = Problem(obj_tvDen)

# Choose solver, SCS is fast but less accurate than MOSEK
#result1_denoise = prob1_denoise.solve(verbose=True,solver=SCS,eps=1e-12)
res_tvDen = prob_tvDen.solve(verbose = True, solver = MOSEK)

print()
print('Objective value is {} '.format(obj_tvDen.value))

# Show result
plt.imshow(u_tvDen.value)
plt.title('ROF denoising')
plt.colorbar()
plt.show()

#%%





