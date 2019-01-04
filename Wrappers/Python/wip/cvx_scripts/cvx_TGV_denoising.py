#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:44:54 2018

@author: evangelos
"""

# Confirm solutions usign CVX template
#  1) Total Generalised Variation denoising (TGV - L2 )
# CVXPY has many solvers, but for more accuracy we prefer MOSEK. Need academic licence.
# Otherwise, we can use SCS which is fast but less accurate.

#%%
###############################################################################
    # TGV - L2 denoising ( Total Generalised Variation )
###############################################################################

# Create a phantom with piecewise smooth structures
N = 100

x = np.zeros((N,N))

x1 = np.linspace(0, 30, N)
x2 = np.linspace(30, 0., N)
xv, yv = np.meshgrid(x1, x2)

xv[25:74, 25:74] = yv[25:74, 25:74].T

x = xv
# Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
x = x/x.max()

plt.imshow(xv)
plt.title('Phantom image')
plt.colorbar()
plt.show()

np.random.seed(1)
noisy = x + 0.05 * np.random.randn(N, N)

fig = plt.figure()
im = plt.imshow(noisy)
plt.title('Noisy image')
plt.show()
     
#%%    
    
# set regularising parameter
alpha_0_tgvDen = 0.2
alpha_1_tgvDen = 1

# Define variables
u_tgvDen  = Variable((N,N))
w1_tgvDen = Variable((N,N))
w2_tgvDen = Variable((N,N))

# disc Step size
discStep = np.ones(len(u_tgvDen.shape))

# Get first regularizing term || \nabla u - w ||_{1}
tmp = GradOper(u_tgvDen.shape, discStep, direction = 'for', order = '1', bndrs = 'Neum') 
DX, DY = tmp[0], tmp[1]
first_term = sum(norm(vstack([DX * vec(u_tgvDen) - vec(w1_tgvDen), DY * vec(u_tgvDen) - vec(w2_tgvDen)]), 2, axis = 0))

# Get second regularizing term || E w ||_{1}, E is symmetrized gradient
second_term = tgv_fun(w1_tgvDen,w2_tgvDen)

# Define objective
obj_tgv_denoise =  Minimize(0.5 * sum_squares(vec(u_tgvDen) - vec(noisy)) +  \
                            alpha_0_tgvDen * first_term  + alpha_1_tgvDen * tgv_fun(w1_tgvDen,w2_tgvDen) )

prob_tgv_denoise = Problem(obj_tgv_denoise)
res_tgv_denoise = prob_tgv_denoise.solve(verbose = True, solver = MOSEK )

print()
print('Objective value is {} '.format(obj_tvDen.value))

# Show result
plt.imshow(u_tgvDen.value)
plt.title('TGV-L2 denoising')
plt.show()

# compare middle line profiles (piecewise affine structures promoted)
plt.plot(np.linspace(0,N,N), x[50,:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), u_tgvDen[50,:].value, label = 'u_tgvDen')
plt.title('Middle Line profiles')
plt.legend()
plt.show()

# show result |w|
w_tgvDen = np.sqrt( w1_tgvDen.value**2 + w2_tgvDen.value**2 )
plt.imshow(w_tgvDen)
plt.title('|w|')
plt.colorbar()
plt.show()