#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:44:54 2018

@author: evangelos
"""

## Confirm solutions usign CVX template
#  1) ROF denoising, ( TV - L2 )
#  2) TGV - L2 denosiing 
#TODO  3) Infimal convolution total variation  
#TODO  Quality measures, implement our own? or use other libraries
#TODO  Color TV, TGV?

# CVXPY has many solvers, but for more accuracy we prefer MOSEK. Need academic licence.
# Otherwise, we can use SCS which is fast but less accurate.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gradOperator import *
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

noisy = x + 0.05 * np.random.randn(N, N)

fig = plt.figure()
im = plt.imshow(noisy)
plt.title('Noisy image')
plt.show()
     
#%%    
    
# set regularising parameter
alpha_0_tgvDen = 0.2
alpha_1_tgvDen = 0.4

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



#%%

###############################################################################
    # COLOR ROF denoising ( Vector Total Variation )
    # Apply TV for each channel (no correlation), i.e., TV(u_R) + TV(u_G) + TV(u_B)
    # or sqrt( TV(u_R)^2 + TV(u_G)^2 + TV(u_B)^2 )
    
###############################################################################

N = 100
import scipy.misc
face = scipy.misc.face()

# Resize and rescale it to [0 1]
gtruth_color = scipy.misc.imresize(face, (N, N, 3), interp='bilinear', mode = None)
gtruth_res_color = gtruth_color/gtruth_color.max()

rows, cols, channels = gtruth_res_color.shape
noisy_color = gtruth_res_color + 0.055 * np.random.randn(N, N, 3)

# Show gtruth and noisy images
plt.imshow(gtruth_res_color)
plt.title('Gtruth')
plt.show()

plt.imshow(noisy_color)
plt.title('Noisy')
plt.show()
    
# set regularising parameter
alpha_tvColorDen = 0.4

# Define the problem, each variable for each RGB
variables = []
fidelity = 0
  
for i in range(channels):
    u_channel = Variable(shape=(rows, cols))
    variables.append(u_channel)
    # L2 norm fidelity, add sum_squares per channel
    fidelity += sum_squares( variables[i] - noisy_color[:,:,i])
    
#vec_tv = tv_funVec(variables, 'aniso')

grad = GradOper((rows, cols), [1, 1], direction = 'for', order = '1', bndrs = 'Neum')

vec_tv = tv_funVec(variables, 'aniso')

#vec_tv = sum(norm(vstack([ grad[0] * vec(variables[0]), grad[1] * vec(variables[0]), \
#                           grad[0] * vec(variables[1]), grad[1] * vec(variables[1]), \
#                           grad[0] * vec(variables[2]), grad[1] * vec(variables[2]) ]), 2, axis = 0 ))
    

 

obj_tvColorDen =  Minimize( 0.5 * fidelity +  alpha_tvColorDen * vec_tv )
prob_tvColorDen = Problem(obj_tvColorDen)

res_tvColorDen = prob_tvColorDen.solve(verbose = True, solver = SCS)

print()
print('Objective value is {} '.format(obj_tvColorDen.value))

# add back all the channels
u_tvColorDen = np.zeros((rows, cols, channels))
for i in range(channels):
    u_tvColorDen[:, :, i] = variables[i].value

# Show result
plt.imshow(u_tvColorDen)
plt.title('Color TV denoising')
plt.colorbar()
plt.show()

#%%

u1 = np.random.randint(10, size = (3,3,2))

norm( tv_funVec(variables, 'iso'), 2, axis = 0 ).value

#norm(tv_funVec(u1, 'iso'),2).value
    






#%%


def gauss_kernel(p2, sigma=0.5):
#        
    siz = (p2 - 1.)/2.
    y,x = np.ogrid[-siz[1]:siz[1]+1,-siz[0]:siz[0]+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h  
#
#
gkernel = gauss_kernel(np.array([5,5]),sigma=3)
#
blurred_noisy = signal.convolve2d(x, d, boundary='symm', mode='same') + 0.025*np.random.randn(N, N)
#
#
#plt.imshow(blurred_noisy)
#plt.title('Blurred and noisy')
#plt.colorbar()
#plt.show()
#
## Cannot work through cvxpy and deblurring, now conv2d function in the cvxpy framework
#
## set regularising parameter
#lam_denoising = 0.1
#
## Define the problem
#x1_blurred_noisy = Variable((N,N))s
#
#
##conv_op = signal.convolve2d(x1_blurred_noisy, d, boundary='symm', mode='same')
#
#objective1_deblur =  Minimize(0.5*sum_squares(x1_blurred_noisy - blurred_noisy) +  lam_denoising * tv_function(x1_blurred_noisy) )
#prob1_deblur = Problem(objective1_deblur)
#
## Choose solverl, SCS is fast but less accurate than MOSEK
##result1_denoise = prob1_denoise.solve(verbose=True,solver=SCS,eps=1e-12)
#result1_deblur = prob1_deblur.solve(verbose=True,solver=MOSEK)
#
## The optimal solution for x is stored in x.value and optimal objective value 
## is in result as well as in objective.value
#print("CVXPY least squares plus 1-norm solution and objective value:")
#print(x1_blurred_noisy.value)
#print(objective1_deblur.value)
#
## Get the value of the cvx expression
#x1_cvx = x1_blurred_noisy.value
#
#
#plt.imshow(x1_cvx)
#plt.title('Blurred and noisy')
#plt.colorbar()
#plt.show()

#%%
