#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:44:54 2018

@author: evangelos
"""

# Confirm solutions usign CVX template
###############################################################################
    # Multichannel TV
    # Different forms of TV that expole isotropic/anisotropic channel coupling
    # [1] "Collaborative Total Variation: A General Framework for Vectorial TV Models"
    
    # 1)  Anis TV --> No coupling, sum of TV for each channel: see [1], l^{1,1,1}
    # 2)  Iso TV --> Euclidean norm of TV for each channel: see [1], l^{2,1,1}
    # 3)  Isol2 TV: see [1], l^{2, 2, 1}
    # 4)  Anis_linf: see [1], l^{\inf, 1, 1}

###############################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.misc
from cvx_functions import *
from cvxpy import *
from skimage import data, img_as_float, transform, exposure
from skimage.util import random_noise

#%% Load color image from skimage, resize it and add noise

N = 100
cat = img_as_float(data.chelsea())
cat_res = transform.resize(cat, [N, N, 3], order=1, mode='reflect', cval=0, clip=True, anti_aliasing=True)
gtruth_res_color = exposure.rescale_intensity(cat_res,(0,1))

rows, cols, channels = gtruth_res_color.shape

np.random.seed(1)
noisy_color = gtruth_res_color + 0.05 * np.random.randn(N, N, 3)

plt.imshow(gtruth_res_color)
plt.title('Gtruth')
plt.show()

plt.imshow(noisy_color)
plt.title('Noisy')
plt.show()
    
# set regularising parameter
alpha_tvColorDen = 0.1

# Define the problem, each variable for each RGB
variables = []
fidelity = 0
  
for i in range(channels):
    u_channel = Variable(shape=(rows, cols))
    variables.append(u_channel)
    # L2 norm fidelity, add sum_squares per channel
    fidelity += sum_squares( variables[i] - noisy_color[:,:,i])
    
# Choose form of vectorial TV, aniso, iso, isol2, anis_linf
#vec_tv = tv_funVec(variables, 'isol2') #--> use MOSEK
#vec_tv = tv_funVec(variables, 'aniso') #--> use MOSEK
#vec_tv = tv_funVec(variables, 'iso') #-->  use SCS ( fails with MOSEK, error about dynamic range, cannot fix it )
vec_tv = tv_funVec(variables, 'anis_linf') #-->  use MOSEK     

obj_tvColorDen =  Minimize( 0.5 * fidelity +  alpha_tvColorDen * vec_tv )
prob_tvColorDen = Problem(obj_tvColorDen)

#res_tvColorDen = prob_tvColorDen.solve(verbose = True, solver = SCS, eps = 1e-9)
res_tvColorDen = prob_tvColorDen.solve(verbose = True, solver = MOSEK)

print()
print('Objective value is {} '.format(obj_tvColorDen.value))

# add back all the channels
u_tvColorDen = np.zeros((rows, cols, channels))
for i in range(channels):
    u_tvColorDen[:, :, i] = variables[i].value

# Show result
plt.imshow(noisy_color)
plt.title('Noisy')
plt.show()

plt.imshow(gtruth_res_color)
plt.title('Gtruth')
plt.show()

plt.imshow(u_tvColorDen)
plt.title('Denoised')
plt.show()


#%% Confirm multichannel TV with cvx

#u = np.random.randint(10, size = (3, 3, 2))
#
#vec_tv_aniso = tv_funVec(u, 'aniso')
#d1 = tv_fun(u[0]) + tv_fun(u[1]) + tv_fun(u[2])
#print(d1.value)
#print()
#print(vec_tv_aniso.value)
#
#vec_tv_iso = tv_funVec(u, 'iso')
#d2 = sqrt(tv_fun(u[0])**2 + tv_fun(u[1])**2 + tv_fun(u[2])**2)
#print(d2.value)
#print()
#print(vec_tv_iso.value)
#
#
#%%
#vec_tv_isol2 = tv_funVec(u, 'isol2')
#discStep = [1, 1]
#grad = GradOper(u[0].shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')
#
#d3 = (grad[0]*u[0].flatten('F'))**2 + (grad[1]*u[0].flatten('F'))**2 + \
#     (grad[0]*u[1].flatten('F'))**2 + (grad[1]*u[1].flatten('F'))**2 + \
#     (grad[0]*u[2].flatten('F'))**2 + (grad[1]*u[2].flatten('F'))**2 
#      
#d3 = sum(sqrt(d3))
#print(d3.value)
#print()
#print(vec_tv_isol2.value)
#
##%%
#
#vec_tv_anis_linf = tv_funVec(u, 'anis_linf')
#
#discStep = [1, 1]
#grad = GradOper(u[0].shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')
#
#
#d4_1 = norm(vstack([grad[0]*u[0].flatten('F'), grad[0]*u[1].flatten('F'), grad[0]*u[2].flatten('F')]), 'inf')
#d4_2 = norm(vstack([grad[1]*u[0].flatten('F'), grad[1]*u[1].flatten('F'), grad[1]*u[2].flatten('F')]), 'inf')
#
#print(sum( d4_1 + d4_2 ).value)
#
#print(vec_tv_anis_linf.value)


