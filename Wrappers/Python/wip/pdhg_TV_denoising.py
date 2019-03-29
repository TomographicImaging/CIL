#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer

import numpy as np                           
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFun, L2NormSquared, \
                      MixedL21Norm, FunctionOperatorComposition, BlockFunction, ScaledFunction

from skimage.util import random_noise

# ############################################################################
# Create phantom for TV denoising

N = 200
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data, mode='gaussian', seed=10)
noisy_data = ImageData(n1)


#%%

# Regularisation Parameter
alpha = 200

#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
method = '0'
if method == '0':

    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)

    # Form Composite Operator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    #### Create functions
#    f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
#                                    L2NormSq(0.5, b = noisy_data) )    
    
    f1 = alpha * MixedL21Norm()
    f2 = 0.5 * L2NormSquared(b = noisy_data)
    
    f = BlockFunction(f1, f2 )                                        
    g = ZeroFun()
    
else:
    
    ###########################################################################
    #         No Composite #
    ###########################################################################
    operator = Gradient(ig)
    f = alpha * FunctionOperatorComposition(operator, MixedL21Norm())
    g = 0.5 * L2NormSquared(b = noisy_data)
    ###########################################################################
#%%
    
# Compute operator Norm
normK = operator.norm()
print ("normK", normK)
# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


#%%
## Number of iterations
opt = {'niter':2000}
##
### Run algorithm
result, total_time, objective = PDHG(f, g, operator, \
                                  tau = tau, sigma = sigma, opt = opt)
#%%
###Show results
if isinstance(result, BlockDataContainer):
    sol = result.get_item(0).as_array()
else:
    sol = result.as_array()
    
#sol = result.as_array()
#
plt.imshow(sol)
plt.colorbar()
plt.show()
#
###
plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.show()
##
plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
plt.legend()
plt.show()


#%%
#
