#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer

import numpy as np                           
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old, FISTA

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, FunctionOperatorComposition, BlockFunction, ScaledFunction
                      
from ccpi.optimisation.algs import FISTA                      

from skimage.util import random_noise

from timeit import default_timer as timer
def dt(steps):
    return steps[-1] - steps[-2]

# Create phantom for TV denoising

N = 100

data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data, mode = 'gaussian', mean=0, var = 0.05, seed=10)
noisy_data = ImageData(n1)


plt.imshow(noisy_data.as_array())
plt.title('Noisy data')
plt.show()

# Regularisation Parameter
alpha = 2

operator = Gradient(ig)
g = alpha * MixedL21Norm()
f = 0.5 * L2NormSquared(b = noisy_data)
    
x_init = ig.allocate()
opt = {'niter':2000}


x = FISTA(x_init, f, g, opt)

#fista = FISTA()
#fista.set_up(x_init, f, g, opt )
#fista.max_iteration = 10
#
#fista.run(2000)
#plt.figure(figsize=(15,15))
#plt.subplot(3,1,1)
#plt.imshow(fista.get_output().as_array())
#plt.title('no memopt class')



