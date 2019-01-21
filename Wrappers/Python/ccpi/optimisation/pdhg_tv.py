#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:02:15 2019

@author: evangelos
"""

import time
import scipy.misc
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt

from ccpi.framework import DataContainer
from ccpi.optimisation.funcs import Identity, Norm1 

from my_changes import *

#%%

#  Noisy data
data = scipy.misc.ascent()
data = data/np.max(data)

N, M = data.shape
np.random.seed(10)

noise = 's&p'

if noise == 'gaussian':
    noisy_data = DataContainer(random_noise(data,'gaussian', mean = 0, var = 0.01))
    fidelity = Norm2sq_new(Identity(), noisy_data, c = 0.5)
elif noise == 's&p':    
    noisy_data = DataContainer(random_noise(data, 's&p', amount = 0.1))
    fidelity = L1Norm(Identity(), noisy_data, c = 1)
elif noise == 'poisson':    
    scale = 0.03
    noisy_image = scale * np.random.poisson(data/scale)   


#%%  
    
alpha = 1
# create regulariser
regulariser = TV(alpha)

# step sizes
tau =  0.95/regulariser.L
sigma = 0.95/regulariser.L

# create operator
operator = regulariser.op

# Options
opt = {'tol': 1e-7, 'iter': 1000, 'show_iter': 100, 'memopt':False}

regulariser.memopt = False
FiniteDiff.memopt = False

res, total_time, obj_value, error = PDHG(noisy_data, regulariser, fidelity, \
                                         operator, tau = tau, sigma = sigma, opt = opt )

plt.gray()
plt.imshow(res.as_array())
plt.colorbar()
plt.show()



