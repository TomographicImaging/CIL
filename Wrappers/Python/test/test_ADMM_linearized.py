#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:09:58 2019

@author: evangelos
"""

#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================


import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, ADMM_linearized

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os
import sys

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
print ("Applying {} noise")

if len(sys.argv) > 2:
    method = sys.argv[2]
else:
    method = '0'
print ("method ", method)


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SHAPES)
ig = data.geometry
ag = ig

# Create noisy data. 
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    scale = 5
    n1 = TestData.random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ig.allocate()
noisy_data.fill(n1)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Regularisation Parameter depending on the noise distribution
if noise == 's&p':
    alpha = 0.8
elif noise == 'poisson':
    alpha = 1
elif noise == 'gaussian':
    alpha = 0.3

# fidelity
if noise == 's&p':
    f2 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f2 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f2 = 0.5 * L2NormSquared(b=noisy_data)

if method == '0':

    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f = BlockFunction(alpha * MixedL21Norm(), f2) 
    g = ZeroFunction()
    
else:
    
    operator = Gradient(ig)
    f =  alpha * MixedL21Norm()
    g = f2
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 500
pdhg.update_objective_interval = 100
pdhg.run(2000)

# Setup and run the PDHG algorithm
admm = ADMM_linearized(f=g, g=f, operator=operator, tau=tau, sigma=sigma)
admm.max_iteration = 500
admm.update_objective_interval = 100
admm.run(2000)
        
plt.figure(figsize=(15,15))

plt.subplot(4,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()

plt.subplot(4,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()

plt.subplot(4,1,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('PDHG')
plt.colorbar()

plt.subplot(4,1,4)
plt.imshow(admm.get_output().as_array())
plt.title('ADMM')
plt.colorbar()

plt.show()

plt.imshow(np.abs(admm.get_output().as_array() - pdhg.get_output().as_array()))
plt.colorbar()
plt.show()

