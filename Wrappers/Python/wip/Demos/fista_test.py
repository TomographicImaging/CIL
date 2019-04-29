# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import FISTA, PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
from ccpi.optimisation.functions import L2NormSquared, L1Norm, \
                                        MixedL21Norm, FunctionOperatorComposition, BlockFunction, ZeroFunction
                                                
from skimage.util import random_noise

# Create phantom for TV Gaussian denoising
N = 100

data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
data = ImageData(data)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data.as_array(), mode = 's&p', salt_vs_pepper = 0.9, amount=0.2)
noisy_data = ImageData(n1)

# Regularisation Parameter
alpha = 5

operator = Gradient(ig)

#fidelity = L1Norm(b=noisy_data)
#regulariser = FunctionOperatorComposition(alpha * L2NormSquared(), operator)

fidelity = FunctionOperatorComposition(alpha * MixedL21Norm(), operator)
regulariser = 0.5 * L2NormSquared(b = noisy_data)

x_init = ig.allocate()

## Setup and run the PDHG algorithm
opt = {'tol': 1e-4, 'memopt':True}
fista = FISTA(x_init=x_init , f=regulariser, g=fidelity, opt=opt)
fista.max_iteration = 2000
fista.update_objective_interval = 50
fista.run(2000, verbose=True)

plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(fista.get_output().as_array())
plt.title('TV Reconstruction')
plt.colorbar()
plt.show()

# Compare with PDHG
method = '0'
#
if method == '0':
#
#    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)
#
#    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) )   
    f = BlockFunction(alpha * L2NormSquared(), fidelity)                                        
    g = ZeroFunction()
  
## Compute operator Norm
normK = operator.norm()
#
## Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
#
#
## Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 50
pdhg.run(2000)
#
#%%
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(fista.get_output().as_array())
plt.title('FISTA')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(pdhg.get_output().as_array())
plt.title('PDHG')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(np.abs(pdhg.get_output().as_array()-fista.get_output().as_array()))
plt.title('Diff FISTA-PDHG')
plt.colorbar()
plt.show()


