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

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from skimage.util import random_noise

# Create phantom for TV Gaussian denoising
import timeit
import os
from tomophantom import TomoP3D
import tomophantom

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 13 # select a model number from the library
N = 64 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

#This will generate a N x N x N phantom (3D)
phantom_tm = TomoP3D.Model(model, N, path_library3D)

# Create noisy data. Add Gaussian noise
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_num_z=N)
ag = ig
n1 = random_noise(phantom_tm, mode = 'gaussian', mean=0, var = 0.001, seed=10)
noisy_data = ImageData(n1)

sliceSel = int(0.5*N)
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(noisy_data.as_array()[sliceSel,:,:],vmin=0, vmax=1)
plt.title('Axial View')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(noisy_data.as_array()[:,sliceSel,:],vmin=0, vmax=1)
plt.title('Coronal View')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(noisy_data.as_array()[:,:,sliceSel],vmin=0, vmax=1)
plt.title('Sagittal View')
plt.colorbar()
plt.show()   


# Regularisation Parameter
alpha = 0.05

method = '0'

if method == '0':

    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions
      
    f1 = alpha * MixedL21Norm()
    f2 = 0.5 * L2NormSquared(b = noisy_data)    
    f = BlockFunction(f1, f2)  
                                      
    g = ZeroFunction()
    
else:
    
    # Without the "Block Framework"
    operator = Gradient(ig)
    f =  alpha * MixedL21Norm()
    g =  0.5 * L2NormSquared(b = noisy_data)
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 200
pdhg.run(2000)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

plt.subplot(2,3,1)
plt.imshow(noisy_data.as_array()[sliceSel,:,:],vmin=0, vmax=1)
plt.axis('off')
plt.title('Axial View')

plt.subplot(2,3,2)
plt.imshow(noisy_data.as_array()[:,sliceSel,:],vmin=0, vmax=1)
plt.axis('off')
plt.title('Coronal View')

plt.subplot(2,3,3)
plt.imshow(noisy_data.as_array()[:,:,sliceSel],vmin=0, vmax=1)
plt.axis('off')
plt.title('Sagittal View')


plt.subplot(2,3,4)
plt.imshow(pdhg.get_output().as_array()[sliceSel,:,:],vmin=0, vmax=1)
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(pdhg.get_output().as_array()[:,sliceSel,:],vmin=0, vmax=1)
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(pdhg.get_output().as_array()[:,:,sliceSel],vmin=0, vmax=1)
plt.axis('off')
im = plt.imshow(pdhg.get_output().as_array()[:,:,sliceSel],vmin=0, vmax=1)


fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)


plt.show()

