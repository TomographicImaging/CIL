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

""" 

Total Variation 2D Tomography Reconstruction using PDHG algorithm:


Problem:     min_u  \alpha * ||\nabla u||_{2,1} + \frac{1}{2}||Au - g||^{2}
             min_u, u>0  \alpha * ||\nabla u||_{2,1} + \int A u  - g log (Au + \eta)

             \nabla: Gradient operator              
             A: System Matrix
             g: Noisy sinogram 
             \eta: Background noise
             
             \alpha: Regularization parameter
 
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, ADMM_linearized

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, IndicatorBox
                      
from ccpi.astra.operators import AstraProjectorSimple

import os, sys
import tomophantom
from tomophantom import TomoP2D

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
    
model = 1 # select a model number from the library
N = 128 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

phantom_2D = TomoP2D.Model(model, N, path_library2D)    
data = ImageData(phantom_2D)
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)

# Create acquisition data and geometry
detectors = N
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors)

# Select device
device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)

# Create noisy data. Apply Gaussian noise
noises = ['gaussian', 'poisson']
noise = noises[which_noise]

if noise == 'poisson':
    scale = 5
    eta = 0
    noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
elif noise == 'gaussian':
    n1 = np.random.normal(0, 1, size = ag.shape)
    noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
    
else:
    raise ValueError('Unsupported Noise ', noise)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(1,2,2)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,1)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Compute operator Norm
normK = operator.norm()

# Create functions
if noise == 'poisson':
    
    alpha = 2
    f2 = KullbackLeibler(noisy_data)  
    g =  IndicatorBox(lower=0)    
    sigma = 1
    tau = 1/(sigma*normK**2)     
        
elif noise == 'gaussian':   
    
    alpha = 10
    f2 = 0.5 * L2NormSquared(b=noisy_data)                                         
    g = ZeroFunction()
    sigma = 10
    tau = 1/(sigma*normK**2) 
    
f1 = alpha * MixedL21Norm() 
f = BlockFunction(f1, f2)   

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000)

# Setup and run the PDHG algorithm
admm = ADMM_linearized(f=g, g=f, operator=operator, tau=tau, sigma=sigma)
admm.max_iteration = 1000
admm.update_objective_interval = 200
admm.run(1000)


#%%
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



