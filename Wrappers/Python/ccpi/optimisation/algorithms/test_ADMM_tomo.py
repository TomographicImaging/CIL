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

Tikhonov 2D Tomography Reconstruction using PDHG algorithm:


Problem:     min_u  \alpha * ||\nabla u||_{2}^{2} + \frac{1}{2}||Au - g||^{2}
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

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, MixedL21Norm, \
                       BlockFunction, KullbackLeibler, IndicatorBox
                      
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
    alpha = 20
    f2 = KullbackLeibler(noisy_data)  
    g =  IndicatorBox(lower=0) 
    sigma = 1
    tau = 1/(sigma*normK**2)    
    
elif noise == 'gaussian':   
    alpha = 200
    f2 = 0.5 * L2NormSquared(b=noisy_data)                                         
    g = ZeroFunction()
    sigma = 10
    tau = 1/(sigma*normK**2)     
    
f1 = alpha * L2NormSquared() 
f = BlockFunction(f1, f2)    

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000)

#%%

#g1new = f2
#g2new = alpha * MixedL21Norm()
gnew = f 
fnew = g

sigma_new = 0.1
tau_new = sigma_new/(operator.norm()**2)

x = operator.domain_geometry().allocate()
z = operator.range_geometry().allocate()
u = operator.range_geometry().allocate()

x0 = operator.domain_geometry().allocate()
z0 = operator.range_geometry().allocate()
u0 = operator.range_geometry().allocate()

x1 = operator.domain_geometry().allocate()
z1 = operator.range_geometry().allocate()
u1 = operator.range_geometry().allocate()

tmp1 = operator.range_geometry().allocate()
tmp3 = operator.range_geometry().allocate()
tmp2 = operator.domain_geometry().allocate()

for i in range(5000):
    
    operator.direct(x0, out = tmp1)
    tmp1 += u0 
    tmp1 += -1 * z0
    operator.adjoint(tmp1, out = tmp2)
    
    fnew.proximal( x0 - (tau_new/sigma_new) * tmp2, tau_new, out = x1)
    
    operator.direct(x1, out = tmp3)
    tmp3 += u0

    gnew.proximal(tmp3, sigma_new, out = z1)
    
    operator.direct(x1, out = u1)
    u1 += u0
    u1 -= z1
#
    u0.fill(u1)
    z0.fill(z1)
    x0.fill(x1)
        
    if i%200 == 0:
        print(i)
        plt.imshow(x0.as_array())
        plt.colorbar()
        plt.show()

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
plt.title('Tikhonov Reconstruction')
plt.colorbar()

plt.subplot(4,1,4)
plt.imshow(x0.as_array())
plt.title('Tikhonov Reconstruction')
plt.colorbar()

plt.show()

plt.imshow(np.abs(x0.as_array() - pdhg.get_output().as_array()))
plt.colorbar()
plt.show()


#plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(N/2),:], label = 'Tikhonov reconstruction')
#plt.legend()
#plt.title('Middle Line Profiles')
#plt.show()        
        
        
