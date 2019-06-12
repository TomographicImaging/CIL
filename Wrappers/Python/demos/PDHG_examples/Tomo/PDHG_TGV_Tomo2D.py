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

Total Generalised Variation (TGV) Tomography 2D using PDHG algorithm:


Problem:     min_{x>0} \alpha * ||\nabla x - w||_{2,1} + \beta *  || E w ||_{2,1} +
                       \frac{1}{2}||Au - g||^{2}
                     
            min_{u>0} \alpha * ||\nabla u - w||_{2,1} + \beta *  || E w ||_{2,1} +
                      int A u - g log(Au + \eta)                     

             \alpha: Regularization parameter
             \beta: Regularization parameter
             
             \nabla: Gradient operator 
              E: Symmetrized Gradient operator
              A: System Matrix
             
             g: Noisy Sinogram 
                          
              K = [ \nabla, - Identity
                   ZeroOperator, E 
                   A, ZeroOperator]
                                         
"""

from ccpi.framework import AcquisitionGeometry, AcquisitionData, ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient, Identity, \
                                    SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import IndicatorBox, KullbackLeibler, ZeroFunction,\
                      MixedL21Norm, BlockFunction, L2NormSquared

from ccpi.astra.ops import AstraProjectorSimple
import os, sys


import tomophantom
from tomophantom import TomoP2D

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 1 
    
# Load Piecewise smooth Shepp-Logan phantom 
model = 2 # select a model number from the library
N = 128 # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
#This will generate a N_size x N_size phantom (2D)
phantom_2D = TomoP2D.Model(model, N, path_library2D)

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)

#Create Acquisition Data 
detectors = N
angles = np.linspace(0, np.pi, N)
ag = AcquisitionGeometry('parallel','2D',angles, detectors)

#device = input('Available device: GPU==1 / CPU==0 ')
device = '1'
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'

Aop = AstraProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(data)

# Create noisy sinogram.
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

# Create Operators
op11 = Gradient(ig)
op12 = Identity(op11.range_geometry())

op22 = SymmetrizedGradient(op11.domain_geometry())    
op21 = ZeroOperator(ig, op22.range_geometry())
    
op31 = Aop
op32 = ZeroOperator(op22.domain_geometry(), ag)

operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
normK = operator.norm()

# Create functions
if noise == 'poisson':
    alpha = 3
    beta = 6
    f3 = KullbackLeibler(noisy_data)    
    g =  BlockFunction(IndicatorBox(lower=0), ZeroFunction()) 
    
    # Primal & dual stepsizes
    sigma = 1
    tau = 1/(sigma*normK**2)    
    
elif noise == 'gaussian':   
    alpha = 20
    beta = 50
    f3 = 0.5 * L2NormSquared(b=noisy_data)                                         
    g = BlockFunction(ZeroFunction(), ZeroFunction())
    
    # Primal & dual stepsizes
    sigma = 10
    tau = 1/(sigma*normK**2)     
    
f1 = alpha * MixedL21Norm()
f2 = beta * MixedL21Norm()  
f = BlockFunction(f1, f2, f3)         
    
# Compute operator Norm
normK = operator.norm()

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 3000
pdhg.update_objective_interval = 500
pdhg.run(3000)
#%%
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
plt.imshow(pdhg.get_output()[0].as_array())
plt.title('TGV Reconstruction')
plt.colorbar()
plt.show()
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[:, int(N/2)], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output()[0].as_array()[:, int(N/2)], label = 'TGV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


