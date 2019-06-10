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


Problem:     min_{x>0} \alpha * ||\nabla x - w||_{2,1} +
                     \beta *  || E w ||_{2,1} +
                     int A x - g log(Ax + \eta)

             \alpha: Regularization parameter
             \beta: Regularization parameter
             
             \nabla: Gradient operator 
              E: Symmetrized Gradient operator
              A: Projection Matrix
             
             g: Noisy Data with Poisson Noise
                          
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
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorSimple
from ccpi.framework import TestData
import os, sys
from skimage.util import random_noise

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))

# Load Data                      
#N = 50
#M = 50
#data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))
#
#ig = data.geometry
#ag = ig
N = 100

data = np.zeros((N,N))

x1 = np.linspace(0, int(N/2), N)
x2 = np.linspace(int(N/2), 0., N)
xv, yv = np.meshgrid(x1, x2)

xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T

data = xv
data = ImageData(data/data.max())

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig


#Create Acquisition Data and apply poisson noise
detectors = N
angles = np.linspace(0, np.pi, N)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)

#device = input('Available device: GPU==1 / CPU==0 ')
device = '0'
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'

Aop = AstraProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(data)

# Create noisy data. Apply Poisson noise
scale = 0.5
eta = 0 
n1 = scale * np.random.poisson(eta + sin.as_array()/scale)

noisy_data = AcquisitionData(n1, ag)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()
#%%
# Regularisation Parameters
alpha = 1
beta =  5

# Create Operators
op11 = Gradient(ig)
op12 = Identity(op11.range_geometry())

op22 = SymmetrizedGradient(op11.domain_geometry())    
op21 = ZeroOperator(ig, op22.range_geometry())
    
op31 = Aop
op32 = ZeroOperator(op22.domain_geometry(), ag)

operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
    
f1 = alpha * MixedL21Norm()
f2 = beta * MixedL21Norm() 
f3 = KullbackLeibler(noisy_data)    
f = BlockFunction(f1, f2, f3)         

g =  BlockFunction(IndicatorBox(lower=0), ZeroFunction())
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 3000
pdhg.update_objective_interval = 500
pdhg.run(3000)

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
## 
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), pdhg.get_output()[0].as_array()[int(N/2),:], label = 'TGV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


