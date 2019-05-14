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

Total Variation Denoising using PDHG algorithm:

Problem:     min_x, x>0  \alpha * ||\nabla x||_{2}^{2} + int A x -g log(Ax + \eta)

             \nabla: Gradient operator 
             
             A: Projection Matrix
             g: Noisy sinogram corrupted with Poisson Noise
             
             \eta: Background Noise
             \alpha: Regularization parameter
             
                
                       
"""


from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import IndicatorBox, L2NormSquared, BlockFunction
from skimage.util import random_noise
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.framework import TestData
import os, sys

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))

# Load Data                      
N = 100
M = 100
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))

ig = data.geometry
ag = ig

#Create Acquisition Data and apply poisson noise

detectors = N
angles = np.linspace(0, np.pi, N)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)

device = input('Available device: GPU==1 / CPU==0 ')

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


# Regularisation Parameter
alpha = 500

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions
      
f1 = alpha * L2NormSquared()
f2 = 0.5 * L2NormSquared(b=noisy_data)    
f = BlockFunction(f1, f2)  
                                      
g = IndicatorBox(lower=0)
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 500
pdhg.run(2000)

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
plt.imshow(pdhg.get_output().as_array())
plt.title('Tikhonov Reconstruction')
plt.colorbar()
plt.show()
## 
plt.plot(np.linspace(0,N,M), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,M), pdhg.get_output().as_array()[int(N/2),:], label = 'Tikhonov reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


