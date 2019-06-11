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

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, IndicatorBox
                      
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.framework import TestData
from PIL import Image
import os, sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise

import scipy.io

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 1

# Load 256 shepp-logan
data256 = scipy.io.loadmat('phantom.mat')['phantom256']
data = ImageData(numpy.array(Image.fromarray(data256).resize((256,256))))
N, M = data.shape
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=M)

# Add it to testdata or use tomophantom
#loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
#data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(50, 50))
#ig = data.geometry

# Create acquisition data and geometry
detectors = N
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors)

# Select device
device = '0'
#device = input('Available device: GPU==1 / CPU==0 ')
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
    alpha = 3
    f2 = KullbackLeibler(noisy_data)  
    g =  IndicatorBox(lower=0) 
    sigma = 1
    tau = 1/(sigma*normK**2)    
    
elif noise == 'gaussian':   
    alpha = 20
    f2 = 0.5 * L2NormSquared(b=noisy_data)                                         
    g = ZeroFunction()
    sigma = 10
    tau = 1/(sigma*normK**2)     
    
f1 = alpha * MixedL21Norm() 
f = BlockFunction(f1, f2)    

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 200
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
plt.title('TV Reconstruction')
plt.colorbar()
plt.show()
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()
