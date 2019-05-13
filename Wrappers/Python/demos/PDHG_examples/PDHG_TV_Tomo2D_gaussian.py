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


Problem:     min_x, x>0  \alpha * ||\nabla x||_{2,1} + \frac{1}{2}||Ax - g||^{2}

             \nabla: Gradient operator 
             
             A: Projection Matrix
             g: Noisy sinogram corrupted with Gaussian Noise
             
             \alpha: Regularization parameter
 
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData, TestData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.operators import AstraProjectorSimple
import sys, os

# Create phantom for TV Gaussian denoising
import tomophantom
from tomophantom import TomoP2D

import sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise


if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0

# Create phantom for TV 2D tomography 
N = 100
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,N))
ig = data.geometry

detectors = int( N * numpy.sqrt(2) )
angles = np.linspace(0, 180., N, dtype=numpy.float32)

ag = AcquisitionGeometry('parallel','2D',angles / 180. * numpy.pi, detectors)


print ("Building 2D phantom using TomoPhantom software")
model = 1 # select a model number from the library
#N = 64 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library = os.path.join(path, "Phantom2DLibrary.dat")

#This will generate a N x N x N phantom (3D)
phantom_tm = TomoP2D.ModelSino(model, N, detectors, angles, path_library)

# Create noisy data. 
# Apply Salt & Pepper noise
# gaussian
# poisson
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = random_noise(phantom_tm, mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    n1 = random_noise(phantom_tm, mode = noise, seed = 10)
elif noise == 'gaussian':
    n1 = random_noise(phantom_tm, mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
#phantom_tm = ImageData(n1)
phantom_tm += n1
#%%
device = input('Available device: GPU==1 / CPU==0 ')

#sin = ag.allocate()
data = ImageData(TomoP2D.Model(1, N, path_library))
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
sinastra = Aop.direct(data)

# Create noisy data. Apply Poisson noise
#n1 = np.random.normal(0, 3, size=ig.shape)
#np.random.normal(0, 3, size=ag.shape)
noisy_data = ag.allocate()
noisy_data.fill(phantom_tm)
#noisy_data = sin

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(sinastra.as_array())
plt.title('Astra Sinogram')
plt.colorbar()

plt.show()

# Regularisation Parameter
alpha = 50

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions
      
f1 = alpha * MixedL21Norm()
f2 = 0.5 * L2NormSquared(b=noisy_data)    
f = BlockFunction(f1, f2)  
                                      
g = ZeroFunction()

# Compute operator Norm
normK = operator.norm()
    
# Primal & dual stepsizes
sigma = 10
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 200
pdhg.run(2000)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('TV Reconstruction')
plt.colorbar()
plt.show()
## 
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff
import astra
import numpy

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False
    
if cvx_not_installable:
    
    ##Construct problem    
    u = Variable(N*N)
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')

    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
    
    # create matrix representation for Astra operator
    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)

    proj_id = astra.create_projector('strip', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)

    ProjMat = astra.matrix.get(matrix_id)
    
    tmp = noisy_data.as_array().ravel()
    
    fidelity = 0.5 * sum_squares(ProjMat * u - tmp)

    solver = MOSEK
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)    
         
    diff_cvx = numpy.abs( pdhg.get_output().as_array() - np.reshape(u.value, (N,N) ))
           
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output().as_array())
    plt.title('PDHG solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(np.reshape(u.value, (N, N)))
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,N,N), np.reshape(u.value, (N,N) )[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))