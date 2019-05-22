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


Problem:     min_x, x>0  \alpha * ||\nabla x||_{2,1} + ||x-g||_{1}

             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data with Salt & Pepper Noise
             
             
             Method = 0 ( PDHG - split ) :  K = [ \nabla,
                                                 Identity]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = \nabla    
             
             
"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os, sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise

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
# Create phantom for TV Salt & Pepper denoising
N = 100

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,N))
data = loader.load(TestData.PEPPERS, size=(N,N))
ig = data.geometry
ag = ig

# Create noisy data. 
# Apply Salt & Pepper noise
# gaussian
# poisson
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    n1 = random_noise(data.as_array(), mode = noise, seed = 10)
elif noise == 'gaussian':
    n1 = random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ig.allocate()
noisy_data.fill(n1)
#noisy_data = ImageData(n1)

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

# Regularisation Parameter
alpha = .2

# fidelity
if noise == 's&p':
    f2 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f2 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f2 = L2NormSquared(b=noisy_data)

if method == '0':

    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACECHANNEL)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f1 = alpha * MixedL21Norm()
    #f2 = L1Norm(b = noisy_data)    
    f = BlockFunction(f1, f2)  
                                      
    g = ZeroFunction()
    
else:
    
    # Without the "Block Framework"
    operator = Gradient(ig)
    f =  alpha * MixedL21Norm()
    #g =  L1Norm(b = noisy_data)
    g = f2
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
opt = {'niter':2000, 'memopt': True}

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 50
pdhg.run(2000)

if data.geometry.channels > 1:
    plt.figure(figsize=(20,15))
    for row in range(data.geometry.channels):
        
        plt.subplot(3,4,1+row*4)
        plt.imshow(data.subset(channel=row).as_array())
        plt.title('Ground Truth')
        plt.colorbar()
        plt.subplot(3,4,2+row*4)
        plt.imshow(noisy_data.subset(channel=row).as_array())
        plt.title('Noisy Data')
        plt.colorbar()
        plt.subplot(3,4,3+row*4)
        plt.imshow(pdhg.get_output().subset(channel=row).as_array())
        plt.title('TV Reconstruction')
        plt.colorbar()
        plt.subplot(3,4,4+row*4)
        plt.plot(np.linspace(0,N,N), data.subset(channel=row).as_array()[int(N/2),:], label = 'GTruth')
        plt.plot(np.linspace(0,N,N), pdhg.get_output().subset(channel=row).as_array()[int(N/2),:], label = 'TV reconstruction')
        plt.legend()
        plt.title('Middle Line Profiles')
    plt.show()
else:
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    plt.imshow(data.subset(channel=0).as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.imshow(noisy_data.subset(channel=0).as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(pdhg.get_output().subset(channel=0).as_array())
    plt.title('TV Reconstruction')
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
    plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()


##%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False


if cvx_not_installable:

    ##Construct problem    
    u = Variable(ig.shape)
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    # Define Total Variation as a regulariser
    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
    fidelity = pnorm( u - noisy_data.as_array(),1)
    
    # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS  
        
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)
    
    diff_cvx = numpy.abs( pdhg.get_output().as_array() - u.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output().as_array())
    plt.title('PDHG solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(u.value)
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))
