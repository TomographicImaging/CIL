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


Problem:     min_x, x>0  \alpha * ||\nabla x||_{2,1} + int A x -g log(Ax + \eta)

             \nabla: Gradient operator 
             
             A: Projection Matrix
             g: Noisy sinogram corrupted with Poisson Noise
             
             \eta: Background Noise
             \alpha: Regularization parameter
 
"""

from ccpi.framework import AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import KullbackLeibler, \
                      MixedL21Norm, BlockFunction, IndicatorBox

from ccpi.astra.ops import AstraProjectorSimple
from ccpi.framework import TestData
import os, sys



loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))

# Load Data                      
N = 50
M = 50
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
n1 = np.random.poisson(eta + sin.as_array())

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
# Regularisation Parameter
alpha = 2

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1,op2, shape=(2,1) ) 

# Create functions
      
f1 = alpha * MixedL21Norm() 
f2 = KullbackLeibler(noisy_data)    
f = BlockFunction(f1, f2)  
                                      
g = IndicatorBox(lower=0)


# Compute operator Norm
normK = operator.norm()
    
# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 3000
pdhg.update_objective_interval = 500
pdhg.run(3000, verbose = True)

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
## 
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution
#
#from ccpi.optimisation.operators import SparseFiniteDiff
#import astra
#import numpy
#
#try:
#    from cvxpy import *
#    cvx_not_installable = True
#except ImportError:
#    cvx_not_installable = False
#
#
#if cvx_not_installable:
#    
#
#    ##Construct problem    
#    u = Variable(N*N)
#    q = Variable()
#    
#    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
#    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
#
#    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
#    
#    # create matrix representation for Astra operator
#
#    vol_geom = astra.create_vol_geom(N, N)
#    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)
#
#    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
#
#    matrix_id = astra.projector.matrix(proj_id)
#
#    ProjMat = astra.matrix.get(matrix_id)
#    
#    tmp = noisy_data.as_array().ravel()
#    
#    fidelity = sum(kl_div(tmp, ProjMat * u))    
#    
#    constraints = [q>=fidelity, u>=0]   
#    solver = SCS
#    obj =  Minimize( regulariser +  q)
#    prob = Problem(obj, constraints)
#    result = prob.solve(verbose = True, solver = solver)    
#             
#    diff_cvx = np.abs(pdhg.get_output().as_array() - np.reshape(u.value, (N, N)))
#           
#    plt.figure(figsize=(15,15))
#    plt.subplot(3,1,1)
#    plt.imshow(pdhg.get_output().as_array())
#    plt.title('PDHG solution')
#    plt.colorbar()
#    plt.subplot(3,1,2)
#    plt.imshow(np.reshape(u.value, (N, N)))
#    plt.title('CVX solution')
#    plt.colorbar()        
#    plt.subplot(3,1,3)
#    plt.imshow(diff_cvx)
#    plt.title('Difference')
#    plt.colorbar()
#    plt.show()    
#    
#    plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
#    plt.plot(np.linspace(0,N,N), np.reshape(u.value, (N, N))[int(N/2),:], label = 'CVX')
#    plt.legend()
#    plt.title('Middle Line Profiles')
#    plt.show()
#            
#    print('Primal Objective (CVX) {} '.format(obj.value))
#    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))