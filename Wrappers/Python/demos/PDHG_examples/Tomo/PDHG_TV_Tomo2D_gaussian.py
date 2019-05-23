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

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorSimple



# Create phantom for TV 2D tomography 
N = 100
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0, np.pi, N)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)

device = input('Available device: GPU==1 / CPU==0 ')

if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)

# Create noisy data. Apply Poisson noise
n1 = np.random.normal(0, 3, size=ag.shape)
noisy_data = AcquisitionData(n1 + sin.as_array(), ag)

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