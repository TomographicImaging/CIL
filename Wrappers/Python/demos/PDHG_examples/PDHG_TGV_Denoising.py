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

Total Generalised Variation (TGV) Denoising using PDHG algorithm:


Problem:     min_{x} \alpha * ||\nabla x - w||_{2,1} +
                     \beta *  || E w ||_{2,1} +
                     \frac{1}{2} * || x - g ||_{2}^{2}

             \alpha: Regularization parameter
             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
              E: Symmetrized Gradient operator
             
             g: Noisy Data with Salt & Pepper Noise
                          
             Method = 0 ( PDHG - split ) :  K = [ \nabla, - Identity
                                                  ZeroOperator, E 
                                                  Identity, ZeroOperator]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = [ \nabla, - Identity
                                                  ZeroOperator, E ] 
                                                                
"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, \
                        Gradient, SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction
import sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise


if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
# Create phantom for TGV SaltPepper denoising

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
noisy_data = ImageData(n1)

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

# Regularisation Parameters
alpha = 0.8
beta = numpy.sqrt(2)* alpha

method = '1'

if method == '0':

    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())
        
    op31 = Identity(ig, ag)
    op32 = ZeroOperator(op22.domain_geometry(), ag)
    
    operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
        
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm() 
    f3 = L1Norm(b=noisy_data)    
    f = BlockFunction(f1, f2, f3)         
    g = ZeroFunction()
        
else:
    
    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())    
    
    operator = BlockOperator(op11, -1*op12, op21, op22, shape=(2,2) )      
    
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm()     
    
    f = BlockFunction(f1, f2)         
    g = BlockFunction(L1Norm(b=noisy_data), ZeroFunction())
     
## Compute operator Norm
normK = operator.norm()
#
# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 50
pdhg.run(2000, verbose = False)

#%%
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(pdhg.get_output()[0].as_array())
plt.title('TGV Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), pdhg.get_output()[0].as_array()[int(N/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False

if cvx_not_installable:    
    
    u = Variable(ig.shape)
    w1 = Variable((N, N))
    w2 = Variable((N, N))
    
    # create TGV regulariser
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u) - vec(w1), \
                                           DY.matrix() * vec(u) - vec(w2)]), 2, axis = 0)) + \
                  beta * sum(norm(vstack([ DX.matrix().transpose() * vec(w1), DY.matrix().transpose() * vec(w2), \
                                      0.5 * ( DX.matrix().transpose() * vec(w2) + DY.matrix().transpose() * vec(w1) ), \
                                      0.5 * ( DX.matrix().transpose() * vec(w2) + DY.matrix().transpose() * vec(w1) ) ]), 2, axis = 0  ) )  
    
    constraints = []
    fidelity = pnorm(u - noisy_data.as_array(),1)    
    solver = MOSEK

        # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS  
        
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)
    
    diff_cvx = numpy.abs( pdhg.get_output()[0].as_array() - u.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output()[0].as_array())
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
    
    plt.plot(np.linspace(0,N,N), pdhg.get_output()[0].as_array()[int(N/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))





