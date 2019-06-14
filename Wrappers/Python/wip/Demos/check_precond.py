# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorSimple

# Create phantom for TV 2D tomography 
N = 75
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0, np.pi, N, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(data)

# Create noisy data
np.random.seed(10)
n1 = np.random.random(sin.shape)
noisy_data = sin + ImageData(5*n1)

#%%

# Regularisation Parameter
alpha = 50

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 



# Create functions
      
f1 = alpha * MixedL21Norm()
f2 = L2NormSquared(b=noisy_data)    
f = BlockFunction(f1, f2)  
                                      
g = ZeroFunction()
    
diag_precon =  True

if diag_precon:
    
    def tau_sigma_precond(operator):
        
        tau = 1/operator.sum_abs_row()
        sigma = 1/ operator.sum_abs_col()
               
        return tau, sigma

    tau, sigma = tau_sigma_precond(operator)
             
else:
    # Compute operator Norm
    normK = operator.norm()
    # Primal & dual stepsizes
    sigma = 10
    tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000)

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

    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)

    ProjMat = astra.matrix.get(matrix_id)
    
    fidelity = sum_squares( ProjMat * u - noisy_data.as_array().ravel()) 
    #constraints = [q>=fidelity]
#    constraints = [u>=0]
        
    solver = MOSEK
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)    


#%%
    
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')

plt.subplot(2,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')

plt.subplot(2,2,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('PDHG Reconstruction')

plt.subplot(2,2,4)
plt.imshow(np.reshape(u.value, ig.shape))
plt.title('CVX Reconstruction')

plt.show()

#%%
plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
plt.plot(np.linspace(0,N,N), np.reshape(u.value, ig.shape)[int(N/2),:], label = 'CVX')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()








