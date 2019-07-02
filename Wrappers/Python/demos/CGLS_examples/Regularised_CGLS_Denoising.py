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
Conjugate Gradient for (Regularized) Least Squares for Tomography


Problem:     min_u alpha * || L x ||^{2}_{2} + || A u - g ||_{2}^{2}

             A: Identity operator
             g: Sinogram
             L: Identity or Gradient Operator

"""


from ccpi.framework import ImageGeometry, ImageData, \
                            AcquisitionGeometry, BlockDataContainer, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
from ccpi.framework import TestData
       
import os, sys

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SHAPES)
ig = data.geometry
ag = ig

noisy_data = ImageData(TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 1))
#noisy_data = ImageData(data.as_array())

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

# Setup and run the regularised CGLS algorithm  (Tikhonov with Gradient)
x_init = ig.allocate() 
alpha = 2
op = Gradient(ig)

block_op = BlockOperator( Identity(ig), alpha * op, shape=(2,1))
block_data = BlockDataContainer(noisy_data, op.range_geometry().allocate())
   
cgls = CGLS(x_init=x_init, operator = block_op, data = block_data)
cgls.max_iteration = 200
cgls.update_objective_interval = 5
cgls.run(200, verbose = True)

# Show results
plt.figure(figsize=(20,10))
plt.subplot(3,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.subplot(3,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy')
plt.subplot(3,1,3)
plt.imshow(cgls.get_output().as_array())
plt.title('Regularised GGLS with Gradient')
plt.show()

#%%

print('Compare CVX vs Regularised CG with L = Gradient')

from ccpi.optimisation.operators import SparseFiniteDiff


try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False


if cvx_not_installable:
    
    ##Construct problem    
    u = Variable(ig.shape)
    #q = Variable()
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')

    regulariser = alpha**2 * sum_squares(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
        
    fidelity = sum_squares( u - noisy_data.as_array()) 
        
    # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS 
        
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = MOSEK)    


plt.figure(figsize=(20,20))
plt.subplot(3,1,1)
plt.imshow(np.reshape(u.value, ig.shape))    
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(cgls.get_output().as_array())    
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(np.abs(cgls.get_output().as_array() - np.reshape(u.value, ig.shape) ))    
plt.colorbar()
plt.show()

print('Primal Objective (CVX) {} '.format(obj.value))
print('Primal Objective (CGLS) {} '.format(cgls.objective[-1]))

