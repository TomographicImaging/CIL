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

             A: Projection operator
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
       
import tomophantom
from tomophantom import TomoP2D
from ccpi.astra.operators import AstraProjectorSimple 
import os


# Load  Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 128 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)

detectors =  N
angles = np.linspace(0, np.pi, 180, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

device = input('Available device: GPU==1 / CPU==0 ')

if device =='1':
    dev = 'gpu'
else:
    dev = 'cpu'

Aop = AstraProjectorSimple(ig, ag, dev)    
sin = Aop.direct(data)

np.random.seed(10)
noisy_data = AcquisitionData( sin.as_array() + np.random.normal(0,1,ag.shape))
#noisy_data = AcquisitionData( sin.as_array() )

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


# Setup and run the simple CGLS algorithm  
x_init = ig.allocate()  

cgls1 = CGLS(x_init = x_init, operator = Aop, data = noisy_data)
cgls1.max_iteration = 20
cgls1.update_objective_interval = 5
cgls1.run(20, verbose = True)


# Setup and run the regularised CGLS algorithm  (Tikhonov with Identity)

x_init = ig.allocate()  
alpha1 = 50
op1 = Identity(ig)

block_op1 = BlockOperator( Aop, alpha1 * op1, shape=(2,1))
block_data1 = BlockDataContainer(noisy_data, op1.range_geometry().allocate())
   
cgls2 = CGLS(x_init = x_init, operator = block_op1, data = block_data1)
cgls2.max_iteration = 200
cgls2.update_objective_interval = 10
cgls2.run(200, verbose=True)

# Setup and run the regularised CGLS algorithm  (Tikhonov with Gradient)

x_init = ig.allocate() 
alpha2 = 25
op2 = Gradient(ig)

block_op2 = BlockOperator( Aop, alpha2 * op2, shape=(2,1))
block_data2 = BlockDataContainer(noisy_data, op2.range_geometry().allocate())
   
cgls3 = CGLS(x_init=x_init, operator = block_op2, data = block_data2)
cgls3.max_iteration = 200
cgls3.update_objective_interval = 5
cgls3.run(200, verbose = True)

# Show results
plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')

plt.subplot(2,2,2)
plt.imshow(cgls1.get_output().as_array())
plt.title('GGLS')

plt.subplot(2,2,3)
plt.imshow(cgls2.get_output().as_array())
plt.title('Regularised GGLS L = {} * Identity'.format(alpha1))

plt.subplot(2,2,4)
plt.imshow(cgls3.get_output().as_array())
plt.title('Regularised GGLS L =  {} * Gradient'.format(alpha2))

plt.show()



print('Compare CVX vs Regularised CG with L = Gradient')

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
    #q = Variable()
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')

    regulariser = alpha2**2 * sum_squares(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
    
    # create matrix representation for Astra operator

    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)

    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)

    ProjMat = astra.matrix.get(matrix_id)
    
    fidelity = sum_squares( ProjMat * u - noisy_data.as_array().ravel()) 
        
    solver = MOSEK
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)    


plt.figure(figsize=(10,20))

plt.subplot(1,3,1)
plt.imshow(np.reshape(u.value, (N, N)))    
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(cgls3.get_output().as_array())    
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.abs(cgls3.get_output().as_array() - np.reshape(u.value, (N, N)) ))    
plt.colorbar()

plt.show()

print('Primal Objective (CVX) {} '.format(obj.value))
print('Primal Objective (CGLS) {} '.format(cgls3.objective[-1]))

