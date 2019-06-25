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
Compare solutions of PDHG & "Block CGLS" algorithms for 


Problem:     min_x alpha * ||\grad x ||^{2}_{2} + || A x - g ||_{2}^{2}


             A: Projection operator
             g: Sinogram

"""


from ccpi.framework import AcquisitionGeometry, BlockDataContainer, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.operators import BlockOperator, Gradient
       
from ccpi.framework import TestData
import os, sys
from ccpi.astra.ops import AstraProjectorSimple 

# Load Data  
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))                 
N = 64
M = 64
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))

ig = data.geometry

detectors = N
angles = np.linspace(0, np.pi, N, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'

Aop = AstraProjectorSimple(ig, ag, dev)    
sin = Aop.direct(data)

noisy_data = AcquisitionData( sin.as_array() + np.random.normal(0,3,ig.shape))

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

# Setup and run the CGLS algorithm  
alpha = 5
Grad = Gradient(ig)
#
## Form Tikhonov as a Block CGLS structure
op_CGLS = BlockOperator( Aop, alpha * Grad, shape=(2,1))
block_data = BlockDataContainer(noisy_data, Grad.range_geometry().allocate())
#
x_init = ig.allocate()      
cgls = CGLS(x_init=x_init, operator=op_CGLS, data=block_data)
cgls.max_iteration = 1000
cgls.update_objective_interval = 200
cgls.run(1000,verbose=True)

# Show results
plt.figure(figsize=(5,5))
plt.imshow(cgls.get_output().as_array())
plt.title('CGLS reconstruction')
plt.colorbar()
plt.show()

#%%
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
    u = Variable(N*M)
    #q = Variable()
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')

    regulariser = alpha * sum_squares(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
    
    # create matrix representation for Astra operator

    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)

    proj_id = astra.create_projector('strip', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)

    ProjMat = astra.matrix.get(matrix_id)
    
    fidelity = sum_squares( ProjMat * u - noisy_data.as_array().ravel()) 
        
    solver = SCS
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj, constraints)
    result = prob.solve(verbose = True, solver = solver)    







            
