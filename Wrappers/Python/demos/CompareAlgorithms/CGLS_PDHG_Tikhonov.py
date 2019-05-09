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


from ccpi.framework import ImageData, ImageGeometry, \
                            AcquisitionGeometry, BlockDataContainer, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, CGLS
from ccpi.optimisation.operators import BlockOperator, Gradient

from ccpi.optimisation.functions import ZeroFunction, BlockFunction, L2NormSquared       

# Create Ground truth phantom and Sinogram
N = 128
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0, np.pi, N, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)
device = input('Available device: GPU==1 / CPU==0 ')
ag = AcquisitionGeometry('parallel','2D', angles, detectors)
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
sin = Aop.direct(data)

noisy_data = AcquisitionData( sin.as_array() + np.random.normal(0,3,ig.shape))

# Setup and run the CGLS algorithm  
alpha = 50
Grad = Gradient(ig)

# Form Tikhonov as a Block CGLS structure
op_CGLS = BlockOperator( Aop, alpha * Grad, shape=(2,1))
block_data = BlockDataContainer(noisy_data, Grad.range_geometry().allocate())

x_init = ig.allocate()      

cgls = CGLS(x_init=x_init, operator=op_CGLS, data=block_data)
cgls.max_iteration = 1000
cgls.update_objective_interval = 200
cgls.run(1000,verbose=False)


#Setup and run the PDHG algorithm 

# Create BlockOperator
op_PDHG = BlockOperator(Grad, Aop, shape=(2,1) ) 

# Create functions     
f1 = 0.5 * alpha**2 * L2NormSquared()
f2 = 0.5 * L2NormSquared(b = noisy_data)    
f = BlockFunction(f1, f2)                                       
g = ZeroFunction()

## Compute operator Norm
normK = op_PDHG.norm()

## Primal & dual stepsizes
sigma = 10
tau = 1/(sigma*normK**2)

pdhg = PDHG(f=f,g=g,operator=op_PDHG, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000, verbose=False)


#%%
# Show results
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)
plt.imshow(cgls.get_output().as_array())
plt.title('CGLS reconstruction')

plt.subplot(2,1,2)
plt.imshow(pdhg.get_output().as_array())
plt.title('PDHG reconstruction')

plt.show()

diff1 = pdhg.get_output() - cgls.get_output()

plt.imshow(diff1.abs().as_array())
plt.title('Diff PDHG vs CGLS')
plt.colorbar()
plt.show()

plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
plt.plot(np.linspace(0,N,N), cgls.get_output().as_array()[int(N/2),:], label = 'CGLS')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()
            








#
#
#
#
#
#
#
#
