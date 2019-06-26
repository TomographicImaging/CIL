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
Compare solutions of FISTA & PDHG & CGLS  
                    & Astra Built-in algorithms for Least Squares


Problem:     min_x || A x - g ||_{2}^{2}

             A: Projection operator
             g: Sinogram

"""


from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, CGLS, FISTA

from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, FunctionOperatorComposition
from ccpi.astra.ops import AstraProjectorSimple
import astra   

import tomophantom
from tomophantom import TomoP2D
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
sinogram = Aop.direct(data)



###############################################################################
# Setup and run Astra CGLS algorithm
vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# Create a sinogram id
sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram.as_array())

# Create a data id
rec_id = astra.data2d.create('-vol', vol_geom)

cgls_astra = astra.astra_dict('CGLS')
cgls_astra['ReconstructionDataId'] = rec_id
cgls_astra['ProjectionDataId'] = sinogram_id
cgls_astra['ProjectorId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cgls_astra)

astra.algorithm.run(alg_id, 500)
recon_cgls_astra = ImageData(astra.data2d.get(rec_id))


#%%

# Setup and run the simple CGLS algorithm  
x_init = ig.allocate()  

cgls = CGLS(x_init = x_init, operator = Aop, data = sinogram)
cgls.max_iteration = 500
cgls.update_objective_interval = 100
cgls.run(500, verbose = True)

#%%

###############################################################################
# Setup and run the PDHG algorithm 
operator = Aop
f = L2NormSquared(b = sinogram)
g = ZeroFunction()

## Compute operator Norm
normK = operator.norm()

## Primal & dual stepsizes
sigma = 0.02
tau = 1/(sigma*normK**2)


pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 100
pdhg.run(1000, verbose=True)

#%%
###############################################################################
# Setup and run the FISTA algorithm 
fidelity = FunctionOperatorComposition(L2NormSquared(b=sinogram), Aop)
regularizer = ZeroFunction()

fista = FISTA(x_init=x_init , f=fidelity, g=regularizer)
fista.max_iteration = 500
fista.update_objective_interval = 100
fista.run(500, verbose = True)

#%% Show results

plt.figure(figsize=(10,10))
plt.suptitle('Reconstructions ', fontsize=16)

plt.subplot(2,2,1)
plt.imshow(cgls.get_output().as_array())
plt.colorbar()
plt.title('CGLS reconstruction')

plt.subplot(2,2,2)
plt.imshow(fista.get_output().as_array())
plt.colorbar()
plt.title('FISTA reconstruction')

plt.subplot(2,2,3)
plt.imshow(pdhg.get_output().as_array())
plt.colorbar()
plt.title('PDHG reconstruction')

plt.subplot(2,2,4)
plt.imshow(recon_cgls_astra.as_array())
plt.colorbar()
plt.title('CGLS astra')

diff1 = pdhg.get_output() - recon_cgls_astra
diff2 = fista.get_output() - recon_cgls_astra
diff3 = cgls.get_output() - recon_cgls_astra

plt.figure(figsize=(15,15))

plt.subplot(3,1,1)
plt.imshow(diff1.abs().as_array())
plt.title('Diff PDHG vs CGLS astra')
plt.colorbar()

plt.subplot(3,1,2)
plt.imshow(diff2.abs().as_array())
plt.title('Diff FISTA vs CGLS astra')
plt.colorbar()

plt.subplot(3,1,3)
plt.imshow(diff3.abs().as_array())
plt.title('Diff CLGS vs CGLS astra')
plt.colorbar()


cgls_astra_obj = fidelity(ImageData(recon_cgls_astra))

print('Primal Objective (FISTA) {} '.format(fista.objective[-1]))
print('Primal Objective (CGLS) {} '.format(cgls.objective[-1]))
print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))
print('Primal Objective (CGLS_astra) {} '.format(cgls_astra_obj))



