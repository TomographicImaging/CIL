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

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorMC

import os
import tomophantom
from tomophantom import TomoP2D

# Create phantom for TV 2D dynamic tomography 

model = 102  # note that the selected model is temporal (2D + time)
N = 128 # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
#This will generate a N_size x N_size x Time frames phantom (2D + time)
phantom_2Dt = TomoP2D.ModelTemporal(model, N, path_library2D)

plt.close('all')
plt.figure(1)
plt.rcParams.update({'font.size': 21})
plt.title('{}''{}'.format('2D+t phantom using model no.',model))
for sl in range(0,np.shape(phantom_2Dt)[0]):
    im = phantom_2Dt[sl,:,:]
    plt.imshow(im, vmin=0, vmax=1)
#    plt.pause(.1)
#    plt.draw

    
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, channels = np.shape(phantom_2Dt)[0])
data = ImageData(phantom_2Dt, geometry=ig)
ag = ig
                      
# Create Noisy data. Add Gaussian noise
np.random.seed(10)
noisy_data = ImageData( data.as_array() + np.random.normal(0, 0.25, size=ig.shape) )

tindex = [3, 6, 10]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
plt.subplot(1,3,1)
plt.imshow(noisy_data.as_array()[tindex[0],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[0]))
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array()[tindex[1],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[1]))
plt.subplot(1,3,3)
plt.imshow(noisy_data.as_array()[tindex[2],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[2]))

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

plt.show()   

#%%
# Regularisation Parameter
alpha = 0.3

# Create operators
#op1 = Gradient(ig)
op1 = Gradient(ig, correlation='Space')
op2 = Gradient(ig, correlation='SpaceChannels')

op3 = Identity(ig, ag)

# Create BlockOperator
operator1 = BlockOperator(op1, op3, shape=(2,1) ) 
operator2 = BlockOperator(op2, op3, shape=(2,1) )

# Create functions
      
f1 = alpha * MixedL21Norm()
f2 = 0.5 * L2NormSquared(b = noisy_data)    
f = BlockFunction(f1, f2)  
                                                                        
g = ZeroFunction()
    
# Compute operator Norm
normK1 = operator1.norm()
normK2 = operator2.norm()

#%%
# Primal & dual stepsizes
sigma1 = 1
tau1 = 1/(sigma1*normK1**2)

sigma2 = 1
tau2 = 1/(sigma2*normK2**2)

# Setup and run the PDHG algorithm
pdhg1 = PDHG(f=f,g=g,operator=operator1, tau=tau1, sigma=sigma1)
pdhg1.max_iteration = 2000
pdhg1.update_objective_interval = 200
pdhg1.run(2000)

# Setup and run the PDHG algorithm
pdhg2 = PDHG(f=f,g=g,operator=operator2, tau=tau2, sigma=sigma2)
pdhg2.max_iteration = 2000
pdhg2.update_objective_interval = 200
pdhg2.run(2000)


#%%

tindex = [3, 6, 10]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

plt.subplot(3,3,1)
plt.imshow(phantom_2Dt[tindex[0],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[0]))

plt.subplot(3,3,2)
plt.imshow(phantom_2Dt[tindex[1],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[1]))

plt.subplot(3,3,3)
plt.imshow(phantom_2Dt[tindex[2],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[2]))

plt.subplot(3,3,4)
plt.imshow(pdhg1.get_output().as_array()[tindex[0],:,:])
plt.axis('off')
plt.subplot(3,3,5)
plt.imshow(pdhg1.get_output().as_array()[tindex[1],:,:])
plt.axis('off')
plt.subplot(3,3,6)
plt.imshow(pdhg1.get_output().as_array()[tindex[2],:,:])
plt.axis('off')


plt.subplot(3,3,7)
plt.imshow(pdhg2.get_output().as_array()[tindex[0],:,:])
plt.axis('off')
plt.subplot(3,3,8)
plt.imshow(pdhg2.get_output().as_array()[tindex[1],:,:])
plt.axis('off')
plt.subplot(3,3,9)
plt.imshow(pdhg2.get_output().as_array()[tindex[2],:,:])
plt.axis('off')

#%%
im = plt.imshow(pdhg1.get_output().as_array()[tindex[0],:,:])


fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)


plt.show()

