# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer, AcquisitionGeometry, AcquisitionData

import numpy as np                           
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFun, L2NormSquared, \
                      MixedL21Norm, BlockFunction, ScaledFunction

from ccpi.astra.ops import AstraProjectorSimple, AstraProjectorMC
from skimage.util import random_noise


#%%###############################################################################
# Create phantom for TV tomography

import numpy as np
import matplotlib.pyplot as plt
import os
import tomophantom
from tomophantom import TomoP2D

model = 102  # note that the selected model is temporal (2D + time)
N = 150 # set dimension of the phantom
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
    plt.pause(.1)
    plt.draw

#N = 150
#x = np.zeros((N,N))
#x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
#x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

#%%
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, channels = np.shape(phantom_2Dt)[0])
data = ImageData(phantom_2Dt, geometry=ig)



detectors = 150
angles = np.linspace(0,np.pi,100)

ag = AcquisitionGeometry('parallel','2D',angles, detectors, channels = np.shape(phantom_2Dt)[0])
Aop = AstraProjectorMC(ig, ag, 'gpu')
sin = Aop.direct(data)

plt.imshow(sin.as_array()[10])
plt.title('Sinogram')
plt.colorbar()
plt.show()

# Add Gaussian noise to the sinogram data
np.random.seed(10)
n1 = np.random.random(sin.shape)

noisy_data = sin + ImageData(5*n1)

plt.imshow(noisy_data.as_array()[10])
plt.title('Noisy Sinogram')
plt.colorbar()
plt.show()


#%% Works only with Composite Operator Structure of PDHG

#ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Form Composite Operator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

alpha = 50
f = BlockFunction( alpha * MixedL21Norm(), \
                   0.5 * L2NormSquared(b = noisy_data) )
g = ZeroFun()

# Compute operator Norm
normK = operator.norm()

## Primal & dual stepsizes

sigma = 1
tau = 1/(sigma*normK**2)

#sigma = 1/normK
#tau = 1/normK

opt = {'niter':2000}

res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
 
plt.figure(figsize=(5,5))
plt.imshow(res.as_array())
plt.colorbar()
plt.show()

#sigma = 10
#tau = 1/(sigma*normK**2)
#
#pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
#pdhg.max_iteration = 5000
#pdhg.update_objective_interval = 20
#
#pdhg.run(5000)
#
##%%
#sol = pdhg.get_output().as_array()
#fig = plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(noisy_data.as_array())
##plt.colorbar()
#plt.subplot(1,2,2)
#plt.imshow(sol)
##plt.colorbar()
#plt.show()


#%%
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
plt.legend()
plt.show()


