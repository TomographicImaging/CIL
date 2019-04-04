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

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFun, L2NormSquared, \
                      MixedL21Norm, BlockFunction, ScaledFunction

from ccpi.astra.ops import AstraProjectorSimple
from skimage.util import random_noise


#%%###############################################################################
# Create phantom for TV tomography

#import os
#import tomophantom
#from tomophantom import TomoP2D
#from tomophantom.supp.qualitymetrics import QualityTools

#model = 1 # select a model number from the library
#N = 150 # set dimension of the phantom
## one can specify an exact path to the parameters file
## path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
#path = os.path.dirname(tomophantom.__file__)
#path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
##This will generate a N_size x N_size phantom (2D)
#phantom_2D = TomoP2D.Model(model, N, path_library2D)
#ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
#data = ImageData(phantom_2D, geometry=ig)

N = 150
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)


detectors = 150
angles = np.linspace(0,np.pi,100)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(data)

plt.imshow(sin.as_array())
plt.title('Sinogram')
plt.colorbar()
plt.show()

# Add Gaussian noise to the sinogram data
np.random.seed(10)
n1 = np.random.random(sin.shape)

noisy_data = sin + ImageData(5*n1)

plt.imshow(noisy_data.as_array())
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
diag_precon = False 

if diag_precon:
    
    def tau_sigma_precond(operator):
        
        tau = 1/operator.sum_abs_row()
        sigma = 1/ operator.sum_abs_col()
    
        return tau, sigma

    tau, sigma = tau_sigma_precond(operator)
    
else:    
    sigma = 10
    tau = 1/(sigma*normK**2)

#pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
#pdhg.max_iteration = 5000
#pdhg.update_objective_interval = 250
#
#pdhg.run(5000)
    
opt = {'niter':2000}
#
res = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 

aaa = res[0].as_array()
#    
plt.imshow(aaa)
plt.colorbar()
plt.show()    

#%%
#sol = pdhg.get_output().as_array()
#fig = plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(noisy_data.as_array())
##plt.colorbar()
#plt.subplot(1,2,2)
#plt.imshow(sol)
##plt.colorbar()
#plt.show()
#
#
##%%
#plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
#plt.legend()
#plt.show()


