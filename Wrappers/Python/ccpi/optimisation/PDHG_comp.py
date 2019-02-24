#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:41:25 2019

@author: evangelos
"""
import time

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.funcs import Norm2sq, Norm1, IndicatorBox
from ccpi.optimisation.ops import PowerMethodNonsquare

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from my_changes import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import *
from operators import *
from regularisers import *
from functions import *
from GradientOperator import FiniteDiff

#from cvx_functions import *
#from cvxpy import *


#%%
# Create a phantom 
N = 200
#
phantom = np.zeros((N,N))
#
x1 = np.linspace(0, int(N/2), N)
x2 = np.linspace(int(N/2), 0., N)
xv, yv = np.meshgrid(x1, x2)
#
xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T
#
phantom = xv
## Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
phantom = phantom/phantom.max()

plt.imshow(phantom)
plt.show()

#%%
# Add noise
noisy_data = ImageData(random_noise(phantom,'gaussian', mean = 0, var = 0.05))


alpha = 1


# Define geometries( Not the correct name )
#ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
#ag = ig
ig = (N, N)
ag = ig

Grad2D = Gradient(ig)
Id = MyTomoIdentity(ig, ag)

operator = [ [Grad2D], [Id] ] 
#operator = CompositeOperator((2,1), gradient(ig), MyTomoIdentity(ig, ag) ) 


f = [ L1NormOld(Grad2D, alpha ), \
      Norm2sq_new(MyTomoIdentity(ig,ag), noisy_data, c = 0.5, memopt = False) ]
g = ZeroFun()


# Compute operator Norm
normK = np.sqrt( Grad2D.norm()**2 + Id.norm()**2)

##%%
## Primal & dual stepsizes
sigma = 1.0/normK
tau = 1.0/normK
#
#
opt = {'niter':1000}
res, total_time, its = PDHG_testGeneric(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)
#
plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.title('Noisy')
plt.show()
#
plt.imshow(res.as_array())
plt.colorbar()
plt.title('Reconstruction')
plt.show()
#
plt.plot(np.linspace(0,N,N), phantom[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'Recon')
plt.legend()
###############################################################################
##%%  Compare with CVX
#
##Construct problem
#u = Variable((N, N))
#
#constraints = []
#fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
#solver = MOSEK
#regulariser = alpha * tv(u)
#
#obj =  Minimize( regulariser +  fidelity )
#prob = Problem( obj, constraints)
#
#resCVX = prob.solve(verbose = True, solver = solver)
#print()
#print('Objective value is {} '.format(obj.value))
#
##%%
## Show results
#plt.imshow(u.value)
#plt.title('CVX-denoising')
#plt.colorbar()
#plt.show()
#
#plt.imshow(res.as_array())
#plt.title('PDHG-denoising')
#plt.colorbar()
#plt.show()
#
#dif = np.abs( res.as_array() - u.value)
#plt.imshow(dif)
#plt.title('Difference')
#plt.colorbar()
#plt.show()
#
#plt.plot(np.linspace(0,N,N), res.as_array()[50,:], label = 'CVX')
#plt.plot(np.linspace(0,N,N), u[50,:].value, label = 'PDHG')
#plt.plot(np.linspace(0,N,N), phantom[50,:], label = 'phantom')
#plt.legend()
#plt.show()







