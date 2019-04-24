#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, \
                        Gradient, SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction

from skimage.util import random_noise

# Create phantom for TGV SaltPepper denoising

N = 300

data = np.zeros((N,N))

x1 = np.linspace(0, int(N/2), N)
x2 = np.linspace(int(N/2), 0., N)
xv, yv = np.meshgrid(x1, x2)

xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T

data = xv
data = ImageData(data/data.max())

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data.as_array(), mode = 's&p', salt_vs_pepper = 0.9, amount=0.2)
noisy_data = ImageData(n1)

# Regularisation Parameters
alpha = 0.8
beta = numpy.sqrt(2)* alpha

method = '0'

if method == '0':

    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())
        
    op31 = Identity(ig, ag)
    op32 = ZeroOperator(op22.domain_geometry(), ag)
    
    operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
        
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm() 
    f3 = L1Norm(b=noisy_data)    
    f = BlockFunction(f1, f2, f3)         
    g = ZeroFunction()
        
else:
    
    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())    
    
    operator = BlockOperator(op11, -1*op12, op21, op22, shape=(2,2) )      
    
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm()     
    
    f = BlockFunction(f1, f2)         
    g = BlockFunction(0.5 * L1Norm(b=noisy_data), ZeroFunction())
     
## Compute operator Norm
normK = operator.norm()
#
# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 50
pdhg.run(2000)

#%%
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(pdhg.get_output()[0].as_array())
plt.title('TGV Reconstruction')
plt.colorbar()
plt.show()
## 
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), pdhg.get_output()[0].as_array()[int(N/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()










#%% Check with CVX solution

#from ccpi.optimisation.operators import SparseFiniteDiff
#
#try:
#    from cvxpy import *
#    cvx_not_installable = True
#except ImportError:
#    cvx_not_installable = False
#
#if cvx_not_installable:    
#    
#    u = Variable(ig.shape)
#    w1 = Variable((N, N))
#    w2 = Variable((N, N))
#    
#    # create TGV regulariser
#    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
#    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
#    
#    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u) - vec(w1), \
#                                           DY.matrix() * vec(u) - vec(w2)]), 2, axis = 0)) + \
#                  beta * sum(norm(vstack([ DX.matrix().transpose() * vec(w1), DY.matrix().transpose() * vec(w2), \
#                                      0.5 * ( DX.matrix().transpose() * vec(w2) + DY.matrix().transpose() * vec(w1) ), \
#                                      0.5 * ( DX.matrix().transpose() * vec(w2) + DY.matrix().transpose() * vec(w1) ) ]), 2, axis = 0  ) )  
#    
#    constraints = []
#    fidelity = 0.5 * sum_squares(u - noisy_data.as_array())    
#    solver = MOSEK
#
#    obj =  Minimize( regulariser +  fidelity)
#    prob = Problem(obj)
#    result = prob.solve(verbose = True, solver = solver)
#    
#    diff_cvx = numpy.abs( res[0].as_array() - u.value )   
#    
#    # Show result
#    plt.figure(figsize=(15,15))
#    plt.subplot(3,1,1)
#    plt.imshow(res[0].as_array())
#    plt.title('PDHG solution')
#    plt.colorbar()
#    
#    plt.subplot(3,1,2)
#    plt.imshow(u.value)
#    plt.title('CVX solution')
#    plt.colorbar()
#    
#    plt.subplot(3,1,3)
#    plt.imshow(diff_cvx)
#    plt.title('Difference')
#    plt.colorbar()
#    plt.show()
#    
#    plt.plot(np.linspace(0,N,N), res[0].as_array()[int(N/2),:], label = 'PDHG')
#    plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
#    plt.legend()   
#    
#    print('Primal Objective (CVX) {} '.format(obj.value))
#    print('Primal Objective (PDHG) {} '.format(primal[-1])) 
#    print('Min/Max of absolute difference {}/{}'.format(diff_cvx.min(), diff_cvx.max()))
#    
    
   
