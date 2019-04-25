#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Identity, \
                        Gradient, SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from skimage.util import random_noise

from timeit import default_timer as timer
from ccpi.astra.ops import AstraProjectorSimple

#def dt(steps):
#    return steps[-1] - steps[-2]

# Create phantom for TGV Gaussian denoising

N = 100

data = np.zeros((N,N))

x1 = np.linspace(0, int(N/2), N)
x2 = np.linspace(int(N/2), 0., N)
xv, yv = np.meshgrid(x1, x2)

xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T

data = xv
data = ImageData(data/data.max())

plt.imshow(data.as_array())
plt.show()

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = N
angles = np.linspace(0,np.pi,N)

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

#%%

alpha, beta = 20, 50


# Create operators
op11 = Gradient(ig)
op12 = Identity(op11.range_geometry())

op22 = SymmetrizedGradient(op11.domain_geometry())    
op21 = ZeroOperator(ig, op22.range_geometry())
    
op31 = Aop
op32 = ZeroOperator(op22.domain_geometry(), ag)

operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 


f1 = alpha * MixedL21Norm()
f2 = beta * MixedL21Norm() 
f3 = 0.5 * L2NormSquared(b = noisy_data)    
f = BlockFunction(f1, f2, f3)         
g = ZeroFunction()
             
## Compute operator Norm
normK = operator.norm()
#
## Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
##
opt = {'niter':5000}
opt1 = {'niter':5000, 'memopt': True}
#
t1 = timer()
res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
t2 = timer()
#
plt.imshow(res[0].as_array())
plt.show()


#t3 = timer()
#res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) 
#t4 = timer()
#
#plt.figure(figsize=(15,15))
#plt.subplot(3,1,1)
#plt.imshow(res[0].as_array())
#plt.title('no memopt')
#plt.colorbar()
#plt.subplot(3,1,2)
#plt.imshow(res1[0].as_array())
#plt.title('memopt')
#plt.colorbar()
#plt.subplot(3,1,3)
#plt.imshow((res1[0] - res[0]).abs().as_array())
#plt.title('diff')
#plt.colorbar()
#plt.show()
#
#print("NoMemopt/Memopt is {}/{}".format(t2-t1, t4-t3))
    

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
    
    
   
