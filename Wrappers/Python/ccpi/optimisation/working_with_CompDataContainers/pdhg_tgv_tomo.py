#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.ops import PowerMethodNonsquare

import astra

import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from cvxpy import *

from Algorithms import PDHG

from Operators import CompositeOperator, Identity, Gradient, CompositeDataContainer, AstraProjectorSimple
from Functions import ZeroFun, L2NormSq, mixed_L12Norm, FunctionOperatorComposition, BlockFunction

from skimage.util import random_noise



#%%###############################################################################
# Create phantom for TV tomography

N = 75
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = 100
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
noisy_data = ImageData(sin.as_array() + 5*n1)

plt.imshow(noisy_data.as_array())
plt.title('Noisy Sinogram')
plt.colorbar()
plt.show()

#%%
alpha = 50
beta = 100

ig = (ig.voxel_num_x,ig.voxel_num_y)

Grad = Gradient(ig)
SymGrad = SymmetrizedGradient( ((2,)+ig), ((3,)+ig))

Id1 = Identity( Grad.range_dim(), Grad.range_dim() )
ZeroOp1 = ZeroOp( ig, SymGrad.range_dim())

ZeroOp2 = ZeroOp( SymGrad.domain_dim(), Aop.range_dim())


operator = CompositeOperator( (3,2), Grad, -1*Id1,\
                                         ZeroOp1, SymGrad,\
                                         Aop, ZeroOp2)

f = BlockFunction(operator, mixed_L12Norm(alpha), \
                            mixed_L12Norm(beta, sym_grad=True),\
                            L2NormSq(0.5, b=noisy_data) )
g = ZeroFun()
            
normK = operator.norm()      
## Primal & dual stepsizes
sigma = 10
tau = 1.0/(sigma*normK**2)
#
opt = {'niter':3000}
result, total_time, its = PDHG(f, g, operator, \
                                  tau = tau, sigma = sigma, opt = opt)


#%%
sol = result.get_item(0).as_array()
#
plt.imshow(sol)
plt.colorbar()
plt.title('Reconstruction')
plt.show()

#%%
#
#plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
#plt.legend()
###############################################################################
##%%  Compare with CVX
#
#try_cvx = input("Do you want CVX comparison (0/1)")
#
#if try_cvx=='0':
#
#    from cvxpy import *
#    import sys
#    sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
#    from cvx_functions import tgv
#
#    ###Construct problem
#    u = Variable((N, N))
#    w1 = Variable((N, N))
#    w2 = Variable((N, N))
#
#    constraints = []
#    fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
#    solver = MOSEK
#    regulariser = tgv(u,w1,w2,alpha,beta)
#
#    obj =  Minimize( regulariser +  fidelity )
#    prob = Problem( obj, constraints)
#
#    resCVX = prob.solve(verbose = True, solver = solver)
#    print()
#    print('Objective value is {} '.format(obj.value))
#
###%%
#    # Show results
#    plt.imshow(u.value)
#    plt.title('CVX-denoising')
#    plt.colorbar()
#    plt.show()
##
#    plt.imshow(result.get_item(0).as_array())
#    plt.title('PDHG-denoising')
#    plt.colorbar()
#    plt.show()
##
#    dif = np.abs( result.get_item(0).as_array() - u.value)
#    plt.imshow(dif)
#    plt.title('Difference')
#    plt.colorbar()
#    plt.show()
##
#    plt.plot(np.linspace(0,N,N), result.get_item(0).as_array()[50,:], label = 'CVX')
#    plt.plot(np.linspace(0,N,N), u[50,:].value, label = 'PDHG')
#    plt.plot(np.linspace(0,N,N), phantom[50,:], label = 'phantom')
##    plt.plot(np.linspace(0,N,N), noisy_data.as_array()[50,:], label = 'noisy')
#    plt.legend()
#    plt.show()
#    
#else:
#    print('No CVX solution available')    
#
