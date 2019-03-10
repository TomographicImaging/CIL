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

#%% Works only with Composite Operator Structure of PDHG

# Create operators
op1 = Gradient((ig.voxel_num_x,ig.voxel_num_y))
op2 = Aop

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 

alpha = 50
f = BlockFunction(operator, mixed_L12Norm(alpha), \
                                     L2NormSq(0.5, b = noisy_data) )
g = ZeroFun()

# Compute operator Norm
normK = operator.norm()
#
#%%
## Primal & dual stepsizes
#
sigma = 10
tau = 1/(sigma*normK**2)

##%%
### Number of iterations
opt = {'niter':1000}
###
#### Run algorithm
res, total_time, objective = PDHG(f, g, operator, \
                                  tau = tau, sigma = sigma, opt = opt)
#%%
####Show results
sol = res.get_item(0).as_array()
###solution = res.as_array()
###
plt.imshow(sol)
plt.colorbar()
plt.show()
##
####
#plt.imshow(noisy_data.as_array())
#plt.colorbar()
#plt.show()
###
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
plt.legend()
plt.show()
#%% CVX TV - tomo solution

try_cvx = input("Do you want CVX comparison (0/1)")

if try_cvx=='0':

    from cvxpy import *
    import sys
    sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
    from cvx_functions import TV_cvx

    # Create volume, geometry,
    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 100, np.linspace(0,np.pi,100,False))

    # create projector
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)

    # create sinogram
    sin_id, sin = astra.create_sino(data.as_array(), proj_id, 'False') 

    # create projection matrix
    matrix_id = astra.projector.matrix(proj_id)

    # Get the projection matrix as a Scipy sparse matrix.
    ProjMat = astra.matrix.get(matrix_id)


    alpha_tvCT = alpha

    # Define the problem
    u_tvCT = Variable( N*N, 1)
    z = noisy_data.as_array().ravel()
    obj_tvCT =  Minimize( 0.5 * sum_squares(ProjMat * u_tvCT - z) + \
                        alpha_tvCT * tv(reshape(u_tvCT, (N,N))) )
    #
    prob_tvCT = Problem(obj_tvCT)
    #
    ## Choose solver, SCS is fast but less accurate than MOSEK
    ##res_tvCT = prob_tvCT.solve(verbose = True,solver = SCS,eps=1e-12)
    res_tvCT = prob_tvCT.solve(verbose = True, solver = MOSEK)
    #
    #print()
    print('Objective value is {} '.format(obj_tvCT.value))

    diff_pdhg_cvx = np.abs(np.reshape(u_tvCT.value, (N,N)) - sol)
    plt.imshow(diff_pdhg_cvx)
    plt.colorbar()
    plt.show()
    #
    plt.plot(np.linspace(0,N,N), np.reshape(u_tvCT.value, (N,N))[int(N/2),:], label = 'CVX')
    plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'PDHG')
    plt.legend()
    plt.show()
