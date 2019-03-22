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

from ccpi.optimisation.operators import BlockOperator, BlockOperatorOLD, Identity, Gradient
from ccpi.optimisation.functions import ZeroFun, L2NormSquared, \
                      MixedL21Norm, BlockFunction, ScaledFunction

from ccpi.astra.ops import AstraProjectorSimple
from skimage.util import random_noise

import astra

#%%###############################################################################
# Create phantom for TV tomography

N = 150
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

noisy_data = sin + ImageData(5*n1)

plt.imshow(noisy_data.as_array())
plt.title('Noisy Sinogram')
plt.colorbar()
plt.show()

#%% Works only with Composite Operator Structure of PDHG

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Form Composite Operator
operator = BlockOperatorOLD(op1, op2, shape=(2,1) ) 

alpha = 50
f = BlockFunction( alpha * MixedL21Norm(), \
                   0.5 * L2NormSquared(b = noisy_data) )
g = ZeroFun()

# Compute operator Norm
normK = operator.norm()
#normK = np.sqrt(op1.norm()**2 + op2.norm()**2)
#
#%%
## Primal & dual stepsizes
#
sigma = 10
tau = 1/(sigma*normK**2)

#c = 0.001
#sigma = 1/(c*normK)
#tau = c/normK

#### Create volume, geometry,
#vol_geom = astra.create_vol_geom(N, N)
#proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)
#proj_id = astra.create_projector('line', proj_geom, vol_geom)
#matrix_id = astra.projector.matrix(proj_id)
#ProjMat = astra.matrix.get(matrix_id)
#
#grad = GradOper(data.shape, [1]*len(data.shape), direction = 'for', order = '1', bndrs = 'Neumann')
#
#t1 = np.abs(grad[0]).sum(axis=0) + np.abs(grad[1]).sum(axis=0) + np.abs(ProjMat).sum(axis=0)
##t1 = np.array(t1)**(1.5)
##tau = 
##
##
#s1, s2, s3 = np.abs(grad[0]).sum(axis=1), np.abs(grad[1]).sum(axis=1), np.abs(ProjMat).sum(axis=1)
###
#tau  =  np.array(1/(np.reshape(t1.T, (N, N)))**(Fraction('1.5')))
#tau = CompositeDataContainer(DataContainer(tau))
###
#z = np.zeros((2, N, N))
#z[0] = np.reshape(1/s1, (N, N))
#z[1] = np.reshape(1/s2, (N, N))
#z[z==inf]=1
#z = np.array(z)
###
#z1 = np.reshape(1/(s3**Fraction('0.5')), (detectors, detectors))
#z1[z1==inf]=1
#z1 = np.array(z1)
###
#sigma = CompositeDataContainer(DataContainer(z), DataContainer(z1), shape=(2,1))



#%%
##%%
### Number of iterations
opt = {'niter':5000}
###
#### Run algorithm
result, total_time, objective = PDHG(f, g, operator, \
                                  tau = tau, sigma = sigma, opt = opt)
#%%
####Show results
sol = result.as_array()
###solution = res.as_array()
###
plt.imshow(sol)
plt.colorbar()
plt.show()

plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
plt.legend()
plt.show()



#%% CVX TV - tomo solution

#try_cvx = input("Do you want CVX comparison (0/1)")
try_cvx = '1'
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
