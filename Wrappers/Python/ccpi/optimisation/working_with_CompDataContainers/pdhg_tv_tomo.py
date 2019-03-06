#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:31:47 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.ops import PowerMethodNonsquare

import astra

import numpy as np
from numpy import inf
import numpy
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG
from operators import CompositeOperator, Identity, CompositeDataContainer, AstraProjectorSimple
from GradientOperator import Gradient
#from functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq ,CompositeFunction
from test_functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq, CompositeFunction, FunctionComposition_new

from Sparse_GradMat import GradOper


#%%###############################################################################
# Create phantom for TV

N = 75
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

x = ImageData(x)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)

detectors = 100
angles = np.linspace(0,np.pi,100)

ag = AcquisitionGeometry('parallel','2D',angles, detectors)
Aop = AstraProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(x)

np.random.seed(1)
noisy_sin = ImageData(sin.as_array() + 5 * np.random.rand(100,100))


# Create operators
op1 = Gradient((ig.voxel_num_x,ig.voxel_num_y))
op2 = Aop

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 

alpha = 50
#f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
#                                      L2NormSq(0.5, b = noisy_sin) )
#g = ZeroFun()

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
opt = {'niter':5000}
###
#### Run algorithm
#res, total_time, objective = PDHG(f, g, operator, \
#                                  tau = tau, sigma = sigma, opt = opt)
##%%
####Show results
#solution = res.get_item(0).as_array()
###solution = res.as_array()
###
#plt.imshow(solution)
#plt.colorbar()
#plt.show()
##
####
#plt.imshow(noisy_data.as_array())
#plt.colorbar()
#plt.show()
###
#plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), solution[int(N/2),:], label = 'Recon')
#plt.legend()
#plt.show()
#%% CVX tomo solution

#import sys
#sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
#from cvx_functions import TV_cvx
#
## Create volume, geometry,
#vol_geom = astra.create_vol_geom(N, N)
#proj_geom = astra.create_proj_geom('parallel', 1.0, 100, np.linspace(0,np.pi,100,False))
#
## create projector
#proj_id = astra.create_projector('strip', proj_geom, vol_geom)
#
## create sinogram
#sin_id, sin = astra.create_sino(x.as_array(), proj_id, 'False') 
#
## create projection matrix
#matrix_id = astra.projector.matrix(proj_id)
#
## Get the projection matrix as a Scipy sparse matrix.
#ProjMat = astra.matrix.get(matrix_id)
#
#
#alpha_tvCT = 50
#
## Define the problem
#u_tvCT = Variable( N*N, 1)
#z = noisy_sin.as_array().ravel()
#obj_tvCT =  Minimize( 0.5 * sum_squares(ProjMat * u_tvCT - z) + \
#                    alpha_tvCT * tv(reshape(u_tvCT, (N,N))) )
#
#prob_tvCT = Problem(obj_tvCT)
#
## Choose solver, SCS is fast but less accurate than MOSEK
##res_tvCT = prob_tvCT.solve(verbose = True,solver = SCS,eps=1e-12)
#res_tvCT = prob_tvCT.solve(verbose = True, solver = MOSEK)
#
#print()
#print('Objective value is {} '.format(obj_tvCT.value))

#diff_pdhg_cvx = np.abs(np.reshape(u_tvCT.value, (N,N)) - solution)
#plt.imshow(diff_pdhg_cvx)
#plt.colorbar()
#plt.show()
#
#plt.plot(np.linspace(0,N,N), np.reshape(u_tvCT.value, (N,N))[int(N/2),:], label = 'CVX')
#plt.plot(np.linspace(0,N,N), solution[int(N/2),:], label = 'PDHG')
#plt.legend()
#plt.show()

#%% FISTA tomo solution

from ccpi.optimisation.algs import CGLS, FISTA

g1 = FunctionComposition_new(op1, mixed_L12Norm(alpha))
f1 = FunctionComposition_new(op2, L2NormSq(0.5, b=noisy_sin))

opt = {'tol': 1e-4, 'iter': 100}

x_init = ImageData(numpy.zeros((N,N)))

x_fista1, it1, timing1, criter1 = FISTA(x_init, f1, g1, opt)

