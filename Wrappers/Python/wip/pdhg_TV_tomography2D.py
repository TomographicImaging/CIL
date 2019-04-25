# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np                           
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorSimple
from skimage.util import random_noise
from timeit import default_timer as timer


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

N = 75
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

data = ImageData(x)
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

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Form Composite Operator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

alpha = 10
f = BlockFunction( alpha * MixedL21Norm(), \
                   0.5 * L2NormSquared(b = noisy_data) )
g = ZeroFunction()

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
    normK = operator.norm()
    sigma = 1
    tau = 1/(sigma*normK**2)

# Compute operator Norm


# Primal & dual stepsizes

opt = {'niter':200}
opt1 = {'niter':200, 'memopt': True}

t1 = timer()
res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
t2 = timer()


t3 = timer()
res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) 
t4 = timer()
#
#
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(res.as_array())
plt.title('no memopt')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(res1.as_array())
plt.title('memopt')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow((res1 - res).abs().as_array())
plt.title('diff')
plt.colorbar()
plt.show()
# 
plt.plot(np.linspace(0,N,N), res1.as_array()[int(N/2),:], label = 'memopt')
plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'no memopt')
plt.legend()
plt.show()
#
print ("Time: No memopt in {}s, \n Time: Memopt in  {}s ".format(t2-t1, t4 -t3))
diff = (res1 - res).abs().as_array().max()
#
print(" Max of abs difference is {}".format(diff))


#%% Check with CVX solution

#from ccpi.optimisation.operators import SparseFiniteDiff
#import astra
#import numpy
#
#try:
#    from cvxpy import *
#    cvx_not_installable = True
#except ImportError:
#    cvx_not_installable = False
#
#
#if cvx_not_installable:
#    
#
#    u = Variable( N*N)
#    
#    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
#    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
#
#    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
#    
#    # create matrix representation for Astra operator
#
#    vol_geom = astra.create_vol_geom(N, N)
#    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)
#
#    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
#
#    matrix_id = astra.projector.matrix(proj_id)
#
#    ProjMat = astra.matrix.get(matrix_id)
#    
#    fidelity = 0.5 * sum_squares(ProjMat * u - noisy_data.as_array().ravel())
#    
#    # choose solver
#    if 'MOSEK' in installed_solvers():
#        solver = MOSEK
#    else:
#        solver = SCS  
#    
#    obj =  Minimize( regulariser +  fidelity)
#    prob = Problem(obj)
#    result = prob.solve(verbose = True, solver = solver)    
#
##%%
#    
#    u_rs = np.reshape(u.value, (N,N))
#        
#    diff_cvx = numpy.abs( res.as_array() - u_rs )
#    
#    # Show result
#    plt.figure(figsize=(15,15))
#    plt.subplot(3,1,1)
#    plt.imshow(res.as_array())
#    plt.title('PDHG solution')
#    plt.colorbar()
#    
#    plt.subplot(3,1,2)
#    plt.imshow(u_rs)
#    plt.title('CVX solution')
#    plt.colorbar()
#    
#    plt.subplot(3,1,3)
#    plt.imshow(diff_cvx)
#    plt.title('Difference')
#    plt.colorbar()
#    plt.show()
#    
#    plt.plot(np.linspace(0,N,N), res1.as_array()[int(N/2),:], label = 'PDHG')
#    plt.plot(np.linspace(0,N,N), u_rs[int(N/2),:], label = 'CVX')
#    plt.legend()
#    
#    
#    print('Primal Objective (CVX) {} '.format(obj.value))
#    print('Primal Objective (PDHG) {} '.format(primal[-1]))