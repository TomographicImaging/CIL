import time

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.algs import FISTA, FBPD, CGLS, SIRT
from ccpi.optimisation.funcs import Norm2sq, Norm1, IndicatorBox, ZeroFun
from ccpi.optimisation.ops import Operator
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.optimisation.ops import PowerMethodNonsquare

import astra

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from my_changes import *


import sys
sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts/')
from cvx_functions import *



#%%

N = 75
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

detectors = 100
angles = np.linspace(0,np.pi,100)
SourceOrig = 200
OrigDetec = 0

# parallel
ag = AcquisitionGeometry('parallel','2D',angles,detectors)

#cone 
#ag = AcquisitionGeometry('cone','2D',angles,detectors,sourcecenter=SourceOrig, centerTodetector=OrigDetec)


# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# create sinogram and noisy sinogram
sin = Aop.direct(Phantom)

np.random.seed(1)
scale = 0.5
#noisy_sin = AcquisitionData(scale * np.random.poisson(sin1.array/scale))
noisy_sin = AcquisitionData(sin.as_array() + np.random.normal(0, 2, sin.shape))

# simple backprojection
backproj = Aop.adjoint(noisy_sin)

plt.imshow(x)
plt.title('Phantom image')
plt.show()

plt.imshow(noisy_sin.array)
plt.title('Simulated data')
plt.show()

plt.imshow(backproj.array)
plt.title('Backprojected data')
plt.show()

#%% reg.parameter

alpha = 50

##%%  Solve with CVX
vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, 100, np.linspace(0,np.pi,100,False))

# create projector
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# create sinogram
sin_id, sin2 = astra.create_sino(x, proj_id, 'False') 

# create projection matrix
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
ProjMat = astra.matrix.get(matrix_id)

ProjMat1 = astra.OpTomo(proj_id)

#%%

# Define the problem
#u_cvx = Variable( N*N, 1)
#obj_cvx =  Minimize( 0.5 * sum_squares(ProjMat * u_cvx - noisy_sin.as_array().ravel()) + \
#                    alpha * TV_cvx(reshape(u_cvx, (N,N))) )
#
#prob_cvx = Problem(obj_cvx)
#
## Choose solver, SCS is fast but less accurate than MOSEK
##res_tvCT = prob_tvCT.solve(verbose = True,solver = SCS,eps=1e-12)
#res_cvx = prob_cvx.solve(verbose = True, solver = MOSEK)
#
#print()
#print('Objective value is {} '.format(obj_cvx.value))

#%% Solve with pdhg


operator = form_Operator(gradient(ig), Aop)
normK = compute_opNorm(operator) 

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

#sigma = 0.9/normK
#tau = 0.9/normK

alpha = 50  

f = [L1Norm(alpha), Norm2sq_new(Aop, noisy_sin, c = 0.5, memopt = False)]
g = ZeroFun()

cProfile.run('res, total_time, its = PDHG(noisy_sin, f, g, operator, \
                                   ig, ag, tau = tau, sigma = sigma, opt = None)')

#%%
# Show result
plt.imshow(res.as_array(), cmap = 'viridis')
plt.title('Reconstruction with PDHG')
plt.colorbar()
plt.show()


plt.imshow(np.reshape(u_cvx.value, (N,N)), cmap = 'viridis')
plt.title('Reconstruction with CVX')
plt.colorbar()
plt.show()


plt.imshow(np.abs(res.as_array() - np.reshape(u_cvx.value, (N,N))))
plt.colorbar()
plt.show()