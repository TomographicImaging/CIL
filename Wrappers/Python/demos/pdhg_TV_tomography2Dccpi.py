# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer, \
   AcquisitionGeometry, AcquisitionData

import numpy as np                           
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, ScaledFunction

from ccpi.plugins.operators import CCPiProjectorSimple
from timeit import default_timer as timer
from ccpi.reconstruction.parallelbeam import alg as pbalg
import os

try:
    import tomophantom
    from tomophantom import TomoP3D
    no_tomophantom = False
except ImportError as ie:
    no_tomophantom = True

#%%

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
#x = np.zeros((N,N))

vert = 4
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, voxel_num_z=vert)

angles_num = 100
det_w = 1.0
det_num = N

angles = np.linspace(-90.,90.,N, dtype=np.float32) 
# Inputs: Geometry, 2D or 3D, angles, horz detector pixel count, 
#         horz detector pixel size, vert detector pixel count, 
#         vert detector pixel size.
ag = AcquisitionGeometry('parallel',
                         '3D',
                         angles,
                         N, 
                         det_w,
                         vert,
                         det_w)

#no_tomophantom = True
if no_tomophantom:
    data = ig.allocate()
    Phantom = data
    # Populate image data by looping over and filling slices
    i = 0
    while i < vert:
        if vert > 1:
            x = Phantom.subset(vertical=i).array
        else:
            x = Phantom.array
        x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
        x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 0.98
        if vert > 1 :
            Phantom.fill(x, vertical=i)
        i += 1
    
    Aop = CCPiProjectorSimple(ig, ag, 'cpu')
    sin = Aop.direct(data)
else:
        
    model = 13 # select a model number from the library
    N_size = N # Define phantom dimensions using a scalar value (cubic phantom)
    path = os.path.dirname(tomophantom.__file__)
    path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
    #This will generate a N_size x N_size x N_size phantom (3D)
    phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
    
    #%%
    Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
    Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
    #angles_num = int(0.5*np.pi*N_size); # angles number
    #angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
    
    print ("Building 3D analytical projection data with TomoPhantom")
    projData3D_analyt = TomoP3D.ModelSino(model, 
                                                      N_size, 
                                                      Horiz_det, 
                                                      Vert_det, 
                                                      angles, 
                                                      path_library3D)
    
    # tomophantom outputs in [vert,angles,horiz]
    # we want [angle,vert,horiz]
    data = np.transpose(projData3D_analyt, [1,0,2])
    ag.pixel_num_h = Horiz_det
    ag.pixel_num_v = Vert_det
    sin = ag.allocate()
    sin.fill(data)
    ig.voxel_num_y = Vert_det
    
    Aop = CCPiProjectorSimple(ig, ag, 'cpu')
    

plt.imshow(sin.subset(vertical=0).as_array())
plt.title('Sinogram')
plt.colorbar()
plt.show()


#%%
# Add Gaussian noise to the sinogram data
np.random.seed(10)
n1 = np.random.random(sin.shape)

noisy_data = sin + ImageData(5*n1)

plt.imshow(noisy_data.subset(vertical=0).as_array())
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
g = ZeroFunction()

normK = Aop.norm()

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
    sigma = 1
    tau = 1/(sigma*normK**2)

# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
niter = 50
opt = {'niter':niter}
opt1 = {'niter':niter, 'memopt': True}



pdhg1 = PDHG(f=f,g=g, operator=operator, tau=tau, sigma=sigma, max_iteration=niter)
#pdhg1.max_iteration = 2000
pdhg1.update_objective_interval = 100

t1_old = timer()
resold, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
t2_old = timer()

pdhg1.run(niter)
print (sum(pdhg1.timing))
res = pdhg1.get_output().subset(vertical=0)

#%%
plt.figure()
plt.subplot(1,4,1)
plt.imshow(res.as_array())
plt.title('Algorithm')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(resold.subset(vertical=0).as_array())
plt.title('function')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow((res - resold.subset(vertical=0)).abs().as_array())
plt.title('diff')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'Algorithm')
plt.plot(np.linspace(0,N,N), resold.subset(vertical=0).as_array()[int(N/2),:], label = 'function')
plt.legend()
plt.show()
#
print ("Time: No memopt in {}s, \n Time: Memopt in  {}s ".format(sum(pdhg1.timing), t2_old -t1_old))
diff = (res - resold.subset(vertical=0)).abs().as_array().max()
#
print(" Max of abs difference is {}".format(diff))

