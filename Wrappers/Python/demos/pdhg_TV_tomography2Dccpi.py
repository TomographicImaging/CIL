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

#from ccpi.astra.ops import AstraProjectorSimple
#from ccpi.plugins.ops import CCPiProjectorSimple
from ccpi.plugins.operators import CCPiProjectorSimple
#from skimage.util import random_noise
from timeit import default_timer as timer

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


#%%
#detectors = N
#angles = np.linspace(0,np.pi,100)
#angles_num = 100
angles_num = N
det_w = 1.0
det_num = N

angles = np.linspace(0,np.pi,angles_num,endpoint=False,dtype=np.float32)*\
             180/np.pi
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

from ccpi.reconstruction.parallelbeam import alg as pbalg
from ccpi.plugins.processors import setupCCPiGeometries
def ssetupCCPiGeometries(ig, ag, counter):
    
    #vg = ImageGeometry(voxel_num_x=voxel_num_x,voxel_num_y=voxel_num_y, voxel_num_z=voxel_num_z)
    #Phantom_ccpi = ImageData(geometry=vg,
    #                    dimension_labels=['horizontal_x','horizontal_y','vertical'])
    ##.subset(['horizontal_x','horizontal_y','vertical'])
    ## ask the ccpi code what dimensions it would like
    Phantom_ccpi = ig.allocate(dimension_labels=[ImageGeometry.HORIZONTAL_X, 
                                                 ImageGeometry.HORIZONTAL_Y,
                                                 ImageGeometry.VERTICAL])    
    
    voxel_per_pixel = 1
    angles = ag.angles
    geoms = pbalg.pb_setup_geometry_from_image(Phantom_ccpi.as_array(),
                                                angles,
                                                voxel_per_pixel )
    
    pg = AcquisitionGeometry('parallel',
                              '3D',
                              angles,
                              geoms['n_h'], 1.0,
                              geoms['n_v'], 1.0 #2D in 3D is a slice 1 pixel thick
                              )
    
    center_of_rotation = Phantom_ccpi.get_dimension_size('horizontal_x') / 2
    #ad = AcquisitionData(geometry=pg,dimension_labels=['angle','vertical','horizontal'])
    ad = pg.allocate(dimension_labels=[AcquisitionGeometry.ANGLE, 
                                       AcquisitionGeometry.VERTICAL,
                                       AcquisitionGeometry.HORIZONTAL])
    geoms_i = pbalg.pb_setup_geometry_from_acquisition(ad.as_array(),
                                                angles,
                                                center_of_rotation,
                                                voxel_per_pixel )
    
    counter+=1
    
    if counter < 4:
        print (geoms, geoms_i)
        if (not ( geoms_i == geoms )):
            print ("not equal and {} {} {}".format(counter, geoms['output_volume_z'], geoms_i['output_volume_z']))
            X = max(geoms['output_volume_x'], geoms_i['output_volume_x'])
            Y = max(geoms['output_volume_y'], geoms_i['output_volume_y'])
            Z = max(geoms['output_volume_z'], geoms_i['output_volume_z'])
            return setupCCPiGeometries(X,Y,Z,angles, counter)
        else:
            print ("happy now {} {} {}".format(counter, geoms['output_volume_z'], geoms_i['output_volume_z']))
            
            return geoms
    else:
        return geoms_i



#voxel_num_x, voxel_num_y, voxel_num_z, angles, counter
print ("###############################################")
print (ig)
print (ag)
g = setupCCPiGeometries(ig, ag, 0)
print (g)
print ("###############################################")
print ("###############################################")
#ag = AcquisitionGeometry('parallel','2D',angles, detectors)
#Aop = AstraProjectorSimple(ig, ag, 'cpu')
Aop = CCPiProjectorSimple(ig, ag, 'cpu')
sin = Aop.direct(data)

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



pdhg1 = PDHG(f=f,g=g, operator=operator, tau=tau, sigma=sigma, memopt=True, max_iteration=niter)
#pdhg1.max_iteration = 2000
pdhg1.update_objective_interval = 100
pdhg2 = PDHG(f=f,g=g, operator=operator, tau=tau, sigma=sigma, memopt=False)
pdhg2.max_iteration = 2000
pdhg2.update_objective_interval = 100

t1_old = timer()
resold, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
t2_old = timer()

print ("memopt = False, shouldn't matter")
pdhg1.run(niter)
print (sum(pdhg1.timing))
res = pdhg1.get_output().subset(vertical=0)
print (pdhg1.objective)
t3 = timer()
#res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) 
print ("memopt = True, shouldn't matter")
pdhg2.run(niter)
print (sum(pdhg2.timing))
res1 = pdhg2.get_output().subset(vertical=0)
t4 = timer()
#
print ("No memopt in {}s, memopt in  {}/{}s old {}s".format(sum(pdhg1.timing),
       sum(pdhg2.timing),t4-t3, t2_old-t1_old))

t1_old = timer()
resold1, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) 
t2_old = timer()

#%%
plt.figure()
plt.subplot(2,3,1)
plt.imshow(res.as_array())
plt.title('no memopt')
plt.colorbar()
plt.subplot(2,3,2)
plt.imshow(res1.as_array())
plt.title('memopt')
plt.colorbar()
plt.subplot(2,3,3)
plt.imshow((res1 - resold1.subset(vertical=0)).abs().as_array())
plt.title('diff')
plt.colorbar()
plt.subplot(2,3,4)
plt.imshow(resold.subset(vertical=0).as_array())
plt.title('old nomemopt')
plt.colorbar()
plt.subplot(2,3,5)
plt.imshow(resold1.subset(vertical=0).as_array())
plt.title('old memopt')
plt.colorbar()
plt.subplot(2,3,6)
plt.imshow((resold1 - resold).subset(vertical=0).as_array())
plt.title('diff old')
plt.colorbar()
#plt.plot(np.linspace(0,N,N), res1.as_array()[int(N/2),:], label = 'memopt')
#plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'no memopt')
#plt.legend()
plt.show()
#
print ("Time: No memopt in {}s, \n Time: Memopt in  {}s ".format(sum(pdhg1.timing), t4 -t3))
diff = (res1 - res).abs().as_array().max()
#
print(" Max of abs difference is {}".format(diff))

