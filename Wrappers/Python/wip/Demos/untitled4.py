#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:21:08 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData
import numpy  
import numpy as np                        
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorSimple, AstraProjector3DSimple
from skimage.util import random_noise
from timeit import default_timer as timer

#N = 75
#x = np.zeros((N,N))
#x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
#x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

#data = ImageData(x)

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
    
angles_num = 100
det_w = 1.0
det_num = N

angles = np.linspace(0,np.pi,angles_num,endpoint=False,dtype=np.float32)*\
             180/np.pi

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

sino = numpy.load("sinogram_demo_tomography2D.npy", mmap_mode='r')
plt.imshow(sino)
plt.title('Sinogram CCPi')
plt.colorbar()
plt.show()
             
#%%
Aop = AstraProjector3DSimple(ig, ag)
sin = Aop.direct(data)

plt.imshow(sin.as_array())

plt.title('Sinogram Astra')
plt.colorbar()
plt.show()



#%%