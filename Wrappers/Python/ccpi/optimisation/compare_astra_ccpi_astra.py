#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:29:40 2019

@author: evangelos
"""

# Compare astra ccpi-astra 

import astra 

import numpy as np
import matplotlib.pyplot as plt
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData, DataContainer
from ccpi.astra.ops import AstraProjectorSimple

N = 75

# Create phantom
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

detectors = 300

# Create volume, geometry,
vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, np.linspace(0,np.pi,100,False))

# create projector
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# create sinogram
sin_id, sin = astra.create_sino(x, proj_id, 'False') 

# create projection matrix
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
#ProjMat = astra.matrix.get(matrix_id)
ProjMat = astra.OpTomo(proj_id)

#s = W.dot(P.ravel())
sin2 = np.reshape(ProjMat.dot(x.ravel()), (len(proj_geom['ProjectionAngles']),proj_geom['DetectorCount']))

# This is the same as proj_geom
ag = AcquisitionGeometry('parallel','2D',np.linspace(0,np.pi,100,False),proj_geom['DetectorCount'])

# This is the same as vol_geom
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)

Aop = AstraProjectorSimple(ig, ag, 'cpu')

# create sinogram and noisy sinogram
sin3 = Aop.direct(ImageData(x, geometry = ig))

plt.imshow(sin)
plt.colorbar()
plt.title('Astra ')
plt.show()

plt.imshow(sin2)
plt.colorbar()
plt.title('Astra Matrix')
plt.show()

plt.imshow(sin3.as_array())
plt.colorbar()
plt.title('CCPi-astra')
plt.show()

plt.imshow(np.abs(sin2 - sin3.as_array()))
plt.colorbar()
plt.title('Difference astra CCPi-astra')
plt.show()

#%%
backproj_astra = np.reshape(ProjMat.T * sin2.ravel(), x.shape)
backproj_ccpi = Aop.adjoint(sin3)

plt.imshow(backproj_astra)
plt.colorbar()
plt.title('Backproj with AstraMat')
plt.show()

plt.imshow(backproj_ccpi.as_array())
plt.colorbar()
plt.title('Backproj with CCPi-astra')
plt.show()

plt.imshow(np.abs(backproj_astra-backproj_ccpi.as_array()))
plt.colorbar()
plt.title('Difference astra CCPi-astra backproj')
plt.show()


#%%

W = astra.OpTomo(proj_id)
fp = W*x
bp = W.T*sin

