#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 22:16:06 2019

@author: evangelos
"""

import tomopy
import matplotlib.pyplot as plt
import numpy as np
#%%

N = 50

# Create phantom
x = np.zeros((1,N,N))
x[0,round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[0,round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

obj = x # Generate an object.
ang = np.linspace(0,np.pi,100,False)

t1 = np.zeros((1,100,100))
t1[0] = noisy_sin

rec_fbp = tomopy.recon(t1, ang, algorithm = 'fbp', sinogram_order = True, num_gridx = 50, num_gridy = 50)
plt.imshow(rec_fbp[0], cmap = 'viridis')
plt.title('Object')
plt.show()

rec_sirt = tomopy.recon(t1, ang, algorithm = 'tv', num_iter = 100, sinogram_order = True, num_gridx = 50, num_gridy = 50)
plt.imshow(rec_sirt[0], cmap = 'viridis')
plt.title('Object')
plt.show()

#['art', 'bart', 'fbp', 'gridrec', \
# 'mlem', 'osem', 'ospml_hybrid', \
# 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad'], or a Python method.


#%%

sim = tomopy.project(obj, ang, pad = False)


test = tomopy.sim.project.project(obj, 100, center=None, \
                           emission=True, pad=False, sinogram_order=False, ncore=None, nchunk=None)

plt.imshow(sim[:,0,:], cmap = 'viridis')
plt.title('Object')
plt.show()

#%%



rec1 = tomopy.recon(t1, ang, algorithm = tomopy.astra, num_gridx = 50, num_gridy = 50 ) # Reconstruct object.

#'fbp': ['num_gridx', 'num_gridy', 'filter_name', 'filter_par'],
plt.imshow(rec1[0], cmap = 'viridis')
plt.title('TV - Reconstruction')
plt.show()


#%%
# Reconstruct object:
rec2 = tomopy.recon(sim, ang, algorithm=tomopy.astra,
       options={'method':'ART', 'num_iter':10*180,
      'proj_type':'linear',
       'extra_options':{'MinConstraint':0}})

#%%
# Show 64th slice of the reconstructed object.
import pylab
pylab.imshow(rec1[0], cmap='gray')
pylab.show()

pylab.imshow(rec2[0], cmap='gray')
pylab.show()

pylab.imshow(obj[0], cmap='gray')
pylab.show()




























#%%

import astra
import numpy as np
import scipy.io


c = np.linspace(-127.5,127.5,256)
x, y = np.meshgrid(c,c)
mask = np.array((x**2 + y**2 < 127.5**2),dtype=np.float)

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(mask)

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,50,False))

# As before, create a sinogram from a phantom
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']

#%%
proj_id = astra.create_projector('strip', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

pylab.figure(2)
pylab.imshow(P)
pylab.figure(3)
pylab.imshow(sinogram)



#%%

import astra
import numpy as np

N = 100

vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, 150, np.linspace(0,np.pi,180,False))

# For CPU-based algorithms, a "projector" object specifies the projection
# model used. In this case, we use the "line" model.
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# Generate the projection matrix for this projection model.
# This creates a matrix W where entry w_{i,j} corresponds to the
# contribution of volume element j to detector element i.
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
W = astra.matrix.get(matrix_id)



x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

s = W.dot(x.ravel())
s = np.reshape(s, (len(proj_geom['ProjectionAngles']),proj_geom['DetectorCount']))

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(s)
pylab.show()

#%%

# Each row of the projection matrix corresponds to a detector element.
# Detector t for angle p is for row 1 + t + p*proj_geom.DetectorCount.
# Each column corresponds to a volume pixel.
# Pixel (x,y) corresponds to column 1 + x + y*vol_geom.GridColCount.


astra.projector.delete(proj_id)
astra.matrix.delete(matrix_id)






import numpy as np

def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


u1 = np.random.randint(10, size = (2,3,2))

print(u1)
print()
print(gradient(u1))
















