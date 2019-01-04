#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:03:41 2019

@author: evangelos
"""

import numpy as np
import astra
import matplotlib.pyplot as plt
from cvx_functions import *
from cvxpy import *
import tomopy

#%%

N = 50

# Create phantom
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

# Create volume, geometry,
vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, 100, np.linspace(0,np.pi,100,False))

# create projector
proj_id = astra.create_projector('strip', proj_geom, vol_geom)

# create sinogram
sin_id, sin = astra.create_sino(x, proj_id, 'False') 

# create projection matrix
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
ProjMat = astra.matrix.get(matrix_id)

# add Noise
np.random.seed(1)

#noisy_sin = sin + 2 * np.random.randn(len(proj_geom['ProjectionAngles']),proj_geom['DetectorCount'])
noisy_sin = 0.5 * np.random.poisson(sin/0.5)

# Note, need to create an ID for the noisy sinogram and apply it to the rec object
noisy_sin_id = astra.data2d.create('-sino', proj_geom, noisy_sin);

plt.figure()
plt.imshow(x, cmap = 'viridis')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(sin, cmap = 'viridis')
plt.title('Sinogram')
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(noisy_sin, cmap = 'viridis')
plt.title('Noisy Sinogram')
plt.colorbar()
plt.show()


#%%
###############################################################################
        # Tomo-reconstruction for CT --> TV - L2
###############################################################################

# Note: Careful with "vec", "reshape" of cvxpy and "ravel" or "np.reshape". 
# They have different row/cols - ordering.
# ProjMat * u_tvCT needs row-order reshaping.
# vec is column-order reshaping.

# set regularising parameter
alpha_tvCT = 50

# Define the problem
u_tvCT = Variable( N*N, 1)

obj_tvCT =  Minimize( 0.5 * sum_squares(ProjMat * u_tvCT - noisy_sin.ravel()) + \
                    alpha_tvCT * tv_fun(reshape(u_tvCT, (N,N))) )

prob_tvCT = Problem(obj_tvCT)

# Choose solver, SCS is fast but less accurate than MOSEK
#res_tvCT = prob_tvCT.solve(verbose = True,solver = SCS,eps=1e-12)
res_tvCT = prob_tvCT.solve(verbose = True, solver = MOSEK)

print()
print('Objective value is {} '.format(obj_tvCT.value))

# Show result
plt.imshow(np.reshape(u_tvCT.value, (N,N)), cmap = 'viridis')
plt.title('Reconstruction TV')
plt.colorbar()
plt.show()

#%%
###############################################################################
                # Reconstruction using ASTRA
###############################################################################
    
# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# config for FBP
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = noisy_sin_id
cfg['ProjectorId'] = proj_id
cfg['FilterType'] = 'ram-lak' 

# possible values for FilterType:
# none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
# triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
# blackman-nuttall, flat-top, kaiser, parzen

# config for iterative reconstructions
#cfg = astra.astra_dict('ART')
#cfg['ReconstructionDataId'] = rec_id
#cfg['ProjectionDataId'] = noisy_sin_id
#cfg['ProjectorId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

astra.algorithm.run(alg_id, 10)

# Get the result
rec = astra.data2d.get(rec_id)

plt.imshow(rec, cmap = 'viridis')
plt.title('Reconstruction FBP')
plt.colorbar()
plt.show()

# Clean up.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
#astra.data2d.delete(noisy_sin_id)
#astra.projector.delete(proj_id)

#%%

ang = np.linspace(0,np.pi,100)
tmp = np.zeros((1,100,100))
tmp[0] = noisy_sin

rec_fbp = tomopy.recon(tmp, ang, algorithm = 'fbp', \
                       sinogram_order = True, num_gridx = 50, num_gridy = 50,
                       filter_name = 'ramlak')
plt.imshow(rec_fbp[0], cmap = 'viridis')
plt.title('Reconstruction FBP')
plt.colorbar()
plt.show()

rec_sirt = tomopy.recon(tmp, ang, algorithm = 'sirt', num_iter = 10, sinogram_order = True, num_gridx = 50, num_gridy = 50)
plt.imshow(rec_sirt[0], cmap = 'viridis')
plt.title('Reconstruction SIRT')
plt.colorbar()
plt.show()
