#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:47:08 2019

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
from test_functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq, CompositeFunction

from Sparse_GradMat import GradOper

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


# Create volume, geometry,
vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, 100, np.linspace(0,np.pi,100,False))

# create projector
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# create sinogram
sin_id, sin1 = astra.create_sino(x.as_array(), proj_id, 'False') 

# create projection matrix
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
ProjMat = astra.matrix.get(matrix_id)

plt.imshow(sin.as_array())
plt.show()

plt.imshow(sin1)
plt.show()

z = np.abs(sin.as_array()-sin1)
plt.imshow(z)
plt.colorbar()
plt.show()
#%%









