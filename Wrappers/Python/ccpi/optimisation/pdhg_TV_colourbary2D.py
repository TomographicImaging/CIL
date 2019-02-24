#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:45:00 2019

@author: evangelos
"""

import numpy
from scipy.io import loadmat
import matplotlib.pyplot as plt

from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.optimisation.algs import CGLS, FISTA
from ccpi.optimisation.funcs import Norm2sq, Norm1, ZeroFun

import sys
sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/CCPi-astra/Wrappers/Python/ccpi/astra/')
from ops import *

from my_changes import *
from operators import *
from functions import *
from algorithms import *

#%%

# Load full data and permute to expected ordering. Change path as necessary.
# The loaded X has dims 80x60x80x150, which is pix x angle x pix x channel.
# Permute (numpy.transpose) puts into our default ordering which is
# (channel, angle, vertical, horizontal).

pathname = '/Users/evangelos/Desktop/Projects/'
filename = 'carbonPd_full_sinogram_stripes_removed.mat'

X = loadmat(pathname + filename)
X = numpy.transpose(X['SS'],(3,1,2,0))


# Store geometric variables for reuse
num_channels = X.shape[0]
num_pixels_h = X.shape[3]
num_pixels_v = X.shape[2]
num_angles = X.shape[1]

# Display a single projection in a single channel
plt.imshow(X[100,5,:,:])
plt.title('Example of a projection image in one channel' )
plt.show()

# Set angles to use
angles = numpy.linspace(-numpy.pi,numpy.pi,num_angles,endpoint=False)

# Define full 3D acquisition geometry and data container.
# Geometric info is taken from the txt-file in the same dir as the mat-file
ag = AcquisitionGeometry('cone',
                         '3D',
                         angles,
                         pixel_num_h=num_pixels_h,
                         pixel_size_h=0.25,
                         pixel_num_v=num_pixels_v,
                         pixel_size_v=0.25,                            
                         dist_source_center=233.0, 
                         dist_center_detector=245.0,
                         channels=num_channels)


data = AcquisitionData(X, geometry=ag)

#%%

# Reduce to central slice by extracting relevant parameters from data and its
# geometry. Perhaps create function to extract central slice automatically?
data2d = data.subset(vertical=40)
ag2d = AcquisitionGeometry('cone',
                         '2D',
                         ag.angles,
                         pixel_num_h=ag.pixel_num_h,
                         pixel_size_h=ag.pixel_size_h,
                         pixel_num_v=1,
                         pixel_size_v=ag.pixel_size_h,                            
                         dist_source_center=ag.dist_source_center, 
                         dist_center_detector=ag.dist_center_detector,
                         channels=ag.channels)
data2d.geometry = ag2d

# Set up 2D Image Geometry.
# First need the geometric magnification to scale the voxel size relative
# to the detector pixel size.
mag = (ag.dist_source_center + ag.dist_center_detector)/ag.dist_source_center
ig2d = ImageGeometry(voxel_num_x=ag2d.pixel_num_h, 
                     voxel_num_y=ag2d.pixel_num_h,  
                     voxel_size_x=ag2d.pixel_size_h/mag, 
                     voxel_size_y=ag2d.pixel_size_h/mag, 
                     channels=X.shape[0])

# Create GPU multichannel projector/backprojector operator with ASTRA.
Aall = AstraProjectorMC(ig2d,ag2d,'cpu')

# Compute and simple backprojction and display one channel as image.
Xbp = Aall.adjoint(data2d)
plt.imshow(Xbp.subset(channel=100).array)
plt.show()

#%%

alpha = 0.01

operator = CompositeOperator((2,1), gradient(ig2d), Aall )
f = [L1Norm(gradient(ig2d), alpha), Norm2sq_new(Aall, data2d, c = 0.5, memopt = False)]
g = ZeroFun()

#normK = compute_opNorm(operator)
normK = np.sqrt( operator.opMatrix()[0][0].opNorm()**2 + Aall.get_max_sing_val()**2 )

opt = {'niter':200, 'show_iter':100, 'stop_crit': cmp_L2norm}

# Primal & dual stepsizes
sigma = 1/normK
tau = 1/normK

t0 = time.time()
res, total_time, its = PDHG_testGeneric(data2d, f, g, operator, \
                                   ig2d, ag2d, tau = tau, sigma = sigma, opt = opt)
t1 = time.time()
print(t1-t0)


plt.imshow(res.as_array()[100])
plt.show()

#%%
#import time
#t0 = time.time()
#for i in range(100):
#    z = operator[0].direct(data2d)
#    w = operator[0].adjoint(z)
#    z1 = Aall.adjoint(data2d)
#    z2 = Aall.direct(z1)
#t1 = time.time()
#print(t1-t0)

#%%
## Set initial guess ImageData with zeros for algorithms, and algorithm options.
#x_init = ImageData(numpy.zeros((num_channels,num_pixels_v,num_pixels_h)),
#                   geometry=ig2d,
#                   dimension_labels=['channel','horizontal_y','horizontal_x'])
#opt_CGLS = {'tol': 1e-4, 'iter': 5}
#
## Run CGLS algorithm and display one channel.
#x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aall, data2d, opt_CGLS)
#
#plt.imshow(x_CGLS.subset(channel=100).array)
#plt.title('CGLS')
#plt.show()
#
#plt.semilogy(criter_CGLS)
#plt.title('CGLS Criterion vs iterations')
#plt.show()
#
## Create least squares object instance with projector, test data and a constant 
## coefficient of 0.5. Note it is least squares over all channels.
#f = Norm2sq(Aall,data2d,c=0.5)
#
## Options for FISTA algorithm.
#opt = {'tol': 1e-4, 'iter': 100}
#
## Run FISTA for least squares without regularization and display one channel
## reconstruction as image.
#x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt)
#
#plt.imshow(x_fista0.subset(channel=100).array)
#plt.title('FISTA LS')
#plt.show()
#
#plt.semilogy(criter0)
#plt.title('FISTA LS Criterion vs iterations')
#plt.show()
#
## Set up 1-norm regularisation (over all channels), solve with FISTA, and 
## display one channel of reconstruction.
#lam = 0.1
#g0 = Norm1(lam)
#
#x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)
#
#plt.imshow(x_fista1.subset(channel=100).array)
#plt.title('FISTA LS+1')
#plt.show()
#
#plt.semilogy(criter1)
#plt.title('FISTA LS+1 Criterion vs iterations')
#plt.show()