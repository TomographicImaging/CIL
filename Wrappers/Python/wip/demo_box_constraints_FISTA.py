#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:46:21 2019

@author: jakob

Demonstrate the use of box constraints in FISTA
"""

# First make all imports
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.algorithms import FISTA
from ccpi.optimisation.functions import Norm2sq, IndicatorBox
from ccpi.astra.ops import AstraProjectorSimple

from ccpi.optimisation.operators import Identity

import numpy as np
import matplotlib.pyplot as plt


# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 128
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.figure()
plt.imshow(x)
plt.title('Phantom image')
plt.show()

# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam and 0 to 2pi for fanbeam, set the width of a detector 
# pixel relative to an object pixel, the number of detector pixels, and the 
# source-origin and origin-detector distance (here the origin-detector distance 
# set to 0 to simulate a "virtual detector" with same detector pixel size as
# object pixel size).
angles_num = 20
det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

test_case = 1

if test_case==1:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num,det_w)
elif test_case==2:
    angles = np.linspace(0,2*np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             det_num,
                             det_w,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)
else:
    NotImplemented

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.
Aop = AstraProjectorSimple(ig, ag, 'gpu')

Aop = Identity(ig,ig)

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
b = Aop.direct(Phantom)
z = Aop.adjoint(b)

plt.figure()
plt.imshow(b.array)
plt.title('Simulated data')
plt.show()

plt.figure()
plt.imshow(z.array)
plt.title('Backprojected data')
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 100}



# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = Norm2sq(Aop,b,c=0.5)

# Run FISTA for least squares without constraints
FISTA_alg = FISTA()
FISTA_alg.set_up(x_init=x_init, f=f, opt=opt)
FISTA_alg.max_iteration = 2000
FISTA_alg.run(opt['iter'])
x_FISTA = FISTA_alg.get_output()

plt.figure()
plt.imshow(x_FISTA.array)
plt.title('FISTA unconstrained')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg.objective)
plt.title('FISTA unconstrained criterion')
plt.show()

# Run FISTA for least squares with lower bound 0.1
FISTA_alg0 = FISTA()
FISTA_alg0.set_up(x_init=x_init, f=f, g=IndicatorBox(lower=0.1), opt=opt)
FISTA_alg0.max_iteration = 2000
FISTA_alg0.run(opt['iter'])
x_FISTA0 = FISTA_alg0.get_output()

plt.figure()
plt.imshow(x_FISTA0.array)
plt.title('FISTA lower bound 0.1')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg0.objective)
plt.title('FISTA criterion, lower bound 0.1')
plt.show()

# Run FISTA for least squares with box constraint [0.1,0.8]
FISTA_alg0 = FISTA()
FISTA_alg0.set_up(x_init=x_init, f=f, g=IndicatorBox(lower=0.1,upper=0.8), opt=opt)
FISTA_alg0.max_iteration = 2000
FISTA_alg0.run(opt['iter'])
x_FISTA0 = FISTA_alg0.get_output()

plt.figure()
plt.imshow(x_FISTA0.array)
plt.title('FISTA box(0.1,0.8) constrained')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg0.objective)
plt.title('FISTA criterion, box(0.1,0.8) constrained criterion')
plt.show()