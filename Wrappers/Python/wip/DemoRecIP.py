#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reading multi-channel data and reconstruction using FISTA modular
"""

import numpy as np
import matplotlib.pyplot as plt

#import sys
#sys.path.append('../../../data/')
from read_IPdata import read_IPdata

from ccpi.astra.astra_ops import AstraProjector, AstraProjectorSimple, AstraProjectorMC
from ccpi.reconstruction.funcs import Norm2sq, Norm1, BaseFunction
from ccpi.reconstruction.algs import FISTA
#from ccpi.reconstruction.funcs import BaseFunction

from ccpi.framework import ImageData, AcquisitionData, AcquisitionGeometry, ImageGeometry

# read IP paper data into a dictionary
dataDICT = read_IPdata('..\..\..\data\IP_data70channels.mat')

# Set ASTRA Projection-backprojection class (fan-beam geometry)
DetWidth = dataDICT.get('im_size')[0] * dataDICT.get('det_width')[0] / \
            dataDICT.get('detectors_numb')[0]
SourceOrig = dataDICT.get('im_size')[0] * dataDICT.get('src_to_rotc')[0] / \
            dataDICT.get('dom_width')[0]
OrigDetec = dataDICT.get('im_size')[0] * \
            (dataDICT.get('src_to_det')[0] - dataDICT.get('src_to_rotc')[0]) /\
            dataDICT.get('dom_width')[0]

N = dataDICT.get('im_size')[0]

vg = ImageGeometry(voxel_num_x=dataDICT.get('im_size')[0],
                   voxel_num_y=dataDICT.get('im_size')[0],
                   channels=1)

pg = AcquisitionGeometry('cone',
                         '2D',
                         angles=(np.pi/180)*dataDICT.get('theta')[0],
                         pixel_num_h=dataDICT.get('detectors_numb')[0],
                         pixel_size_h=DetWidth,
                         dist_source_center=SourceOrig, 
                         dist_center_detector=OrigDetec,
                         channels=1)


sino = dataDICT.get('data_norm')[0][:,:,34] # select mid-channel 
b = AcquisitionData(sino,geometry=pg)

# Initial guess
x_init = ImageData(np.zeros((N, N)),geometry=vg)





Aop = AstraProjectorSimple(vg,pg,'gpu')
f = Norm2sq(Aop,b,c=0.5)

# Run FISTA for least squares without regularization
opt = {'tol': 1e-4, 'iter': 10}
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt)

plt.imshow(x_fista0.array)
plt.show()

# Now least squares plus 1-norm regularization
g1 = Norm1(10)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1, opt)

plt.imshow(x_fista1.array)
plt.show()

# Multiple channels
sino_mc = dataDICT.get('data_norm')[0][:,:,32:37] # select mid-channel 

vg_mc = ImageGeometry(voxel_num_x=dataDICT.get('im_size')[0],
                   voxel_num_y=dataDICT.get('im_size')[0],
                   channels=5)

pg_mc = AcquisitionGeometry('cone',
                         '2D',
                         angles=(np.pi/180)*dataDICT.get('theta')[0],
                         pixel_num_h=dataDICT.get('detectors_numb')[0],
                         pixel_size_h=DetWidth,
                         dist_source_center=SourceOrig, 
                         dist_center_detector=OrigDetec,
                         channels=5)

b_mc = AcquisitionData(np.transpose(sino_mc,(2,0,1)),
                       geometry=pg_mc, 
                       dimension_labels=("channel","angle","horizontal"))

# ASTRA operator using volume and sinogram geometries
Aop_mc = AstraProjectorMC(vg_mc, pg_mc, 'gpu')

f_mc = Norm2sq(Aop_mc,b_mc,c=0.5)

# Initial guess
x_init_mc = ImageData(np.zeros((5, N, N)),geometry=vg_mc)


x_fista0_mc, it0_mc, timing0_mc, criter0_mc = FISTA(x_init_mc, f_mc, None, opt)

plt.imshow(x_fista0_mc.as_array()[4,:,:])
plt.show()