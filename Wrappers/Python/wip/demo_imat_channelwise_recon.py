#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:35:45 2019

@author: evelina
"""

# All third-party imports.
import numpy
from astropy.io import fits
import os
import matplotlib.pyplot as plt


# All own imports.
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.optimisation.algs import CGLS


# load angles
angles_file = open('/media/newhd/shared/Data/neutrondata/Feb2018_IMAT_rods/golden_ratio_angles.txt', 'r') 

angles = []
for angle in angles_file:
    angles.append(float(angle.strip('0')))
angles_file.close()

pathname = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/projections/'
projection_prefix = 'Angle_{}'
projection_channel_prefix = 'IMAT000{}_Tomo_000_'
projection_counter = 10699
image_format = 'fits'

# image geometry
n_angles = 187
imsize = 512
n_channels = 2332

# sort sinograms to make sinograms readable
angles = numpy.array(angles[:n_angles])
idx = numpy.argsort(angles)

# specify  number of slice to reconstruct
slice_id = 350

# create a folder to store sinos
path_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/sino/'

if not os.path.isdir(path_sino):
    os.mkdir(path_sino)
    
if not os.path.isdir((path_sino + 'slice_{}'.format(slice_id))):
    os.mkdir((path_sino + 'slice_{}'.format(slice_id)))

# create a folder to store recons
path_recon = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/recon/'

if not os.path.isdir(path_recon):
    os.mkdir(path_recon) 

if not os.path.isdir((path_recon + 'slice_{}'.format(slice_id))):
    os.mkdir((path_recon + 'slice_{}'.format(slice_id))) 

# allocate memory
sino = numpy.zeros((n_angles, imsize), dtype = float)
data_rel = numpy.zeros((n_angles, imsize), dtype = float)

# compensate for CoR shift
# cor offset
cor = (imsize / 2 + 52 / 2) 

# padding
delta = imsize - 2 * numpy.abs(cor)
padded_width = int(numpy.ceil(abs(delta)) + imsize)
delta_pix = padded_width - imsize

data_rel_padded = numpy.zeros((n_angles, padded_width), dtype = float)

for i in range(n_channels):
    
    print('Reconstructing channel # {}'.format(i))
    
    for j in range(n_angles):
        
        filename_projection = (pathname + 
                               projection_prefix + 
                               '/' + 
                               projection_channel_prefix + 
                               '{:05d}' + 
                               '_correcred' + 
                               '.' + 
                               image_format).format(angles[j], projection_counter + j, i) 

        projection_handler = fits.open(filename_projection)
        slice_tmp = numpy.array(projection_handler[0].data[:, slice_id], dtype = float)
        projection_handler.close()
        
        sino[j, :] = slice_tmp
    
    sino = sino[idx, :]
    
    filename_sino = (path_sino + 
                     'slice_{}/sino_{}' + 
                     '.' + 
                     image_format).format(slice_id, i)
        
    # write fits files
    hdu = fits.PrimaryHDU(numpy.float32(sino))
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename_sino)
    
    nonzero = sino > 0
    # take negative logarithm and pad image
    data_rel[nonzero] = -numpy.log(sino[nonzero])
    
    data_rel_padded[:, :-delta_pix] = data_rel
        
    # Create 2D acquisition geometry and acquisition data
    ag2d = AcquisitionGeometry('parallel',
                               '2D',
                               numpy.array(angles[idx]) * numpy.pi / 180,
                               pixel_num_h = padded_width)

    # Create 2D image geometry
    ig2d = ImageGeometry(voxel_num_x = ag2d.pixel_num_h, 
                     voxel_num_y = ag2d.pixel_num_h)

    b2d = AcquisitionData(data_rel_padded, geometry = ag2d)
    
    # Create GPU projector/backprojector operator with ASTRA.
    Aop = AstraProjectorSimple(ig2d,ag2d,'gpu')
    
    # Set initial guess ImageData with zeros for algorithms, and algorithm options.
    x_init = ImageData(numpy.zeros((padded_width, padded_width)),
                       geometry=ig2d)
    
    opt_CGLS = {'tol': 1e-4, 'iter': 100}

    # Run CGLS algorithm and display reconstruction.
    x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b2d, opt_CGLS)
    
    filename_recon = (path_recon + 
                      'slice_{}/recon_{}' + 
                      '.' + 
                      image_format).format(slice_id, i)
        
    # write fits files
    hdu = fits.PrimaryHDU(numpy.float32(x_CGLS.array))
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename_recon)

    #plt.imshow(x_CGLS.array, cmap = 'gray')
    #plt.title('CGLS channel {}'.format(i))
    #plt.colorbar()
    #plt.show()        
    