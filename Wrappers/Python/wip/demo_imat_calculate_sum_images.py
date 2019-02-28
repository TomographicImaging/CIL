#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Feb 28 14:15:45 2019

@author: evelina
'''

# This script loop through IMAT data, 
# applies flat field correction, triggers scaling,dead pixel correction,
# and generates sum images.
# We laso calculate and generate average flat field correction 
# from 5 cquiured 30 min flats.


# All third-party imports.
import numpy
from astropy.io import fits
import os
from scipy.signal import convolve2d


# load angles
angles_file = open('/media/newhd/shared/Data/neutrondata/Feb2018_IMAT_rods/golden_ratio_angles.txt', 'r') 

angles = []
for angle in angles_file:
    angles.append(float(angle.strip('0')))
angles_file.close()

# path to projections and flat images
pathname = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/'
projection_prefix = 'Angle_{}'
projection_channel_prefix = 'IMAT000{}_Tomo_000_0'
projection_counter = 10699
flat_prefix = 'Flat{}'
flat_channel_prefix = 'IMAT000{}_ Tomo_000_0'
flat_counter = 10887
image_format = 'fits'

# create a folder to store sum images
path_to_sum_images = '/home/evelina/sum_images/'

if not os.path.isdir(path_to_sum_images):
    os.mkdir(path_to_sum_images)

# create a folder to store average flats
path_to_av_flat = '/home/evelina/av_flat/'

if not os.path.isdir(path_to_av_flat):
    os.mkdir(path_to_av_flat)

# image geometry
n_angles = 187
imsize = 512
n_channels = 2332

# number of flats to average
n_flats = 5

# allocate memory for average flat, skip zero channel
av_flat = numpy.zeros((imsize, imsize, n_channels-1), dtype = float)
av_flat_n_triggers = numpy.zeros((n_channels-1), dtype = float)

# calculate average flat
for i in range(n_flats):
    
    print('Loop through flats, flat # {}'.format(i))
    
    for j in range(n_channels-1):
        # Load a flat field  image
        # generate filename
        filename_flat = (pathname + flat_prefix + '/' + flat_channel_prefix + '{:04d}' + '.' + image_format).format(i + 1, flat_counter + i, j + 1)
        
        # load flat
        flat_handler = fits.open(filename_flat)
        av_flat_n_triggers[j] += flat_handler[0].header['N_TRIGS']
        flat_tmp = numpy.array(flat_handler[0].data, dtype = float)
        flat_handler.close()
        
        # accumulate
        av_flat[:, :, j] += flat_tmp

# round and cast to int to avoid division by nearly zero numbers during flat field correction
av_flat = numpy.int_(numpy.around(av_flat / n_flats))
av_flat_n_triggers = numpy.int_(numpy.around(av_flat_n_triggers / n_flats))


# store calculated flats
for j in range(n_channels-1):
    # generate filename
    filename_flat = (path_to_av_flat + 'av_flat_{:04d}' + '.' + image_format).format(j + 1)
    
    # write fits files
    hdu = fits.PrimaryHDU(numpy.uint16(av_flat[:,:,j]))
    hdul = fits.HDUList([hdu])
    hdul[0].header['N_TRIGS'] = av_flat_n_triggers[j]
    hdul.writeto(filename_flat)
    

# Calculate sum images

# loop through projections and apply flats
for i in range(n_angles):
    
    print('Generating sum image # {}'.format(i))
          
    # allocate array to store sum image
    sum_image = numpy.zeros((imsize, imsize), dtype = float)
    
    # zero channel is always zero and we skip it
    for j in range(n_channels-1):
        
        tmp = numpy.zeros((imsize, imsize), dtype = float)
        
        # generate filename (j+1 to skip zero channel)
        filename_projection  = (pathname + projection_prefix + '/' + projection_channel_prefix + '{:04d}' + '.' + image_format).format(angles[i], projection_counter + i, j + 1)
        
        # load projection
        projection_handler = fits.open(filename_projection)
        projection_tmp = numpy.array(projection_handler[0].data, dtype = float)
        projection_n_triggers = projection_handler[0].header['N_TRIGS']
        projection_handler.close()
        
        # flat field correction + trigger scaling
        nonzero = av_flat[:, :, j] > 0
        tmp[nonzero] = (projection_tmp[nonzero] / av_flat[nonzero, j]) * (av_flat_n_triggers[j] / projection_n_triggers)     
                
        # mask zeros and accumulate
        dummy = convolve2d(tmp, (numpy.ones((3, 3), dtype = float) / 9), mode = 'same', boundary = 'wrap')
        sum_image += numpy.where(nonzero, tmp, dummy)
        
    # save sum images
    filename_sum_image  = (path_to_sum_images + 'sum_image_' + projection_prefix + '.' + image_format).format(angles[i])
        
    # write fits files, we do not cast to uint to avoid precision loss
    hdu = fits.PrimaryHDU(sum_image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename_sum_image)