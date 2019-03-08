#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:50:18 2019

@author: evelina
"""

# All third-party imports.
import numpy
from astropy.io import fits
import csv
import os
from scipy.signal import convolve2d


# load angles
angles_file = open('/media/newhd/shared/Data/neutrondata/Feb2018_IMAT_rods/golden_ratio_angles.txt', 'r') 

angles = []
for angle in angles_file:
    angles.append(float(angle.strip('0')))
angles_file.close()

# parse file with shutter values
with open('/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/ShutterValues.txt', 'r') as shutter_values_csv:
    csv_reader = csv.reader(shutter_values_csv, delimiter = '\t')
    
    n_intervals = sum([1 for row in csv_reader])
   
    shutter_values_csv.seek(0)
    
    counter = 0
    tof_lim_1 = numpy.zeros(n_intervals, dtype = float)
    tof_lim_2 = numpy.zeros(n_intervals, dtype = float)
    tof_bin = numpy.zeros(n_intervals, dtype = float)
    
    for row in csv_reader:
        tof_lim_1[counter] = float(row[0])
        tof_lim_2[counter] = float(row[1])
        tof_bin[counter] = float(row[3])
        
        counter += 1

# calculate number of bins in each interval
# TOF is in seconds, bins in microseconds
n_bins = numpy.int_(numpy.floor((tof_lim_2 - tof_lim_1) / (tof_bin * 1e-6)))
n_bins_total = numpy.sum(n_bins)

# path to projections and flat images
pathname = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/'
projection_prefix = 'Angle_{}'
projection_channel_prefix = 'IMAT000{}_Tomo_000_'
projection_counter = 10699
flat_prefix = 'Flat{}'
flat_channel_prefix = 'IMAT000{}_ Tomo_000_'
flat_counter = 10887
image_format = 'fits'

# create a folder to store preprocessed images
path_preprocessed_data = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/'

if not os.path.isdir(path_preprocessed_data):
    os.mkdir(path_preprocessed_data)

# create a folder to store preprocessed images
path_preprocessed_projections = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/projections/'

if not os.path.isdir(path_preprocessed_projections):
    os.mkdir(path_preprocessed_projections)
    
# create a folder to store sum images
path_sum_images = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/sum_images/'

if not os.path.isdir(path_sum_images):
    os.mkdir(path_sum_images)
    
# create a folder to store averaged flats
path_av_flat = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed/av_flat/'

if not os.path.isdir(path_av_flat):
    os.mkdir(path_av_flat)

# image geometry
n_angles = 187
imsize = 512
n_channels = 2332

# -----------------------------------------------------------------------------
# Here we load 5 30 min flats, perform overlap correction, calculate averaged 
# flat and write corrected flats as well as averaged flat
# -----------------------------------------------------------------------------

# number of flats to average
n_flats = 5

# allocate memory for a single flat
flat = numpy.zeros((imsize, imsize, n_channels), dtype = int)
flat_n_triggers = numpy.zeros((n_channels), dtype = int)

flat_overlap_corrected =  numpy.zeros((imsize, imsize, n_channels), dtype = float)

# allocate memory for average flat
av_flat = numpy.zeros((imsize, imsize, n_channels), dtype = float)
av_flat_n_triggers = numpy.zeros((n_channels), dtype = float)

for i in range(n_flats):
    
    print('Loop through flats, flat # {}'.format(i))

    # parse file with shutter counts
    shutter_counts = numpy.zeros(256, dtype = 'int_')
    
    filename_shutter_counts = (pathname + 
                               flat_prefix + 
                               '/' + 
                               flat_channel_prefix + 
                               'ShutterCount.txt').format(i + 1, flat_counter + i)

    with open(filename_shutter_counts) as shutter_counts_csv:
        csv_reader = csv.reader(shutter_counts_csv, delimiter = '\t')
    
        counter = 0
    
        for row in csv_reader:
            shutter_counts[counter] = float(row[1])
            counter += 1
    
    # Load a flat field  image
    print('Loading current flat field image')
    for j in range(n_channels):
        
        # generate filename
        filename_flat = (pathname + 
                         flat_prefix + 
                         '/' + 
                         flat_channel_prefix + 
                         '{:05d}' + 
                         '.' + 
                         image_format).format(i + 1, flat_counter + i, j)
        
        # load file
        flat_handler = fits.open(filename_flat)
        flat_n_triggers[j] = flat_handler[0].header['N_TRIGS']
        flat_tmp = numpy.array(flat_handler[0].data, dtype = float)
        flat_handler.close()
        
        # mask dead pixels
        nonzero = flat_tmp > 0
                
        # mask zeros and accumulate
        dummy = convolve2d(flat_tmp, 
                           (numpy.ones((3, 3), dtype = float) / 9), 
                           mode = 'same', boundary = 'wrap') 
        
        flat[:, :, j] = numpy.where(nonzero, flat_tmp, dummy)
    
    # perform overlap correction
    print('Performing overlap correction')
    for j in range(n_intervals):
        
        tmp_sum = numpy.zeros((imsize, imsize), dtype = float)
        
        for k in range(n_bins[j]):
        
            idx = numpy.sum(n_bins[0:j]) + k
            
            tmp_sum += flat[:, :, idx]
            
            # calculate overlap correction factor
            denominator  = 1 - tmp_sum / shutter_counts[j]
            
            f = numpy.ones((imsize, imsize), dtype = float)
            f[denominator > 0] = 1 / denominator[denominator > 0]
    
            flat_overlap_corrected[:, :, idx] = f * flat[:, :, idx]
            
            if not os.path.isdir((path_preprocessed_projections + flat_prefix + '/').format(i + 1)):
                os.mkdir((path_preprocessed_projections + flat_prefix + '/').format(i + 1))
    
            # write corrected images
            filename_flat = (path_preprocessed_projections + 
                             flat_prefix + 
                             '/' + 
                             flat_channel_prefix + 
                             '{:05d}' + 
                             '_correcred.' + 
                             image_format).format(i + 1, flat_counter + i, idx)
    
            # write fits files
            hdu = fits.PrimaryHDU(numpy.float32(flat_overlap_corrected[:, :, idx]))
            hdul = fits.HDUList([hdu])
            hdul[0].header['N_TRIGS'] = flat_n_triggers[idx]
            hdul.writeto(filename_flat)
    
    # accumulate
    av_flat += flat_overlap_corrected
    av_flat_n_triggers += flat_n_triggers
    
# calculate average flat and write data
av_flat = av_flat / n_flats
av_flat_n_triggers = numpy.int_(numpy.around(av_flat_n_triggers / n_flats))

# store calculated flats
print('Writing calculated average flat')
for j in range(n_channels):
    
    # generate filename
    filename_flat = (path_av_flat + 
                     'av_flat_{:04d}' + 
                     '.' + 
                     image_format).format(j)
    
    # write fits files
    hdu = fits.PrimaryHDU(numpy.float32(av_flat[:,:,j]))
    hdul = fits.HDUList([hdu])
    hdul[0].header['N_TRIGS'] = av_flat_n_triggers[j]
    hdul.writeto(filename_flat)


#------------------------------------------------------------------------------
# Here we load projections, perform overlap correction, flat field correction 
# and trigger scaling
#------------------------------------------------------------------------------

# allocate memory for a single projection
projection = numpy.zeros((imsize, imsize, n_channels), dtype = int)
projection_n_triggers = numpy.zeros((n_channels), dtype = int)
projection_corrected =  numpy.zeros((imsize, imsize, n_channels), dtype = float)

# allocate memory for average flat
sum_image = numpy.zeros((imsize, imsize, n_channels), dtype = float)

for i in range(n_angles):
    
    print('Loop through projections, projection # {}'.format(i))

    # parse file with shutter counts
    shutter_counts = numpy.zeros(256, dtype = 'int_')
    
    filename_shutter_counts = (pathname + 
                               projection_prefix + 
                               '/' + 
                               projection_channel_prefix + 
                               'ShutterCount.txt').format(angles[i], projection_counter + i)

    with open(filename_shutter_counts) as shutter_counts_csv:
        csv_reader = csv.reader(shutter_counts_csv, delimiter = '\t')
    
        counter = 0
    
        for row in csv_reader:
            shutter_counts[counter] = float(row[1])
            counter += 1
    
    # Load a projection
    print('Loading current projection')
    for j in range(n_channels):
        
        # generate filename
        filename_projection = (pathname + 
                               projection_prefix + 
                               '/' + 
                               projection_channel_prefix + 
                               '{:05d}' + 
                               '.' + 
                               image_format).format(angles[i], projection_counter + i, j)
        
        # load file
        projection_handler = fits.open(filename_projection)
        projection_n_triggers[j] = projection_handler[0].header['N_TRIGS']
        projection_tmp = numpy.array(projection_handler[0].data, dtype = float)
        projection_handler.close()
        
        # mask dead pixels
        nonzero = projection_tmp > 0
                
        # mask zeros and accumulate
        dummy = convolve2d(projection_tmp, 
                           (numpy.ones((3, 3), dtype = float) / 9), 
                           mode = 'same', boundary = 'wrap') 
        
        projection[:, :, j] = numpy.where(nonzero, projection_tmp, dummy)
    
    # perform overlap correction
    print('Performing overlap correction, flat field correction and trigger scaling')
    for j in range(n_intervals):
        
        tmp_sum = numpy.zeros((imsize, imsize), dtype = float)
        
        for k in range(n_bins[j]):
        
            idx = numpy.sum(n_bins[0:j]) + k
            
            tmp_sum += projection[:, :, idx]
            
            # calculate overlap correction factor
            denominator  = 1 - tmp_sum / shutter_counts[j]
            
            f = numpy.ones((imsize, imsize), dtype = float)
            f[denominator > 0] = 1 / denominator[denominator > 0]
            
            # apply overlap correction
            projection_corrected[:, :, idx] = (f * projection[:, :, idx] / av_flat[:, :, idx]) * \
                                                (av_flat_n_triggers[idx] / projection_n_triggers[idx]) 
            
            if not os.path.isdir((path_preprocessed_projections + projection_prefix + '/').format(angles[i])):
                os.mkdir((path_preprocessed_projections + projection_prefix + '/').format(angles[i]))
    
            # write corrected images
            filename_projection = (path_preprocessed_projections + 
                                   projection_prefix + 
                                   '/' + 
                                   projection_channel_prefix + 
                                   '{:05d}' + 
                                   '_correcred' + 
                                   '.' + 
                                   image_format).format(angles[i], projection_counter + i, idx)
    
            # write fits files
            hdu = fits.PrimaryHDU(numpy.float32(projection_corrected[:, :, idx]))
            hdul = fits.HDUList([hdu])
            hdul[0].header['N_TRIGS'] = projection_n_triggers[idx]
            hdul.writeto(filename_projection)
    
    # accumulate
    print('Calculate sum image')
    sum_image = numpy.sum(projection_corrected, axis = 2)

    # write sum images
    # generate filename
    filename_sum_image = (path_sum_images + 
                          'sum_image_' + 
                          projection_prefix + 
                          '.' + 
                          image_format).format(angles[i])
    
    # write fits files
    hdu = fits.PrimaryHDU(numpy.float32(sum_image))
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename_sum_image)