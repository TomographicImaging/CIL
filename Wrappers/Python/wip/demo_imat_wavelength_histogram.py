#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Feb 22 10:42:07 2019

@author: evelina
'''

# This script parse sgutter values file and 
# generate wavelength histogram of channel data of a single projection


# All third-party imports.
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
import csv



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
    
    line_count = 0
    tof_lim_1 = numpy.zeros(n_intervals, dtype = float)
    tof_lim_2 = numpy.zeros(n_intervals, dtype = float)
    tof_bin = numpy.zeros(n_intervals, dtype = float)
    
    for row in csv_reader:
        tof_lim_1[line_count] = float(row[0])
        tof_lim_2[line_count] = float(row[1])
        tof_bin[line_count] = float(row[3])
        
        line_count += 1
        
# TOF is in seconds, bins in microseconds
n_bins = numpy.int_(numpy.floor((tof_lim_2 - tof_lim_1) / (tof_bin * 1e-6)))
n_bins_total = numpy.sum(n_bins)

# x axis for histogram
bins = numpy.zeros((n_bins_total + (n_intervals )), dtype = float)
counter = 0
for i in range(n_intervals):
    bins[counter:(counter + n_bins[i] + 1)] = numpy.arange(tof_lim_1[i], tof_lim_2[i], tof_bin[i] * 1e-6, dtype = float)
    counter += n_bins[i] + 1

# convert TOF to Angstrom
# m_n – neutron mass = 1.67*10e-27 kg,
# h – Planck’s constant = 6.63*10-34 Js 
# l - flight path = 56.4 m
l = 56.4
# full equation
# angstrom_lim_1 = (tof_lim_1 * const.h) / (const.m_n * l) * 1e10
# angstrom_lim_2 = (tof_lim_2 * const.h) / (const.m_n * l) * 1e10
# and it's simplified form
angstrom_lim_1 = (tof_lim_1 * 3957) / l
angstrom_lim_2 = (tof_lim_2 * 3957) / l
angstrom_bins = (bins * 3957) / l

# path to projections
pathname = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/'
projection_prefix = 'Angle_{}'
projection_channel_prefix = 'IMAT000{}_Tomo_000_0'
projection_counter = 10699
path_to_av_flat = '/home/evelina/av_flat/'
flat_prefix = 'av_flat_'
image_format = 'fits'

# image geometry
n_angles = 187
imsize = 512
n_channels = 2332

print('Loading averaged flat images')

# load averaged flat images
# allocate memory (no zero channel)
av_flat = numpy.zeros((imsize, imsize, n_channels-1), dtype = int)
av_flat_n_triggers = numpy.zeros((n_channels-1), dtype = int)

for j in range(n_channels - 1):
    
    filename_flat = (path_to_av_flat + 'av_flat_{:04d}' + '.' + image_format).format(j + 1)
    
    # load flat
    flat_handler = fits.open(filename_flat)
    av_flat_n_triggers[j] = flat_handler[0].header['N_TRIGS']
    flat_tmp = numpy.array(flat_handler[0].data, dtype = int)
    flat_handler.close()
    
    av_flat[:, :, j] = flat_tmp


# load projection
projection_id = 0

projection = numpy.zeros((imsize, imsize, n_channels - 1), dtype = float)
sum_image = numpy.zeros((imsize, imsize), dtype = float)

# load one projection and visualise intensity plot
# zero channel is always zero and we skip it
for j in range(n_channels-1):
    
    print('Loading channel # {}'.format(j))
   
    # generate filename
    filename_projection  = (pathname + projection_prefix + '/' + projection_channel_prefix + '{:04d}' + '.' + image_format).format(angles[projection_id], projection_counter + projection_id, j + 1)

    # load projection
    projection_handler = fits.open(filename_projection)
    projection_tmp = numpy.array(projection_handler[0].data, dtype = float)
    projection_n_triggers = projection_handler[0].header['N_TRIGS']
    projection_handler.close()
    
    flat_tmp = av_flat[:, :, j]

    # flat field correction + scaling
    nonzero = flat_tmp > 0
    projection[nonzero, j] = (projection_tmp[nonzero] / av_flat[nonzero, j]) * (av_flat_n_triggers[j] / projection_n_triggers) 
    sum_image[nonzero] += projection[nonzero, j]
   

plt.imshow(sum_image, cmap = 'gray')
plt.title('Sum image')
plt.colorbar()
plt.show()

plt.imshow(projection[:,:,100], cmap = 'gray')
plt.title('Channel # 100')
plt.colorbar()
plt.show()

# generate histogram for selected ROI
# left top corner
roi_row = 50
roi_col = 50
roi_pix_v = 100 # number of pixels along vertical axis
roi_pix_h = 100 # number of pixels along horizontal axis

# visualise ROI
fig, ax = plt.subplots(1)
ax.imshow(sum_image, cmap = 'gray')
rect = patches.Rectangle((roi_col, roi_row), roi_pix_v, roi_pix_h, linewidth = 1, edgecolor = 'r', facecolor = 'none') # Create a Rectangle patch
ax.add_patch(rect)
plt.title('Selected ROI')
plt.colorbar()
plt.show()

#  calculate histogram value in every bin
histogram_values = numpy.zeros(n_channels + (n_intervals - 1))

counter = 0
for i in range(n_intervals):
    roi = projection[roi_row:(roi_row+roi_pix_v), roi_col:(roi_col+roi_pix_h), counter:(counter+n_bins[i])]
    histogram_values[(counter + i + 1):((counter + n_bins[i]) + i + 1)] = numpy.sum(numpy.sum(roi, axis = 0), axis = 0)
    counter += n_bins[i]

# mask zeros between intervals (now we simply take average of neighbouring bins, no "overlap" correction)
# should we re-implement "overlap" correction?
histogram_values[0] = histogram_values[1]
counter = 0
for i in range(n_intervals-1):
    counter += n_bins[i]
    histogram_values[counter + 1 + i] = (histogram_values[counter + i] + histogram_values[counter + 2 + i]) / 2
    print(counter)

# plot histogram

# calculate middle point of every bin
x_axis = (angstrom_bins[1:] + angstrom_bins[0:-1]) / 2   

plt.plot(x_axis, histogram_values)
plt.xlabel('Wavelength, Angstrom')
plt.ylabel('Intensity, arbitrary units')
plt.title('Intensity vs Wavelength, ROI = [{} {}] - [{} {}]'.format(roi_row, roi_col, roi_row + roi_pix_v, roi_col + roi_pix_h))
plt.show()

