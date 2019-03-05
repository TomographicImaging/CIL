#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Feb 26 09:04:22 2019

@author: evelina
'''

# In this script we load 0 and 180 degree images and 
# use cross-correlation to calculate centre of rotation.
# Then we zero-pad projections and perform CGLS reconstruction.
# Calculated CoR value can be used to initialise CenterOfRotationFinder

# All third-party imports.
import numpy
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.signal import convolve2d


# All own imports.
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.optimisation.algs import CGLS


# path to 0 degree
pathname_0 = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/Angle_0.0/'
projection_0_channel_prefix = 'IMAT00010699_Tomo_000_0'

# path to 180 degree
pathname_180 = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/180/'
projection_180_channel_prefix = 'IMAT00010886_ Tomo_000_0'

# path to flats
pathname = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/'
flat_prefix = 'Flat{}'
flat_channel_prefix = 'IMAT000{}_ Tomo_000_0'
flat_counter = 10887
image_format = 'fits'

# image geometry
imsize = 512
n_channels = 2332

# allocate memory
sum_image_0 = numpy.zeros((imsize, imsize), dtype = float)
sum_image_180 = numpy.zeros((imsize, imsize), dtype = float)

for j in range(n_channels-1):
    
    # Load a flat field image
    # generate filename
    filename_flat = (pathname + flat_prefix + '/' + flat_channel_prefix + '{:04d}' + '.' + image_format).format(1, flat_counter, j + 1)
    
    # load flat
    flat_handler = fits.open(filename_flat)
    flat_n_triggers = flat_handler[0].header['N_TRIGS']
    flat_tmp = numpy.array(flat_handler[0].data, dtype = float)
    flat_handler.close()
   
    
    # 0 degree
    # generate filename
    filename_0_projection  = (pathname_0 + projection_0_channel_prefix + '{:04d}' + '.' + image_format).format(j + 1)

    # load projection
    projection_handler = fits.open(filename_0_projection)
    projection_tmp = numpy.array(projection_handler[0].data, dtype = float)
    projection_n_triggers = projection_handler[0].header['N_TRIGS']
    projection_handler.close()

    # flat field correction + scaling
    nonzero = flat_tmp > 0
    sum_image_0[nonzero] += (projection_tmp[nonzero] / flat_tmp[nonzero]) * (flat_n_triggers / projection_n_triggers)
    
    
    # 180 degree
    # generate filename
    filename_180_projection  = (pathname_180 + projection_180_channel_prefix + '{:04d}' + '.' + image_format).format(j + 1)

    # load projection
    projection_handler = fits.open(filename_180_projection)
    projection_tmp = numpy.array(projection_handler[0].data, dtype = float)
    projection_n_triggers = projection_handler[0].header['N_TRIGS']
    projection_handler.close()

    # flat field correction + scaling
    nonzero = flat_tmp > 0    
    sum_image_180[nonzero] += (projection_tmp[nonzero] / flat_tmp[nonzero]) * (flat_n_triggers / projection_n_triggers)


# flip 180 deree projection
sum_image_180 = numpy.flip(sum_image_180, axis = 0)

plt.imshow(sum_image_0, cmap = 'gray')
plt.title('0 degree')
plt.colorbar()
plt.show()

plt.imshow(sum_image_180, cmap = 'gray')
plt.title('180 degree (flipped)')
plt.colorbar()
plt.show()


# registration with scikit-image to get initial estimate
# following an example here
# https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html#sphx-glr-auto-examples-transform-plot-register-translation-py
# pixel precision first
shift, error, diffphase = register_translation(sum_image_0, sum_image_180)

'''
fig = plt.figure(figsize = (8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex = ax1, sharey = ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(sum_image_0, cmap = 'gray')
ax1.set_axis_off()
ax1.set_title('0 degrees')

ax2.imshow(sum_image_180.real, cmap = 'gray')
ax2.set_axis_off()
ax2.set_title('180 degrees (flipped)')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = numpy.fft.fft2(sum_image_0) * numpy.fft.fft2(sum_image_180).conj()
cc_image = numpy.fft.fftshift(numpy.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title('Cross-correlation')

plt.show()
'''

print('Detected pixel offset (y, x): {}'.format(shift))


'''
# subpixel precision
shift, error, diffphase = register_translation(sum_image_0, sum_image_180, 100)

fig = plt.figure(figsize = (8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex = ax1, sharey = ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(sum_image_0, cmap = 'gray')
ax1.set_axis_off()
ax1.set_title('0 degrees')

ax2.imshow(sum_image_180.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('180 degrees (flipped)')

# Calculate the upsampled DFT, again to show what the algorithm is doing
# behind the scenes.  Constants correspond to calculated values in routine.
# See source code for details.
cc_image = _upsampled_dft(image_product, 150, 100, (shift * 100) + 75).conj()
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title('Supersampled XC sub-area')


plt.show()

print('Detected subpixel offset (y, x): {}'.format(shift))
'''


# Reconstruct some slices with new CoR
# load angles
angles_file = open('/media/newhd/shared/Data/neutrondata/Feb2018_IMAT_rods/golden_ratio_angles.txt', 'r') 

angles = []
for angle in angles_file:
    angles.append(float(angle.strip('0')))
angles_file.close()

# path to sum images
path_to_sum_images = '/home/evelina/sum_images/'
projection_prefix = 'Angle_{}'
image_format = 'fits'

# image geometry
n_angles = 187
imsize = 512

# allocate memory and load sum images
sum_image = numpy.zeros((imsize, imsize, n_angles), dtype = float)

for i in range(n_angles):
    
    filename_sum_image  = (path_to_sum_images + 'sum_image_' + projection_prefix + '.' + image_format).format(angles[i])
    
    # load projection
    projection_handler = fits.open(filename_sum_image)
    projection_tmp = numpy.array(projection_handler[0].data, dtype = float)
    projection_handler.close()
    
    sum_image[:,:,i] = projection_tmp
    
# cor offset
cor = (imsize / 2 + shift[0] / 2) 

# padding
delta = imsize - 2 * numpy.abs(cor)
padded_width = int(numpy.ceil(abs(delta)) + imsize)
delta_pix = padded_width - imsize

# take negative logarithm and zero-pad image
data_rel = -numpy.log(sum_image / numpy.amax(sum_image))
data_rel_padded = numpy.zeros((padded_width, imsize, n_angles), dtype = float)

data_rel_padded[:-delta_pix,:,:] = data_rel

# some reconstructions
slices_to_recon = numpy.arange(10, 500, 50, dtype = int)

# Create 2D acquisition geometry and acquisition data
ag2d = AcquisitionGeometry('parallel',
                         '2D',
                         numpy.array(angles[0:n_angles]) * numpy.pi / 180,
                         pixel_num_h = padded_width)

# Create 2D image geometry
ig2d = ImageGeometry(voxel_num_x = ag2d.pixel_num_h, 
                     voxel_num_y = ag2d.pixel_num_h)

for i in range(slices_to_recon.size):
    b2d = AcquisitionData(numpy.transpose(data_rel_padded[:,slices_to_recon[i],:]), geometry = ag2d)
    
    # Create GPU projector/backprojector operator with ASTRA.
    Aop = AstraProjectorSimple(ig2d,ag2d,'gpu')
    
    # Set initial guess ImageData with zeros for algorithms, and algorithm options.
    x_init = ImageData(numpy.zeros((padded_width, padded_width)),
                       geometry=ig2d)
    opt_CGLS = {'tol': 1e-4, 'iter': 100}

    # Run CGLS algorithm and display reconstruction.
    x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b2d, opt_CGLS)

    plt.imshow(x_CGLS.array, cmap = 'gray')
    plt.title('CGLS slice {}'.format(slices_to_recon[i]))
    plt.colorbar()
    plt.show()
