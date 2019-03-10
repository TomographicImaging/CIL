#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData
from astropy.io import fits
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.optimisation.algs import CGLS

import astra

import numpy
from numpy import inf
import matplotlib.pyplot as plt
from cvxpy import *

from Algorithms import PDHG

from Operators import CompositeOperator, Identity, Gradient, CompositeDataContainer, AstraProjectorSimple
from Functions import ZeroFun, L2NormSq, mixed_L12Norm, FunctionOperatorComposition, BlockFunction

#%%

filename = '/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/working_with_CompDataContainers/IMAT_recon/sino_0_sum.fits'
        
# load file
file_handler = fits.open(filename)
sino = numpy.array(file_handler[0].data, dtype = float)
file_handler.close()

#sino is in float32 format, dimensions are 187 x 512

#you need to compensate for CoR offset
#I implemented this way:
imsize = 512
n_angles = 187

# compensate for CoR shift
# cor offset
cor = (imsize / 2 + 52 / 2) 

# padding
delta = imsize - 2 * numpy.abs(cor)
padded_width = int(numpy.ceil(np.abs(delta)) + imsize)
delta_pix = padded_width - imsize

data_rel_padded = numpy.zeros((n_angles, padded_width), dtype = float)

nonzero = sino > 0
#%%
# take negative logarithm and pad image
data_rel = np.zeros(sino.shape)
data_rel[nonzero] = -numpy.log(sino[nonzero] / numpy.amax(sino))
    
data_rel_padded[:, :-delta_pix] = data_rel

#%%

with open("/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/working_with_CompDataContainers/IMAT_recon/golden_angles.txt") as f:
    angles_string = [line.rstrip() for line in f]
    angles = np.array(angles_string).astype(float)
#%%

# Create 2D acquisition geometry and acquisition data
ag2d = AcquisitionGeometry('parallel',
                           '2D',
                            angles * numpy.pi / 180,
                            pixel_num_h = padded_width)

# Create 2D image geometry
ig2d = ImageGeometry(voxel_num_x = ag2d.pixel_num_h, 
                     voxel_num_y = ag2d.pixel_num_h)

b2d = AcquisitionData(data_rel_padded, geometry = ag2d)
    
    # Create GPU projector/backprojector operator with ASTRA.
Aop = AstraProjectorSimple(ig2d,ag2d,'cpu')

alpha = 80
f = FunctionOperatorComposition(Aop, L2NormSq(0.5, b=b2d))
g0 = L2NormSq(alpha)


opt = { 'iter': 400}

x_init = ImageData(np.zeros((564,564)))

x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)

plt.imshow(x_fista1.as_array())
plt.colorbar()
plt.show()