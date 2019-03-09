from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData
from astropy.io import fits

import astra

import numpy as np
import matplotlib.pyplot as plt

from Algorithms import PDHG
from Operators.CompositeOperator_DataContainer import CompositeOperator, CompositeDataContainer
from Operators.GradientOperator import Gradient
from Operators.AstraProjectorSimpleOperator import AstraProjectorSimple

from Functions.FunctionComposition import FunctionComposition_new
from Functions.mixed_L12Norm import mixed_L12Norm
from Functions.L2NormSquared import L2NormSq
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
padded_width = int(numpy.ceil(abs(delta)) + imsize)
delta_pix = padded_width - imsize

data_rel_padded = numpy.zeros((n_angles, padded_width), dtype = float)

nonzero = sino > 0
#%%
# take negative logarithm and pad image
data_rel = np.zeros(sino.shape)
data_rel[nonzero] = -numpy.log(sino[nonzero] / numpy.amax(sino))
    
data_rel_padded[:, :-delta_pix] = data_rel

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
Aop = AstraProjectorSimple(ig2d,ag2d,'gpu')
    
# Set initial guess ImageData with zeros for algorithms, and algorithm options.
x_init = ImageData(numpy.zeros((padded_width, padded_width)),
                       geometry=ig2d)
    
opt_CGLS = {'tol': 1e-4, 'iter': 100}

    # Run CGLS algorithm and display reconstruction.
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b2d, opt_CGLS)
    

