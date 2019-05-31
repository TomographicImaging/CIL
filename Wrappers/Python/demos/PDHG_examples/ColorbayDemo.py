#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================


from ccpi.framework import ImageGeometry, ImageData, AcquisitionGeometry, AcquisitionData, BlockDataContainer

import numpy as numpy                 
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, CGLS
from ccpi.optimisation.algs import CGLS as CGLS_old

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.operators import AstraProjectorMC
from scipy.io import loadmat
import h5py

#%%

phantom = 'powder'

if phantom == 'carbon':
    pathname = '/media/newhd/shared/Data/ColourBay/spectral_data_sets/CarbonPd/'
    filename = 'carbonPd_full_sinogram_stripes_removed.mat'
    X = loadmat(pathname + filename)
    X = numpy.transpose(X['SS'],(3,1,2,0))
    
elif phantom == 'powder':
    pathname = '/media/newhd/shared/DataProcessed/' 
    filename = 'S_180.mat'
    path = pathname + filename
    arrays = {}
    f = h5py.File(path)
    for k, v in f.items():
        arrays[k] = numpy.array(v)
    XX = arrays['S']    
    X = numpy.transpose(XX,(0,2,1,3))
    X = X[0:250]
    
        
    
#%% Setup Geometry of Colorbay

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
Aall = AstraProjectorMC(ig2d,ag2d,'gpu')

# Compute and simple backprojction and display one channel as image.
Xbp = Aall.adjoint(data2d)
plt.imshow(Xbp.subset(channel=100).array)
plt.show()

#%% CGLS

x_init = ig2d.allocate()      
cgls1 = CGLS(x_init=x_init, operator=Aall, data=data2d)
cgls1.max_iteration = 100
cgls1.update_objective_interval = 1
cgls1.run(5,verbose=True)

plt.imshow(cgls1.get_output().subset(channel=100).array)
plt.title('CGLS')
plt.show()

#%% Tikhonov 

alpha = 2.5
Grad = Gradient(ig2d, correlation=Gradient.CORRELATION_SPACE) # use also CORRELATION_SPACECHANNEL

# Form Tikhonov as a Block CGLS structure
op_CGLS = BlockOperator( Aall, alpha * Grad, shape=(2,1))
block_data = BlockDataContainer(data2d, Grad.range_geometry().allocate())
    
cgls2 = CGLS(x_init=x_init, operator=op_CGLS, data=block_data)
cgls2.max_iteration = 100
cgls2.update_objective_interval = 1

cgls2.run(10,verbose=True)

plt.imshow(cgls2.get_output().subset(channel=100).array)
plt.title('Tikhonov')
plt.show()

#%% Total Variation

# Regularisation Parameter
alpha_TV = 50

# Create operators
op1 = Gradient(ig2d, correlation=Gradient.CORRELATION_SPACE)
op2 = Aall

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions      
f1 = alpha * MixedL21Norm()
f2 = 0.5 * L2NormSquared(b=data2d)    
f = BlockFunction(f1, f2)  
g = ZeroFunction()

# Compute operator Norm
normK = 8.70320267279591 # Run one time no need to compute again takes time
    
# Primal & dual stepsizes
sigma = 10
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 100
pdhg.run(1000, verbose =True)


#%% Show sinograms
channel_ind = [25,75,125]

plt.figure(figsize=(15,15))

plt.subplot(4,3,1)
plt.imshow(data2d.subset(channel = channel_ind[0]).as_array())
plt.title('Channel {}'.format(channel_ind[0]))
plt.colorbar()

plt.subplot(4,3,2)
plt.imshow(data2d.subset(channel = channel_ind[1]).as_array())
plt.title('Channel {}'.format(channel_ind[1]))
plt.colorbar()

plt.subplot(4,3,3)
plt.imshow(data2d.subset(channel = channel_ind[2]).as_array())
plt.title('Channel {}'.format(channel_ind[2]))
plt.colorbar()

###############################################################################
# Show CGLS
plt.subplot(4,3,4)
plt.imshow(cgls1.get_output().subset(channel = channel_ind[0]).as_array())
plt.colorbar()

plt.subplot(4,3,5)
plt.imshow(cgls1.get_output().subset(channel = channel_ind[1]).as_array())
plt.colorbar()

plt.subplot(4,3,6)
plt.imshow(cgls1.get_output().subset(channel = channel_ind[2]).as_array())
plt.colorbar()

###############################################################################
# Show Tikhonov

plt.subplot(4,3,7)
plt.imshow(cgls2.get_output().subset(channel = channel_ind[0]).as_array())
plt.colorbar()

plt.subplot(4,3,8)
plt.imshow(cgls2.get_output().subset(channel = channel_ind[1]).as_array())
plt.colorbar()

plt.subplot(4,3,9)
plt.imshow(cgls2.get_output().subset(channel = channel_ind[2]).as_array())
plt.colorbar()


###############################################################################
# Show Total variation

plt.subplot(4,3,10)
plt.imshow(pdhg.get_output().subset(channel = channel_ind[0]).as_array())
plt.colorbar()

plt.subplot(4,3,11)
plt.imshow(pdhg.get_output().subset(channel = channel_ind[1]).as_array())
plt.colorbar()

plt.subplot(4,3,12)
plt.imshow(pdhg.get_output().subset(channel = channel_ind[2]).as_array())
plt.colorbar()


###############################################################################












