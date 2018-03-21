
from ccpi.io.reader import XTEKReader
import numpy as np
import matplotlib.pyplot as plt
from ccpi.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData, ImageData
from ccpi.astra.astra_ops import AstraProjectorSimple
from ccpi.reconstruction.algs import CGLS

# Set up reader object and read the data
datareader = XTEKReader("C:/Users/mbbssjj2/Documents/SophiaBeads_256_averaged/SophiaBeads_256_averaged.xtekct")
data = datareader.getAcquisitionData()

# Extract central slice, scale and negative-log transform
sino = -np.log(data.as_array()[:,:,1000]/60000.0)

# Apply centering correction by zero padding, amount found manually
cor_pad = 30
sino_pad = np.zeros((sino.shape[0],sino.shape[1]+cor_pad))
sino_pad[:,cor_pad:] = sino

# Simple beam hardening correction as done in SophiaBeads coda
sino_pad = sino_pad**2

# Extract AcquisitionGeometry for central slice for 2D fanbeam reconstruction
ag2d = AcquisitionGeometry('cone',
                          '2D',
                          angles=-np.pi/180*data.geometry.angles,
                          pixel_num_h=data.geometry.pixel_num_h + cor_pad,
                          pixel_size_h=data.geometry.pixel_size_h,
                          dist_source_center=data.geometry.dist_source_center, 
                          dist_center_detector=data.geometry.dist_center_detector)

# Set up AcquisitionData object for central slice 2D fanbeam
data2d = AcquisitionData(sino_pad,geometry=ag2d)

# Choose the number of voxels to reconstruct onto as number of detector pixels
N = data.geometry.pixel_num_h

# Geometric magnification
mag = (np.abs(data.geometry.dist_center_detector) + \
      np.abs(data.geometry.dist_source_center)) / \
      np.abs(data.geometry.dist_source_center)

# Voxel size is detector pixel size divided by mag
voxel_size_h = data.geometry.pixel_size_h / mag

# Construct the appropriate ImageGeometry
ig2d = ImageGeometry(voxel_num_x=N,
                   voxel_num_y=N,
                   voxel_size_x=voxel_size_h, 
                   voxel_size_y=voxel_size_h)

# Set up the Projector (AcquisitionModel) using ASTRA on GPU
Aop = AstraProjectorSimple(ig2d, ag2d,"gpu")

# Set initial guess for CGLS reconstruction
x_init = ImageData(np.zeros((N,N)),geometry=ig2d)

# Run CGLS reconstruction
num_iter = 50
x, it, timing, criter = CGLS(Aop,data2d,num_iter,x_init)

# Display reconstruction
plt.figure()
plt.imshow(x.as_array())