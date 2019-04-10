# This script demonstrates how to load IMAT fits data
# into the CIL optimisation framework and run reconstruction methods.
#
# This demo loads the summedImg files which are the non-spectral images 
# resulting from summing projections over all spectral channels.

# needs dxchange: conda install -c conda-forge dxchange
# needs astropy: conda install astropy


# All third-party imports.
import numpy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from dxchange.reader import read_fits

# All own imports.
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.astra.ops import AstraProjectorSimple, AstraProjector3DSimple
from ccpi.optimisation.algs import CGLS, FISTA
from ccpi.optimisation.funcs import Norm2sq, Norm1

# Load and display a couple of summed projection as examples
pathname0 = '/media/newhd/shared/Data/neutrondata/PSI_phantom_IMAT/DATA/Sample/angle0/'
filename0 = 'IMAT00004675_Tomo_test_000_SummedImg.fits'

data0 = read_fits(pathname0 + filename0)

pathname10 = '/media/newhd/shared/Data/neutrondata/PSI_phantom_IMAT/DATA/Sample/angle10/'
filename10 = 'IMAT00004685_Tomo_test_000_SummedImg.fits'

data10 = read_fits(pathname10 + filename10)

# Load a flat field (more are available, should we average over them?)
flat1 = read_fits('/media/newhd/shared/Data/neutrondata/PSI_phantom_IMAT/DATA/OpenBeam_aft1/IMAT00004932_Tomo_test_000_SummedImg.fits')

# Apply flat field and display after flat-field correction and negative log
data0_rel = numpy.zeros(numpy.shape(flat1), dtype = float)
nonzero = flat1 > 0
data0_rel[nonzero] = data0[nonzero] / flat1[nonzero]
data10_rel =  numpy.zeros(numpy.shape(flat1), dtype = float)
data10_rel[nonzero] = data10[nonzero] / flat1[nonzero]

plt.figure()
plt.imshow(data0_rel)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(-numpy.log(data0_rel))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(data10_rel)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(-numpy.log(data10_rel))
plt.colorbar()
plt.show()

# Set up for loading all summed images at 250 angles.
pathname = '/media/newhd/shared/Data/neutrondata/PSI_phantom_IMAT/DATA/Sample/angle{}/'
filename = 'IMAT0000{}_Tomo_test_000_SummedImg.fits'

# Dimensions
num_angles = 250
imsize = 512

# Initialise array
data = numpy.zeros((num_angles,imsize,imsize))

# Load only 0-249, as 250 is at repetition of zero degrees just like image 0
for i in range(0,250):
    curimfile = (pathname + filename).format(i, i+4675)
    data[i,:,:] = read_fits(curimfile)

# Apply flat field and take negative log
nonzero = flat1 > 0
for i in range(0,250):
    data[i,nonzero] = data[i,nonzero]/flat1[nonzero]

eqzero = data == 0
data[eqzero] = 1

data_rel = -numpy.log(data)

# Permute order to get: angles, vertical, horizontal, as default in framework.
data_rel = numpy.transpose(data_rel,(0,2,1))

# Set angles to use
angles = numpy.linspace(-numpy.pi,numpy.pi,num_angles,endpoint=False)

# Create 3D acquisition geometry and acquisition data
ag = AcquisitionGeometry('parallel',
                         '3D',
                         angles,
                         pixel_num_h=imsize,
                         pixel_num_v=imsize)
b = AcquisitionData(data_rel, geometry=ag)

# Reduce to single (noncentral) slice by extracting relevant parameters from data and its
# geometry. Perhaps create function to extract central slice automatically?
b2d = b.subset(vertical=128)
ag2d = AcquisitionGeometry('parallel',
                         '2D',
                         ag.angles,
                         pixel_num_h=ag.pixel_num_h)
b2d.geometry = ag2d

# Create 2D image geometry
ig2d = ImageGeometry(voxel_num_x=ag2d.pixel_num_h, 
                     voxel_num_y=ag2d.pixel_num_h)

# Create GPU projector/backprojector operator with ASTRA.
Aop = AstraProjectorSimple(ig2d,ag2d,'gpu')

# Demonstrate operator is working by applying simple backprojection.
z = Aop.adjoint(b2d)
plt.figure()
plt.imshow(z.array)
plt.title('Simple backprojection')
plt.colorbar()
plt.show()

# Set initial guess ImageData with zeros for algorithms, and algorithm options.
x_init = ImageData(numpy.zeros((imsize,imsize)),
                   geometry=ig2d)
opt_CGLS = {'tol': 1e-4, 'iter': 20}

# Run CGLS algorithm and display reconstruction.
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b2d, opt_CGLS)

plt.figure()
plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(criter_CGLS)
plt.title('CGLS Criterion vs iterations')
plt.show()