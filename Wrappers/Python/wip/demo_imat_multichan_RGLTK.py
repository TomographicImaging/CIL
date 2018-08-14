# This script demonstrates how to load IMAT fits data
# into the CIL optimisation framework and run reconstruction methods.
#
# This demo loads the summedImg files which are the non-spectral images 
# resulting from summing projections over all spectral channels.

# needs dxchange: conda install -c conda-forge dxchange
# needs astropy: conda install astropy


# All third-party imports.
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from dxchange.reader import read_fits

# All own imports.
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.astra.ops import AstraProjectorSimple, AstraProjector3DSimple
from ccpi.optimisation.algs import CGLS, FISTA
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.plugins.regularisers import ROF_TV, FGP_TV, SB_TV


pathname0 = '/media/algol/HD-LXU3/DATA_DANIIL/PSI_DATA/DATA/Sample/angle0/'
filenameG = "IMAT00004675_Tomo_test_000_"

n = 512 
totalAngles = 250 # total number of projection angles
# spectral discretization parameter
num_average = 120
numChannels = 2970
totChannels = round(numChannels/num_average) # total no. of averaged channels
Projections_stack = np.zeros((num_average,n,n),dtype='uint16')
ProjAngleChannels = np.zeros((totalAngles,totChannels,n,n),dtype='float32')

counterT = 0
for i in range(0,2,1):
    for j in range(0,num_average,1):
        if counterT < 10:
            outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathname0,filenameG,'0000',str(counterT))
        if ((counterT  >= 10) & (counterT < 100)):
            outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathname0,filenameG,'000',str(counterT))
        if ((counterT  >= 100) & (counterT < 1000)):
            outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathname0,filenameG,'00',str(counterT))
        if ((counterT  >= 1000) & (counterT < 10000)):
            outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathname0,filenameG,'0',str(counterT))
        Projections_stack[j,:,:] = read_fits(outfile)
        counterT = counterT + 1
    AverageProj=np.mean(Projections_stack,axis=0) # averaged projection
ProjAngleChannels[0,i,:,:] = AverageProj


filename0 = 'IMAT00004675_Tomo_test_000_SummedImg.fits'

data0 = read_fits(pathname0 + filename0)

pathname10 = '/media/algol/HD-LXU3/DATA_DANIIL/PSI_DATA/DATA/Sample/angle10/'
filename10 = 'IMAT00004685_Tomo_test_000_SummedImg.fits'

data10 = read_fits(pathname10 + filename10)

# Load a flat field (more are available, should we average over them?)
flat1 = read_fits('/media/algol/HD-LXU3/DATA_DANIIL/PSI_DATA/DATA/OpenBeam_aft1/IMAT00004932_Tomo_test_000_SummedImg.fits')

# Apply flat field and display after flat-field correction and negative log
data0_rel = np.zeros(np.shape(flat1), dtype = float)
nonzero = flat1 > 0
data0_rel[nonzero] = data0[nonzero] / flat1[nonzero]
data10_rel =  np.zeros(np.shape(flat1), dtype = float)
data10_rel[nonzero] = data10[nonzero] / flat1[nonzero]

plt.figure()
plt.imshow(data0_rel)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(-np.log(data0_rel))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(data10_rel)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(-np.log(data10_rel))
plt.colorbar()
plt.show()

# Set up for loading all summed images at 250 angles.
pathname = '/media/algol/HD-LXU3/DATA_DANIIL/PSI_DATA/DATA/Sample/angle{}/'
filename = 'IMAT0000{}_Tomo_test_000_SummedImg.fits'

# Dimensions
num_angles = 250
imsize = 512

# Initialise array
data = np.zeros((num_angles,imsize,imsize))

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

data_rel = -np.log(data)

# Permute order to get: angles, vertical, horizontal, as default in framework.
data_rel = np.transpose(data_rel,(0,2,1))

# Set angles to use
angles = np.linspace(-np.pi,np.pi,num_angles,endpoint=False)

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
x_init = ImageData(np.zeros((imsize,imsize)),
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


f = Norm2sq(Aop,b2d,c=0.5)

opt = {'tol': 1e-4, 'iter': 50}

lamtv = 1.0
# Repeat for FGP variant.
g_fgp = FGP_TV(lambdaReg = lamtv,
                 iterationsTV=50,
                 tolerance=1e-5,
                 methodTV=0,
                 nonnegativity=0,
                 printing=0,
                 device='cpu')

x_fista_fgp, it1, timing1, criter_fgp = FISTA(x_init, f, g_fgp,opt)

plt.figure()
plt.subplot(121)
plt.imshow(x_fista_fgp.array)
plt.title('FISTA FGP TV')
plt.subplot(122)
plt.semilogy(criter_fgp)
plt.show()
