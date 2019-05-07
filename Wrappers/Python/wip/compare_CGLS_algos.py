# This demo illustrates how to use the SIRT algorithm without and with 
# nonnegativity and box constraints. The ASTRA 2D projectors are used.

# First make all imports
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.algs import FISTA, FBPD, CGLS, SIRT
from ccpi.astra.operators import AstraProjectorSimple

from ccpi.optimisation.algorithms import CGLS as CGLSalg

import numpy as np
import matplotlib.pyplot as plt

# Choose either a parallel-beam (1=parallel2D) or fan-beam (2=cone2D) test case
test_case = 1

# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 128
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

#plt.figure()
#plt.imshow(x)
#plt.title('Phantom image')
#plt.show()

# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam and 0 to 2pi for fanbeam, set the width of a detector 
# pixel relative to an object pixel, the number of detector pixels, and the 
# source-origin and origin-detector distance (here the origin-detector distance 
# set to 0 to simulate a "virtual detector" with same detector pixel size as
# object pixel size).
angles_num = 20
det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

if test_case==1:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num,det_w)
elif test_case==2:
    angles = np.linspace(0,2*np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             det_num,
                             det_w,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)
else:
    NotImplemented

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
b = Aop.direct(Phantom)
z = Aop.adjoint(b)

#plt.figure()
#plt.imshow(b.array)
#plt.title('Simulated data')
#plt.show()

#plt.figure()
#plt.imshow(z.array)
#plt.title('Backprojected data')
#plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 7}

# First a CGLS reconstruction using the function version of CGLS can be done:
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b, opt)

#plt.figure()
#plt.imshow(x_CGLS.array)
#plt.title('CGLS')
#plt.colorbar()
#plt.show()

#plt.figure()
#plt.semilogy(criter_CGLS)
#plt.title('CGLS criterion')
#plt.show()



# Now CLGS using the algorithm class
CGLS_alg = CGLSalg()
CGLS_alg.set_up(x_init, Aop, b )
CGLS_alg.max_iteration = 2000
CGLS_alg.run(opt['iter'])
x_CGLS_alg = CGLS_alg.get_output()

#plt.figure()
#plt.imshow(x_CGLS_alg.as_array())
#plt.title('CGLS ALG')
#plt.colorbar()
#plt.show()

#plt.figure()
#plt.semilogy(CGLS_alg.objective)
#plt.title('CGLS criterion')
#plt.show()

print(criter_CGLS)
print(CGLS_alg.objective)

print((x_CGLS - x_CGLS_alg).norm())