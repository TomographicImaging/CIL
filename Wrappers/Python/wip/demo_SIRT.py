# This demo illustrates how to use the SIRT algorithm without and with 
# nonnegativity and box constraints. The ASTRA 2D projectors are used.

# First make all imports
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.functions import IndicatorBox
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.optimisation.algorithms import SIRT, CGLS

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

plt.figure()
plt.imshow(x)
plt.title('Phantom image')
plt.show()

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
Aop = AstraProjectorSimple(ig, ag, 'gpu')

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
b = Aop.direct(Phantom)
z = Aop.adjoint(b)

plt.figure()
plt.imshow(b.as_array())
plt.title('Simulated data')
plt.show()

plt.figure()
plt.imshow(z.as_array())
plt.title('Backprojected data')
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 100}


# First run a simple CGLS reconstruction:
CGLS_alg = CGLS()
CGLS_alg.set_up(x_init, Aop, b )
CGLS_alg.max_iteration = 2000
CGLS_alg.run(opt['iter'])
x_CGLS_alg = CGLS_alg.get_output()

plt.figure()
plt.imshow(x_CGLS_alg.as_array())
plt.title('CGLS ALG')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(CGLS_alg.objective)
plt.title('CGLS criterion')
plt.show()


# A SIRT reconstruction can be done simply by replacing CGLS by SIRT.
# In the first instance, no constraints are enforced.
SIRT_alg = SIRT()
SIRT_alg.set_up(x_init, Aop, b )
SIRT_alg.max_iteration = 2000
SIRT_alg.run(opt['iter'])
x_SIRT_alg = SIRT_alg.get_output()

plt.figure()
plt.imshow(x_SIRT_alg.as_array())
plt.title('SIRT unconstrained')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(SIRT_alg.objective)
plt.title('SIRT unconstrained criterion')
plt.show()

# The SIRT algorithm is stopped after the specified number of iterations has 
# been run. It can be resumed by calling the run command again, which will run 
# it for the specificed number of iterations
SIRT_alg.run(opt['iter'])
x_SIRT_alg2 = SIRT_alg.get_output()

plt.figure()
plt.imshow(x_SIRT_alg2.as_array())
plt.title('SIRT unconstrained, extra iterations')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(SIRT_alg.objective)
plt.title('SIRT unconstrained criterion, extra iterations')
plt.show()


# A SIRT nonnegativity constrained reconstruction can be done using the 
# additional input "constraint" set to a box indicator function with 0 as the 
# lower bound and the default upper bound of infinity. First setup a new 
# instance of the SIRT algorithm.
SIRT_alg0 = SIRT()
SIRT_alg0.set_up(x_init, Aop, b, constraint=IndicatorBox(lower=0) )
SIRT_alg0.max_iteration = 2000
SIRT_alg0.run(opt['iter'])
x_SIRT_alg0 = SIRT_alg0.get_output()

plt.figure()
plt.imshow(x_SIRT_alg0.as_array())
plt.title('SIRT nonnegativity constrained')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(SIRT_alg0.objective)
plt.title('SIRT nonnegativity criterion')
plt.show()


# A SIRT reconstruction with box constraints on [0,1] can also be done. 
SIRT_alg01 = SIRT()
SIRT_alg01.set_up(x_init, Aop, b, constraint=IndicatorBox(lower=0,upper=1) )
SIRT_alg01.max_iteration = 2000
SIRT_alg01.run(opt['iter'])
x_SIRT_alg01 = SIRT_alg01.get_output()

plt.figure()
plt.imshow(x_SIRT_alg01.as_array())
plt.title('SIRT boc(0,1)')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(SIRT_alg01.objective)
plt.title('SIRT box(0,1) criterion')
plt.show()

# The test image has values in the range [0,1], so enforcing values in the 
# reconstruction to be within this interval improves a lot. Just for fun
# we can also easily see what happens if we choose a narrower interval as 
# constrint in the reconstruction, lower bound 0.2, upper bound 0.8. 
SIRT_alg0208 = SIRT()
SIRT_alg0208.set_up(x_init,Aop,b,constraint=IndicatorBox(lower=0.2,upper=0.8))
SIRT_alg0208.max_iteration = 2000
SIRT_alg0208.run(opt['iter'])
x_SIRT_alg0208 = SIRT_alg0208.get_output()

plt.figure()
plt.imshow(x_SIRT_alg0208.as_array())
plt.title('SIRT boc(0.2,0.8)')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(SIRT_alg0208.objective)
plt.title('SIRT box(0.2,0.8) criterion')
plt.show()