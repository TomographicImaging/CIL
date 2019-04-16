# This demo illustrates how to use the SIRT algorithm without and with 
# nonnegativity and box constraints. The ASTRA 2D projectors are used.

# First make all imports
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.algs import FISTA, FBPD, CGLS, SIRT
from ccpi.optimisation.funcs import Norm2sq, Norm1, IndicatorBox
from ccpi.astra.ops import AstraProjectorSimple

from ccpi.optimisation.algorithms import CGLS as CGLSALG
from ccpi.optimisation.algorithms import SIRT as SIRTALG

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
plt.imshow(b.array)
plt.title('Simulated data')
plt.show()

plt.figure()
plt.imshow(z.array)
plt.title('Backprojected data')
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 100}

# First a CGLS reconstruction can be done:
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b, opt)

plt.figure()
plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()


my_CGLS_alg = CGLSALG()
my_CGLS_alg.set_up(x_init, Aop, b )
my_CGLS_alg.max_iteration = 2000
my_CGLS_alg.run(opt['iter'])
x_CGLS_alg = my_CGLS_alg.get_output()

plt.figure()
plt.imshow(x_CGLS_alg.array)
plt.title('CGLS ALG')
plt.colorbar()
plt.show()


# A SIRT unconstrained reconstruction can be done: similarly:
x_SIRT, it_SIRT, timing_SIRT, criter_SIRT = SIRT(x_init, Aop, b, opt)

plt.figure()
plt.imshow(x_SIRT.array)
plt.title('SIRT unconstrained')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(criter_SIRT)
plt.title('SIRT unconstrained criterion')
plt.show()



my_SIRT_alg = SIRTALG()
my_SIRT_alg.set_up(x_init, Aop, b )
my_SIRT_alg.max_iteration = 2000
my_SIRT_alg.run(opt['iter'])
x_SIRT_alg = my_SIRT_alg.get_output()

plt.figure()
plt.imshow(x_SIRT_alg.array)
plt.title('SIRT ALG')
plt.colorbar()
plt.show()

# A SIRT nonnegativity constrained reconstruction can be done using the 
# additional input "constraint" set to a box indicator function with 0 as the 
# lower bound and the default upper bound of infinity:
x_SIRT0, it_SIRT0, timing_SIRT0, criter_SIRT0 = SIRT(x_init, Aop, b, opt,
                                                      constraint=IndicatorBox(lower=0))
plt.figure()
plt.imshow(x_SIRT0.array)
plt.title('SIRT nonneg')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(criter_SIRT0)
plt.title('SIRT nonneg criterion')
plt.show()


my_SIRT_alg0 = SIRTALG()
my_SIRT_alg0.set_up(x_init, Aop, b, constraint=IndicatorBox(lower=0) )
my_SIRT_alg0.max_iteration = 2000
my_SIRT_alg0.run(opt['iter'])
x_SIRT_alg0 = my_SIRT_alg0.get_output()

plt.figure()
plt.imshow(x_SIRT_alg0.array)
plt.title('SIRT ALG0')
plt.colorbar()
plt.show()


# A SIRT reconstruction with box constraints on [0,1] can also be done:
x_SIRT01, it_SIRT01, timing_SIRT01, criter_SIRT01 = SIRT(x_init, Aop, b, opt,
         constraint=IndicatorBox(lower=0,upper=1))

plt.figure()
plt.imshow(x_SIRT01.array)
plt.title('SIRT box(0,1)')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(criter_SIRT01)
plt.title('SIRT box(0,1) criterion')
plt.show()

my_SIRT_alg01 = SIRTALG()
my_SIRT_alg01.set_up(x_init, Aop, b, constraint=IndicatorBox(lower=0,upper=1) )
my_SIRT_alg01.max_iteration = 2000
my_SIRT_alg01.run(opt['iter'])
x_SIRT_alg01 = my_SIRT_alg01.get_output()

plt.figure()
plt.imshow(x_SIRT_alg01.array)
plt.title('SIRT ALG01')
plt.colorbar()
plt.show()

# The indicator function can also be used with the FISTA algorithm to do 
# least squares with nonnegativity constraint.

'''
# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = Norm2sq(Aop,b,c=0.5)
# Run FISTA for least squares without constraints
x_fista, it, timing, criter = FISTA(x_init, f, None,opt)
plt.figure()
plt.imshow(x_fista.array)
plt.title('FISTA Least squares')
plt.show()
plt.figure()
plt.semilogy(criter)
plt.title('FISTA Least squares criterion')
plt.show()
# Run FISTA for least squares with nonnegativity constraint
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, IndicatorBox(lower=0),opt)
plt.figure()
plt.imshow(x_fista0.array)
plt.title('FISTA Least squares nonneg')
plt.show()
plt.figure()
plt.semilogy(criter0)
plt.title('FISTA Least squares nonneg criterion')
plt.show()
# Run FISTA for least squares with box constraint [0,1]
x_fista01, it01, timing01, criter01 = FISTA(x_init, f, IndicatorBox(lower=0,upper=1),opt)
plt.figure()
plt.imshow(x_fista01.array)
plt.title('FISTA Least squares box(0,1)')
plt.show()
plt.figure()
plt.semilogy(criter01)
plt.title('FISTA Least squares box(0,1) criterion')
plt.show()
'''