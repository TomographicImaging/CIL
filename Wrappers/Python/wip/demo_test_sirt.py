# This demo illustrates how ASTRA 2D projectors can be used with
# the modular optimisation framework. The demo sets up a 2D test case and 
# demonstrates reconstruction using CGLS, as well as FISTA for least squares 
# and 1-norm regularisation and FBPD for 1-norm and TV regularisation.

# First make all imports
from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D
from ccpi.astra.ops import AstraProjectorSimple

import numpy as np
import matplotlib.pyplot as plt

import time

def SIRT(x_init, operator , data , opt=None):
    '''Simultaneous Iterative Reconstruction Technique
    
    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
      opt: additional algorithm 
    '''
    
    if opt is None: 
        opt = {'tol': 1e-4, 'iter': 1000}
    else:
        try:
            max_iter = opt['iter']
        except KeyError as ke:
            opt[ke] = 1000
        try:
            opt['tol'] = 1000
        except KeyError as ke:
            opt[ke] = 1e-4
    tol = opt['tol']
    max_iter = opt['iter']
    
    #r = data.clone()
    x = x_init.clone()
    
    #d = operator.adjoint(r)
    
    #normr2 = (d**2).sum()
    
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)
    
    # Relaxation parameter must be strictly between 0 and 2.
    relax_par = 1.0
    
    # Set up scaling matrices D and M.
    im1 = ImageData(geometry=x_init.geometry)
    im1.array[:] = 1.0
    M = 1/operator.direct(im1)
    del im1
    
    aq1 = AcquisitionData(geometry=M.geometry)
    aq1.array[:] = 1.0
    D = 1/operator.adjoint(aq1)
    del aq1
    
    # algorithm loop
    for it in range(0, max_iter):
        t = time.time()
        r = b - operator.direct(x)
        
        x = x + relax_par * (D*operator.adjoint(M*r))
        x.array[x.array<0] = 0.0
        
        timing[it] = time.time() - t
        if it > 0:
            criter[it-1] = (r**2).sum()
    
    r = b - operator.direct(x)
    criter[it] = (r**2).sum()
    return x, it, timing,  criter







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

plt.imshow(b.array)
plt.title('Simulated data')
plt.show()

plt.imshow(z.array)
plt.title('Backprojected data')
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 1000}

# First a CGLS reconstruction can be done:
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b, opt)

plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.colorbar()
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()

# First a CGLS reconstruction can be done:
x_SIRT, it_SIRT, timing_SIRT, criter_SIRT = SIRT(x_init, Aop, b, opt)

plt.imshow(x_SIRT.array)
plt.title('SIRT')
plt.colorbar()
plt.show()

plt.semilogy(criter_SIRT)
plt.title('SIRT criterion')
plt.show()