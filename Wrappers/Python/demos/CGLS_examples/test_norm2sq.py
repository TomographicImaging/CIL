# This demo illustrates how ASTRA 2D projectors can be used with
# the modular optimisation framework. The demo sets up a 2D test case and 
# demonstrates reconstruction using CGLS, as well as FISTA for least squares 
# and 1-norm regularisation.

# First make all imports
from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algorithms import FISTA, CGLS
from ccpi.optimisation.functions import Norm2Sq, L1Norm, FunctionOperatorComposition, L2NormSquared
from ccpi.astra.operators import AstraProjectorSimple

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
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
b = Aop.direct(Phantom)
z = Aop.adjoint(b)

plt.imshow(b.array)
plt.title('Simulated data')
plt.show()

plt.imshow(z.array)
plt.title('Backprojected data')
plt.colorbar()
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(geometry=ig)
opt = {'tol': 1e-4, 'iter': 5000}

# First a CGLS reconstruction can be done:
CGLS_alg = CGLS(x_init=x_init, operator=Aop, data=b , tolerance = 1e-9)
CGLS_alg.max_iteration = 2000
CGLS_alg.run(opt['iter'])
#
x_CGLS = CGLS_alg.get_output()

#%%
#
plt.figure()
plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.show()
#
plt.figure()
plt.semilogy(CGLS_alg.objective)
plt.title('CGLS criterion')
plt.show()

# CGLS solves the simple least-squares problem. The same problem can be solved 
# by FISTA by setting up explicitly a least squares function object and using 
# no regularisation:

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
#f = Norm2Sq(Aop,b,c=0.5)
f = FunctionOperatorComposition(L2NormSquared(b=b), Aop)

# Run FISTA for least squares without constraints
FISTA_alg = FISTA(x_init=x_init, f=f, opt=opt)
FISTA_alg.max_iteration = 5000
FISTA_alg.update_objective_interval = 500
FISTA_alg.run(opt['iter'], verbose=True)
x_FISTA = FISTA_alg.get_output()

plt.figure()
plt.imshow(x_FISTA.array)
plt.title('FISTA Least squares reconstruction')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg.objective)
plt.title('FISTA Least squares criterion')
plt.show()


# FISTA can also solve regularised forms by specifying a second function object
# such as 1-norm regularisation with choice of regularisation parameter lam:

# Create 1-norm function object
lam = 1.0
g0 = lam * L1Norm()

# Run FISTA for least squares plus 1-norm function.
FISTA_alg1 = FISTA()
FISTA_alg1.set_up(x_init=x_init, f=f, g=g0, opt=opt)
FISTA_alg1.max_iteration = 2000
FISTA_alg1.update_objective_interval = 500
FISTA_alg1.run(opt['iter'])
x_FISTA1 = FISTA_alg1.get_output()

plt.figure()
plt.imshow(x_FISTA1.array)
plt.title('FISTA LS+L1Norm reconstruction')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg1.objective)
plt.title('FISTA LS+L1norm criterion')
plt.show()


# Compare all reconstruction and criteria
clims = (0,1)
cols = 2
rows = 2
current = 1

fig = plt.figure()
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(Phantom.as_array())))
imgplot = plt.imshow(Phantom.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS')
imgplot = plt.imshow(x_FISTA.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS+1')
imgplot = plt.imshow(x_FISTA1.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

fig = plt.figure()
a=fig.add_subplot(1,1,1)
a.set_title('criteria')
imgplot = plt.loglog(CGLS_alg.objective, label='CGLS')
imgplot = plt.loglog(FISTA_alg.objective , label='FISTA LS')
imgplot = plt.loglog(FISTA_alg1.objective , label='FISTA LS+1')
a.legend(loc='lower left')
plt.show()