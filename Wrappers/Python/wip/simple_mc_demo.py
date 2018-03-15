#import sys
#sys.path.append("..")

from ccpi.framework import ImageData, AcquisitionData, ImageGeometry, AcquisitionGeometry
from ccpi.reconstruction.algs import FISTA
from ccpi.reconstruction.funcs import Norm2sq, Norm1
from ccpi.astra.astra_ops import AstraProjectorMC

import numpy
import matplotlib.pyplot as plt

test_case = 1   # 1=parallel2D, 2=cone2D

# Set up phantom
N = 128
M = 100
numchannels = 3

vg = ImageGeometry(voxel_num_x=N,voxel_num_y=N,channels=numchannels)
Phantom = ImageData(geometry=vg)

x = Phantom.as_array()
x[0 , round(N/4):round(3*N/4) , round(N/4):round(3*N/4)  ] = 1.0
x[0 , round(N/8):round(7*N/8) , round(3*N/8):round(5*N/8)] = 2.0

x[1 , round(N/4):round(3*N/4) , round(N/4):round(3*N/4)  ] = 0.7
x[1 , round(N/8):round(7*N/8) , round(3*N/8):round(5*N/8)] = 1.2

x[2 , round(N/4):round(3*N/4) , round(N/4):round(3*N/4)  ] = 1.5
x[2 , round(N/8):round(7*N/8) , round(3*N/8):round(5*N/8)] = 2.2

#x = numpy.zeros((N,N,1,numchannels))

#x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),:,0] = 1.0
#x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),:,0] = 2.0

#x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),:,1] = 0.7
#x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),:,1] = 1.2

#x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),:,2] = 1.5
#x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),:,2] = 2.2

f, axarr = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarr[k].imshow(x[k],vmin=0,vmax=2.5)
plt.show()

#vg = ImageGeometry(N,N,None, 1,1,None,channels=numchannels)


#Phantom = ImageData(x,geometry=vg)

# Set up measurement geometry
angles_num = 20; # angles number

if test_case==1:
    angles = numpy.linspace(0,numpy.pi,angles_num,endpoint=False)
elif test_case==2:
    angles = numpy.linspace(0,2*numpy.pi,angles_num,endpoint=False)
else:
    NotImplemented

det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

# Parallelbeam geometry test
if test_case==1:
    pg = AcquisitionGeometry('parallel',
                          '2D',
                          angles,
                          det_num,
                          det_w,
                          channels=numchannels)
elif test_case==2:
    pg = AcquisitionGeometry('cone',
                          '2D',
                          angles,
                          det_num,
                          det_w,
                          dist_source_center=SourceOrig, 
                          dist_center_detector=OrigDetec,
                          channels=numchannels)

# ASTRA operator using volume and sinogram geometries
Aop = AstraProjectorMC(vg, pg, 'cpu')


# Try forward and backprojection
b = Aop.direct(Phantom)

fb, axarrb = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrb[k].imshow(b.as_array()[k],vmin=0,vmax=250)
plt.show()

out2 = Aop.adjoint(b)

fo, axarro = plt.subplots(1,numchannels)
for k in range(3):
    axarro[k].imshow(out2.as_array()[k],vmin=0,vmax=3500)
plt.show()

# Create least squares object instance with projector and data.
f = Norm2sq(Aop,b,c=0.5)

# Initial guess
x_init = ImageData(numpy.zeros(x.shape),geometry=vg)

# FISTA options
opt = {'tol': 1e-4, 'iter': 200}

# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt)


ff0, axarrf0 = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrf0[k].imshow(x_fista0.as_array()[k],vmin=0,vmax=2.5)
plt.show()

plt.semilogy(criter0)
plt.title('Criterion vs iterations, least squares')
plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)


# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)

ff1, axarrf1 = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrf1[k].imshow(x_fista1.as_array()[k],vmin=0,vmax=2.5)
plt.show()

plt.semilogy(criter1)
plt.title('Criterion vs iterations, least squares plus 1-norm regu')
plt.show()