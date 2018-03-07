

import sys

sys.path.append("..")

from ccpi.framework import *
from ccpi.reconstruction.algs import *
from ccpi.reconstruction.funcs import *
from ccpi.reconstruction.ops import *
from ccpi.reconstruction.astra_ops import *
from ccpi.reconstruction.geoms import *

import numpy as np
import matplotlib.pyplot as plt

test_case = 2   # 1=parallel2D, 2=cone2D

# Set up phantom
N = 128

numchannels = 3

x = np.zeros((N,N,1,numchannels))

x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),:,0] = 1.0
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),:,0] = 2.0

x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),:,1] = 0.7
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),:,1] = 1.2

x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),:,2] = 1.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),:,2] = 2.2

f, axarr = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarr[k].imshow(x[:,:,0,k],vmin=0,vmax=2.5)
plt.show()

vg = VolumeGeometry(N,N,None, 1,1,None,channels=numchannels)


Phantom = VolumeData(x,geometry=vg)

# Set up measurement geometry
angles_num = 20; # angles number

if test_case==1:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
elif test_case==2:
    angles = np.linspace(0,2*np.pi,angles_num,endpoint=False)
else:
    NotImplemented

det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

# Parallelbeam geometry test
if test_case==1:
    pg = SinogramGeometry('parallel',
                          '2D',
                          angles,
                          det_num,
                          det_w,
                          channels=numchannels)
elif test_case==2:
    pg = SinogramGeometry('cone',
                          '2D',
                          angles,
                          det_num,
                          det_w,
                          dist_source_center=SourceOrig, 
                          dist_center_detector=OrigDetec,
                          channels=numchannels)

# ASTRA operator using volume and sinogram geometries
Aop = AstraProjectorMC(vg, pg, 'gpu')


# Try forward and backprojection
b = Aop.direct(Phantom)

fb, axarrb = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrb[k].imshow(b.as_array()[:,:,0,k],vmin=0,vmax=250)
plt.show()

out2 = Aop.adjoint(b)

fo, axarro = plt.subplots(1,numchannels)
for k in range(3):
    axarro[k].imshow(out2.as_array()[:,:,0,k],vmin=0,vmax=3500)
plt.show()

