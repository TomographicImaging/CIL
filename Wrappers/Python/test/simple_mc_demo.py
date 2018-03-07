

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

x = np.zeros((N,N,numchannels))

x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),0] = 1.0
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),0] = 2.0

x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),1] = 0.7
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),1] = 1.2

x[round(N/4):round(3*N/4),round(N/4):round(3*N/4),2] = 1.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8),2] = 2.2

f, axarr = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarr[k].imshow(x[:,:,k],vmin=0,vmax=2.5)
plt.show()

vg = VolumeGeometry(N,N,None, 1,1,None,channels=numchannels)


Phantom = VolumeData(x,geometry=vg)