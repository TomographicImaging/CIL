
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

# Set up phantom
N = 128

x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 1.0
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 2.0

plt.imshow(x)
plt.show()

vg = VolumeGeometry(N,N,None, 1,1,None)

Phantom = VolumeData(x,geometry=vg)

# Set up measurement geometry
angles_num = 20; # angles number
angles = np.linspace(0,np.pi,angles_num,endpoint=False)
det_w = 1.0
det_num = N
SourceOrig = 500
OrigDetec = 0

# Parallelbeam geometry test
pg = SinogramGeometry('parallel','2D',angles,det_num,det_w)

# ASTRA operator using volume and sinogram geometries
Aop = AstraProjectorSimple(vg, pg, 'gpu')

# Unused old astra projector without geometry
# Aop_old = AstraProjector(det_w, det_num, SourceOrig, 
#                      OrigDetec, angles, 
#                      N,'fanbeam','gpu') 

# Try forward and backprojection
b = Aop.direct(Phantom)
out2 = Aop.adjoint(b)

plt.imshow(b.array)
plt.show()

plt.imshow(out2.array)
plt.show()

# Create least squares object instance with projector and data.
f = Norm2sq(Aop,b,c=0.5)

# Initial guess
x_init = VolumeData(np.zeros(x.shape),geometry=vg)

# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None)

plt.imshow(x_fista0.array)
plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0)

plt.imshow(x_fista1.array)
plt.show()

# Delete projector
#Aop.delete()
