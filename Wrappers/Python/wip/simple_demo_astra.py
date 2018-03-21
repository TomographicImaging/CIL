#import sys
#sys.path.append("..")

from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry
from ccpi.reconstruction.algs import FISTA, FBPD, CGLS
from ccpi.reconstruction.funcs import Norm2sq, Norm1 , TV2D
from ccpi.astra.astra_ops import AstraProjectorSimple

import numpy as np
import matplotlib.pyplot as plt

test_case = 1   # 1=parallel2D, 2=cone2D

# Set up phantom
N = 128


vg = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=vg)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.show()

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
    pg = AcquisitionGeometry('parallel',
                          '2D',
                          angles,
                          det_num,det_w)
elif test_case==2:
    pg = AcquisitionGeometry('cone',
                          '2D',
                          angles,
                          det_num,
                          det_w,
                          dist_source_center=SourceOrig, 
                          dist_center_detector=OrigDetec)

# ASTRA operator using volume and sinogram geometries
Aop = AstraProjectorSimple(vg, pg, 'cpu')

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
x_init = ImageData(np.zeros(x.shape),geometry=vg)

# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None)

plt.imshow(x_fista0.array)
plt.title('FISTA0')
plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0)

plt.imshow(x_fista1.array)
plt.title('FISTA1')
plt.show()

plt.semilogy(criter1)
plt.show()

# Run FBPD=Forward Backward Primal Dual method on least squares plus 1-norm
opt = {'tol': 1e-4, 'iter': 100}
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)

plt.imshow(x_fbpd1.array)
plt.title('FBPD1')
plt.show()

plt.semilogy(criter_fbpd1)
plt.show()

# Now FBPD for least squares plus TV
#lamtv = 1
#gtv = TV2D(lamtv)

#x_fbpdtv, it_fbpdtv, timing_fbpdtv, criter_fbpdtv = FBPD(x_init,None,f,gtv,opt=opt)

#plt.imshow(x_fbpdtv.array)
#plt.show()

#plt.semilogy(criter_fbpdtv)
#plt.show()


# Run CGLS, which should agree with the FISTA0
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b, opt )

plt.imshow(x_CGLS.array)
plt.title('CGLS')
#plt.title('CGLS recon, compare FISTA0')
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()


#%%

clims = (0,1)
cols = 3
rows = 2
current = 1
fig = plt.figure()
# projections row
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(Phantom.as_array())))

imgplot = plt.imshow(Phantom.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA0')
imgplot = plt.imshow(x_fista0.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA1')
imgplot = plt.imshow(x_fista1.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD1')
imgplot = plt.imshow(x_fbpd1.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.as_array(),vmin=clims[0],vmax=clims[1])

#current = current + 1
#a=fig.add_subplot(rows,cols,current)
#a.set_title('FBPD TV')
#imgplot = plt.imshow(x_fbpdtv.as_array(),vmin=clims[0],vmax=clims[1])

fig = plt.figure()
# projections row
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter0 , label='FISTA0')
imgplot = plt.loglog(criter1 , label='FISTA1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD1')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
#imgplot = plt.loglog(criter_fbpdtv, label='FBPD TV')
b.legend(loc='right')
plt.show()
#%%