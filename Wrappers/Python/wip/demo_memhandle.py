
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, DataContainer
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, ZeroFun, Norm1, TV2D

from ccpi.optimisation.ops import LinearOperatorMatrix, Identity

# Requires CVXPY, see http://www.cvxpy.org/
# CVXPY can be installed in anaconda using
# conda install -c cvxgrp cvxpy libgcc


import numpy as np
import matplotlib.pyplot as plt

# Problem data.
m = 30
n = 20
np.random.seed(1)
Amat = np.random.randn(m, n)
A = LinearOperatorMatrix(Amat)
bmat = np.random.randn(m)
bmat.shape = (bmat.shape[0],1)



# A = Identity()
# Change n to equal to m.

b = DataContainer(bmat)

# Regularization parameter
lam = 10

# Create object instances with the test data A and b.
f = Norm2sq(A,b,c=0.5, memopt=True)
g0 = ZeroFun()

# Initial guess
x_init = DataContainer(np.zeros((n,1)))

f.grad(x_init)
opt = {'memopt': True}
# Run FISTA for least squares plus zero function.
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, g0)
x_fista0_m, it0_m, timing0_m, criter0_m = FISTA(x_init, f, g0, opt=opt)

iternum = [i for i in range(len(criter0))]
# Print solution and final objective/criterion value for comparison
print("FISTA least squares plus zero function solution and objective value:")
print(x_fista0.array)
print(criter0[-1])


# Plot criterion curve to see FISTA converge to same value as CVX.
#iternum = np.arange(1,1001)
plt.figure()
plt.loglog(iternum,criter0,label='FISTA LS')
plt.loglog(iternum,criter0_m,label='FISTA LS memopt')
plt.legend()
plt.show()
#%%
# Create 1-norm object instance
g1 = Norm1(lam)

g1(x_init)
g1.prox(x_init,0.02)

# Combine with least squares and solve using generic FISTA implementation
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1)
x_fista1_m, it1_m, timing1_m, criter1_m = FISTA(x_init, f, g1, opt=opt)
iternum = [i for i in range(len(criter1))]
# Print for comparison
print("FISTA least squares plus 1-norm solution and objective value:")
print(x_fista1)
print(criter1[-1])


# Now try another algorithm FBPD for same problem:
x_fbpd1, itfbpd1, timingfbpd1, criterfbpd1 = FBPD(x_init, None, f, g1)
iternum = [i for i in range(len(criterfbpd1))]
print(x_fbpd1)
print(criterfbpd1[-1])

# Plot criterion curve to see both FISTA and FBPD converge to same value.
# Note that FISTA is very efficient for 1-norm minimization so it beats
# FBPD in this test by a lot. But FBPD can handle a larger class of problems 
# than FISTA can.
plt.figure()
plt.loglog(iternum,criter1,label='FISTA LS+1')
plt.loglog(iternum,criter1_m,label='FISTA LS+1 memopt')
plt.legend()
plt.show()

plt.figure()
plt.loglog(iternum,criter1,label='FISTA LS+1')
plt.loglog(iternum,criterfbpd1,label='FBPD LS+1')
plt.legend()
plt.show()
#%%
# Now try 1-norm and TV denoising with FBPD, first 1-norm.

# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 64
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
plt.show()

# Identity operator for denoising
I = Identity()

# Data and add noise
y = I.direct(Phantom)
y.array +=  0.1*np.random.randn(N, N)

plt.figure()
plt.imshow(y.array)
plt.title('Noisy image')
plt.show()

# Data fidelity term
f_denoise = Norm2sq(I,y,c=0.5)

# 1-norm regulariser
lam1_denoise = 1.0
g1_denoise = Norm1(lam1_denoise)

# Initial guess
x_init_denoise = ImageData(np.zeros((N,N)))

# Combine with least squares and solve using generic FISTA implementation
x_fista1_denoise, it1_denoise, timing1_denoise, \
        criter1_denoise = FISTA(x_init_denoise, f_denoise, g1_denoise)
x_fista1_denoise_m, it1_denoise_m, timing1_denoise_m, \
      criter1_denoise_m = FISTA(x_init_denoise, f_denoise, g1_denoise, opt=opt)

print(x_fista1_denoise)
print(criter1_denoise[-1])

plt.figure()
plt.imshow(x_fista1_denoise.as_array())
plt.title('FISTA LS+1')
plt.show()

plt.figure()
plt.imshow(x_fista1_denoise_m.as_array())
plt.title('FISTA LS+1 memopt')
plt.show()
#%%
# Now denoise LS + 1-norm with FBPD
x_fbpd1_denoise, itfbpd1_denoise, timingfbpd1_denoise, criterfbpd1_denoise = FBPD(x_init_denoise, None, f_denoise, g1_denoise)
print(x_fbpd1_denoise)
print(criterfbpd1_denoise[-1])

plt.figure()
plt.imshow(x_fbpd1_denoise.as_array())
plt.title('FBPD LS+1')
plt.show()


# Now TV with FBPD
lam_tv = 0.1
gtv = TV2D(lam_tv)
gtv(gtv.op.direct(x_init_denoise))

opt_tv = {'tol': 1e-4, 'iter': 10000}

x_fbpdtv_denoise, itfbpdtv_denoise, timingfbpdtv_denoise, criterfbpdtv_denoise = FBPD(x_init_denoise, None, f_denoise, gtv,opt=opt_tv)
print(x_fbpdtv_denoise)
print(criterfbpdtv_denoise[-1])

plt.imshow(x_fbpdtv_denoise.as_array())
plt.title('FBPD TV')
plt.show()
