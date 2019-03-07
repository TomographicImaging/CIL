
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, DataContainer
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, ZeroFun, Norm1, TV2D, Norm2

from ccpi.optimisation.ops import LinearOperatorMatrix, TomoIdentity
from ccpi.optimisation.ops import Identity
from ccpi.optimisation.ops import FiniteDiff2D

# Requires CVXPY, see http://www.cvxpy.org/
# CVXPY can be installed in anaconda using
# conda install -c cvxgrp cvxpy libgcc

# Whether to use or omit CVXPY
use_cvxpy = True
if use_cvxpy:
    from cvxpy import *

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
opt = {'memopt':True}
# Create object instances with the test data A and b.
f = Norm2sq(A,b,c=0.5, memopt=True)
g0 = ZeroFun()

# Initial guess
x_init = DataContainer(np.zeros((n,1)))

f.grad(x_init)

# Run FISTA for least squares plus zero function.
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, g0 , opt=opt)

# Print solution and final objective/criterion value for comparison
print("FISTA least squares plus zero function solution and objective value:")
print(x_fista0.array)
print(criter0[-1])

if use_cvxpy:
    # Compare to CVXPY
    
    # Construct the problem.
    x0 = Variable(n)
    objective0 = Minimize(0.5*sum_squares(Amat*x0 - bmat.T[0]) )
    prob0 = Problem(objective0)
    
    # The optimal objective is returned by prob.solve().
    result0 = prob0.solve(verbose=False,solver=SCS,eps=1e-9)
    
    # The optimal solution for x is stored in x.value and optimal objective value 
    # is in result as well as in objective.value
    print("CVXPY least squares plus zero function solution and objective value:")
    print(x0.value)
    print(objective0.value)

# Plot criterion curve to see FISTA converge to same value as CVX.
iternum = np.arange(1,1001)
plt.figure()
plt.loglog(iternum[[0,-1]],[objective0.value, objective0.value], label='CVX LS')
plt.loglog(iternum,criter0,label='FISTA LS')
plt.legend()
plt.show()

# Create 1-norm object instance
g1 = Norm1(lam)

g1(x_init)
x_rand = DataContainer(np.reshape(np.random.rand(n),(n,1)))
x_rand2 = DataContainer(np.reshape(np.random.rand(n-1),(n-1,1)))
v = g1.prox(x_rand,0.02)
#vv = g1.prox(x_rand2,0.02)
vv = v.copy() 
vv *= 0
print (">>>>>>>>>>vv" , vv.as_array())
vv.fill(v)
print (">>>>>>>>>>fill" , vv.as_array())
g1.proximal(x_rand, 0.02, out=vv)
print (">>>>>>>>>>v" , v.as_array())
print (">>>>>>>>>>gradient" , vv.as_array())

print (">>>>>>>>>>" , (v-vv).as_array())
import sys
#sys.exit(0)
# Combine with least squares and solve using generic FISTA implementation
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1,opt=opt)

# Print for comparison
print("FISTA least squares plus 1-norm solution and objective value:")
print(x_fista1)
print(criter1[-1])

if use_cvxpy:
    # Compare to CVXPY
    
    # Construct the problem.
    x1 = Variable(n)
    objective1 = Minimize(0.5*sum_squares(Amat*x1 - bmat.T[0]) + lam*norm(x1,1) )
    prob1 = Problem(objective1)
    
    # The optimal objective is returned by prob.solve().
    result1 = prob1.solve(verbose=False,solver=SCS,eps=1e-9)
    
    # The optimal solution for x is stored in x.value and optimal objective value 
    # is in result as well as in objective.value
    print("CVXPY least squares plus 1-norm solution and objective value:")
    print(x1.value)
    print(objective1.value)
    
# Now try another algorithm FBPD for same problem:
#x_fbpd1, itfbpd1, timingfbpd1, criterfbpd1 = FBPD(x_init,Identity(), None, f, g1)
#print(x_fbpd1)
#print(criterfbpd1[-1])

# Plot criterion curve to see both FISTA and FBPD converge to same value.
# Note that FISTA is very efficient for 1-norm minimization so it beats
# FBPD in this test by a lot. But FBPD can handle a larger class of problems 
# than FISTA can.
plt.figure()
plt.loglog(iternum[[0,-1]],[objective1.value, objective1.value], label='CVX LS+1')
plt.loglog(iternum,criter1,label='FISTA LS+1')
plt.legend()
plt.show()

#plt.figure()
#plt.loglog(iternum[[0,-1]],[objective1.value, objective1.value], label='CVX LS+1')
#plt.loglog(iternum,criter1,label='FISTA LS+1')
#plt.loglog(iternum,criterfbpd1,label='FBPD LS+1')
#plt.legend()
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
I = TomoIdentity(ig)

# Data and add noise
y = I.direct(Phantom)
y.array = y.array + 0.1*np.random.randn(N, N)

plt.imshow(y.array)
plt.title('Noisy image')
plt.show()


###################
# Data fidelity term
f_denoise = Norm2sq(I,y,c=0.5,memopt=True)

# 1-norm regulariser
lam1_denoise = 1.0
g1_denoise = Norm1(lam1_denoise)

# Initial guess
x_init_denoise = ImageData(np.zeros((N,N)))

# Combine with least squares and solve using generic FISTA implementation
x_fista1_denoise, it1_denoise, timing1_denoise, criter1_denoise = FISTA(x_init_denoise, f_denoise, g1_denoise, opt=opt)

print(x_fista1_denoise)
print(criter1_denoise[-1])

#plt.imshow(x_fista1_denoise.as_array())
#plt.title('FISTA LS+1')
#plt.show()

# Now denoise LS + 1-norm with FBPD
x_fbpd1_denoise, itfbpd1_denoise, timingfbpd1_denoise, \
  criterfbpd1_denoise = FBPD(x_init_denoise, I, None, f_denoise, g1_denoise)
print(x_fbpd1_denoise)
print(criterfbpd1_denoise[-1])

#plt.imshow(x_fbpd1_denoise.as_array())
#plt.title('FBPD LS+1')
#plt.show()

if use_cvxpy:
    # Compare to CVXPY
    
    # Construct the problem.
    x1_denoise = Variable(N**2,1)
    objective1_denoise = Minimize(0.5*sum_squares(x1_denoise - y.array.flatten()) + lam1_denoise*norm(x1_denoise,1) )
    prob1_denoise = Problem(objective1_denoise)
    
    # The optimal objective is returned by prob.solve().
    result1_denoise = prob1_denoise.solve(verbose=False,solver=SCS,eps=1e-12)
    
    # The optimal solution for x is stored in x.value and optimal objective value 
    # is in result as well as in objective.value
    print("CVXPY least squares plus 1-norm solution and objective value:")
    print(x1_denoise.value)
    print(objective1_denoise.value)

x1_cvx = x1_denoise.value
x1_cvx.shape = (N,N)



#plt.imshow(x1_cvx)
#plt.title('CVX LS+1')
#plt.show()

fig = plt.figure()
plt.subplot(1,4,1)
plt.imshow(y.array)
plt.title("LS+1")
plt.subplot(1,4,2)
plt.imshow(x_fista1_denoise.as_array())
plt.title("fista")
plt.subplot(1,4,3)
plt.imshow(x_fbpd1_denoise.as_array())
plt.title("fbpd")
plt.subplot(1,4,4)
plt.imshow(x1_cvx)
plt.title("cvx")
plt.show()

#%%

##############################################################
# Now TV with FBPD and Norm2
lam_tv = 0.1
gtv = TV2D(lam_tv)
norm2 = Norm2(lam_tv)
op = FiniteDiff2D()
#gtv(gtv.op.direct(x_init_denoise))

opt_tv = {'tol': 1e-4, 'iter': 10000}

x_fbpdtv_denoise, itfbpdtv_denoise, timingfbpdtv_denoise, \
 criterfbpdtv_denoise = FBPD(x_init_denoise, op, None, \
                             f_denoise, norm2 ,opt=opt_tv)
print(x_fbpdtv_denoise)
print(criterfbpdtv_denoise[-1])

plt.imshow(x_fbpdtv_denoise.as_array())
plt.title('FBPD TV')
#plt.show()

if use_cvxpy:
    # Compare to CVXPY
    
    # Construct the problem.
    xtv_denoise = Variable((N,N))
    #print (xtv_denoise.value.shape)
    objectivetv_denoise = Minimize(0.5*sum_squares(xtv_denoise - y.array) + lam_tv*tv(xtv_denoise) )
    probtv_denoise = Problem(objectivetv_denoise)
    
    # The optimal objective is returned by prob.solve().
    resulttv_denoise = probtv_denoise.solve(verbose=False,solver=SCS,eps=1e-12)
    
    # The optimal solution for x is stored in x.value and optimal objective value 
    # is in result as well as in objective.value
    print("CVXPY least squares plus 1-norm solution and objective value:")
    print(xtv_denoise.value)
    print(objectivetv_denoise.value)
    
plt.imshow(xtv_denoise.value)
plt.title('CVX TV')
#plt.show()

fig = plt.figure()
plt.subplot(1,3,1)
plt.imshow(y.array)
plt.title("TV2D")
plt.subplot(1,3,2)
plt.imshow(x_fbpdtv_denoise.as_array())
plt.title("fbpd tv denoise")
plt.subplot(1,3,3)
plt.imshow(xtv_denoise.value)
plt.title("CVX tv")
plt.show()



plt.loglog([0,opt_tv['iter']], [objectivetv_denoise.value,objectivetv_denoise.value], label='CVX TV')
plt.loglog(criterfbpdtv_denoise, label='FBPD TV')
