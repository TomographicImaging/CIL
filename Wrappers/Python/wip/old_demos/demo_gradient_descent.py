
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

import numpy as np
import matplotlib.pyplot as plt

class Algorithm(object):
    def __init__(self, *args, **kwargs):
        pass
    def set_up(self, *args, **kwargs):
        raise NotImplementedError()
    def update(self):
        raise NotImplementedError()
    
    def should_stop(self):
        raise NotImplementedError()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.should_stop():
            raise StopIteration()
        else:
            self.update()
        
class GradientDescent(Algorithm):
    x = None
    rate = 0
    objective_function = None
    regulariser = None
    iteration = 0
    stop_cryterion = 'max_iter'
    __max_iteration = 0
    __loss = []
    def __init__(self, **kwargs):
        args = ['x_init', 'objective_function', 'rate']
        present = True
        for k,v in kwargs.items():
            if k in args:
                args.pop(args.index(k))
        if len(args) == 0:
            return self.set_up(x_init=kwargs['x_init'],
                               objective_function=kwargs['objective_function'],
                               rate=kwargs['rate'])
    
    def should_stop(self):
        return self.iteration >= self.max_iteration
    
    def set_up(self, x_init, objective_function, rate):
        self.x = x_init.copy()
        self.x_update = x_init.copy()
        self.objective_function = objective_function
        self.rate = rate
        self.__loss.append(objective_function(x_init))
        
    def update(self):
        
        self.objective_function.gradient(self.x, out=self.x_update)
        self.x_update *= -self.rate
        self.x += self.x_update
        self.__loss.append(self.objective_function(self.x))
        self.iteration += 1
        
    def get_output(self):
        return self.x
    def get_current_loss(self):
        return self.__loss[-1]
    @property
    def loss(self):
        return self.__loss
    @property
    def max_iteration(self):
        return self.__max_iteration
    @max_iteration.setter
    def max_iteration(self, value):
        assert isinstance(value, int)
        self.__max_iteration = value
        
    



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

gd = GradientDescent(x_init=x_init, objective_function=f, rate=0.001)
gd.max_iteration = 5000

for i,el in enumerate(gd):
    if i%100 == 0:
        print ("\rIteration {} Loss: {}".format(gd.iteration, 
               gd.get_current_loss()))


#%%


#
#if use_cvxpy:
#    # Compare to CVXPY
#    
#    # Construct the problem.
#    x0 = Variable(n)
#    objective0 = Minimize(0.5*sum_squares(Amat*x0 - bmat.T[0]) )
#    prob0 = Problem(objective0)
#    
#    # The optimal objective is returned by prob.solve().
#    result0 = prob0.solve(verbose=False,solver=SCS,eps=1e-9)
#    
#    # The optimal solution for x is stored in x.value and optimal objective value 
#    # is in result as well as in objective.value
#    print("CVXPY least squares plus zero function solution and objective value:")
#    print(x0.value)
#    print(objective0.value)
#
## Plot criterion curve to see FISTA converge to same value as CVX.
#iternum = np.arange(1,1001)
#plt.figure()
#plt.loglog(iternum[[0,-1]],[objective0.value, objective0.value], label='CVX LS')
#plt.loglog(iternum,criter0,label='FISTA LS')
#plt.legend()
#plt.show()
#
## Create 1-norm object instance
#g1 = Norm1(lam)
#
#g1(x_init)
#x_rand = DataContainer(np.reshape(np.random.rand(n),(n,1)))
#x_rand2 = DataContainer(np.reshape(np.random.rand(n-1),(n-1,1)))
#v = g1.prox(x_rand,0.02)
##vv = g1.prox(x_rand2,0.02)
#vv = v.copy() 
#vv *= 0
#print (">>>>>>>>>>vv" , vv.as_array())
#vv.fill(v)
#print (">>>>>>>>>>fill" , vv.as_array())
#g1.proximal(x_rand, 0.02, out=vv)
#print (">>>>>>>>>>v" , v.as_array())
#print (">>>>>>>>>>gradient" , vv.as_array())
#
#print (">>>>>>>>>>" , (v-vv).as_array())
#import sys
##sys.exit(0)
## Combine with least squares and solve using generic FISTA implementation
#x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1,opt=opt)
#
## Print for comparison
#print("FISTA least squares plus 1-norm solution and objective value:")
#print(x_fista1)
#print(criter1[-1])
#
#if use_cvxpy:
#    # Compare to CVXPY
#    
#    # Construct the problem.
#    x1 = Variable(n)
#    objective1 = Minimize(0.5*sum_squares(Amat*x1 - bmat.T[0]) + lam*norm(x1,1) )
#    prob1 = Problem(objective1)
#    
#    # The optimal objective is returned by prob.solve().
#    result1 = prob1.solve(verbose=False,solver=SCS,eps=1e-9)
#    
#    # The optimal solution for x is stored in x.value and optimal objective value 
#    # is in result as well as in objective.value
#    print("CVXPY least squares plus 1-norm solution and objective value:")
#    print(x1.value)
#    print(objective1.value)
#    
## Now try another algorithm FBPD for same problem:
#x_fbpd1, itfbpd1, timingfbpd1, criterfbpd1 = FBPD(x_init,Identity(), None, f, g1)
#print(x_fbpd1)
#print(criterfbpd1[-1])
#
## Plot criterion curve to see both FISTA and FBPD converge to same value.
## Note that FISTA is very efficient for 1-norm minimization so it beats
## FBPD in this test by a lot. But FBPD can handle a larger class of problems 
## than FISTA can.
#plt.figure()
#plt.loglog(iternum[[0,-1]],[objective1.value, objective1.value], label='CVX LS+1')
#plt.loglog(iternum,criter1,label='FISTA LS+1')
#plt.legend()
#plt.show()
#
#plt.figure()
#plt.loglog(iternum[[0,-1]],[objective1.value, objective1.value], label='CVX LS+1')
#plt.loglog(iternum,criter1,label='FISTA LS+1')
#plt.loglog(iternum,criterfbpd1,label='FBPD LS+1')
#plt.legend()
#plt.show()

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
x_fista1_denoise, it1_denoise, timing1_denoise, criter1_denoise = \
   FISTA(x_init_denoise, f_denoise, g1_denoise, opt=opt)

print(x_fista1_denoise)
print(criter1_denoise[-1])

f_2 = 
gd = GradientDescent(x_init=x_init_denoise, 
                     objective_function=f, rate=0.001)
gd.max_iteration = 5000

for i,el in enumerate(gd):
    if i%100 == 0:
        print ("\rIteration {} Loss: {}".format(gd.iteration, 
               gd.get_current_loss()))

plt.imshow(gd.get_output().as_array())
plt.title('GD image')
plt.show()
    
