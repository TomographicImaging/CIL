#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer

import numpy as np                           
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, FunctionOperatorComposition, BlockFunction, ScaledFunction

from skimage.util import random_noise

from timeit import default_timer as timer
def dt(steps):
    return steps[-1] - steps[-2]

#%%

# Create phantom for TV denoising

N = 200

data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data, mode = 'gaussian', mean=0, var = 0.05, seed=10)
noisy_data = ImageData(n1)

#plt.imshow(noisy_data.as_array())
#plt.show()

#%%

# Regularisation Parameter
alpha = 2

#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
method = '0'

if method == '0':

    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)

    # Form Composite Operator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    #### Create functions
      
    f1 = alpha * MixedL21Norm()
    f2 = 0.5 * L2NormSquared(b = noisy_data)    
    f = BlockFunction(f1, f2)  
                                      
    g = ZeroFunction()
    
else:
    
    ###########################################################################
    #         No Composite #
    ###########################################################################
    operator = Gradient(ig)
    f = alpha * FunctionOperatorComposition(operator, MixedL21Norm())
    g = L2NormSquared(b = noisy_data)
    
    ###########################################################################
#%%
    
# Compute operator Norm
normK = operator.norm()
print ("normK", normK)

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

opt = {'niter':2000}
opt1 = {'niter':2000, 'memopt': True}

#t1 = timer()
#res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
#print(timer()-t1)
#
#print("with memopt \n")
#
#t2 = timer()
#res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) 
#print(timer()-t2)

pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 100


pdhgo = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhgo.max_iteration = 2000
pdhgo.update_objective_interval = 100

steps = [timer()]
pdhgo.run(2000)
steps.append(timer())
t1 = dt(steps)

pdhg.run(2000)
steps.append(timer())
t2 = dt(steps)

print ("Time difference {}s {}s {}s Speedup {:.2f}".format(t1,t2,t2-t1, t2/t1))
res = pdhg.get_output()
res1 = pdhgo.get_output()

diff = (res-res1)
print ("diff norm {} max {}".format(diff.norm(), diff.abs().as_array().max()))
print ("Sum ( abs(diff) )  {}".format(diff.abs().sum()))


plt.figure(figsize=(5,5))
plt.subplot(1,3,1)
plt.imshow(res.as_array())
plt.colorbar()
#plt.show()
 
#plt.figure(figsize=(5,5))
plt.subplot(1,3,2)
plt.imshow(res1.as_array())
plt.colorbar()
#plt.show()



#=======
## opt = {'niter':2000, 'memopt': True}
#
## res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
# 
#>>>>>>> origin/pdhg_fix
#
#
## opt = {'niter':2000, 'memopt': False}
## res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
#
## plt.figure(figsize=(5,5))
## plt.subplot(1,3,1)
## plt.imshow(res.as_array())
## plt.title('memopt')
## plt.colorbar()
## plt.subplot(1,3,2)
## plt.imshow(res1.as_array())
## plt.title('no memopt')
## plt.colorbar()
## plt.subplot(1,3,3)
## plt.imshow((res1 - res).abs().as_array())
## plt.title('diff')
## plt.colorbar()
## plt.show()
#pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
#pdhg.max_iteration = 2000
#pdhg.update_objective_interval = 100
#
#
#pdhgo = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
#pdhgo.max_iteration = 2000
#pdhgo.update_objective_interval = 100
#
#steps = [timer()]
#pdhgo.run(200)
#steps.append(timer())
#t1 = dt(steps)
#
#pdhg.run(200)
#steps.append(timer())
#t2 = dt(steps)
#
#print ("Time difference {} {} {}".format(t1,t2,t2-t1))
#sol = pdhg.get_output().as_array()
##sol = result.as_array()
##
#fig = plt.figure()
#plt.subplot(1,3,1)
#plt.imshow(noisy_data.as_array())
#plt.colorbar()
#plt.subplot(1,3,2)
#plt.imshow(sol)
#plt.colorbar()
#plt.subplot(1,3,3)
#plt.imshow(pdhgo.get_output().as_array())
#plt.colorbar()
#
#plt.show()
###
##
####
##plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
##plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
##plt.legend()
##plt.show()
#
#
##%%
##
