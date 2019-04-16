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
                      MixedL21Norm, FunctionOperatorComposition, BlockFunction

from skimage.util import random_noise

from timeit import default_timer as timer
def dt(steps):
    return steps[-1] - steps[-2]

#%%

# Create phantom for TV denoising

import timeit
import os
from tomophantom import TomoP3D
import tomophantom

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 13 # select a model number from the library
N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
#toc=timeit.default_timer()
#Run_time = toc - tic
#print("Phantom has been built in {} seconds".format(Run_time))
#
#sliceSel = int(0.5*N_size)
##plt.gray()
#plt.figure() 
#plt.subplot(131)
#plt.imshow(phantom_tm[sliceSel,:,:],vmin=0, vmax=1)
#plt.title('3D Phantom, axial view')
#
#plt.subplot(132)
#plt.imshow(phantom_tm[:,sliceSel,:],vmin=0, vmax=1)
#plt.title('3D Phantom, coronal view')
#
#plt.subplot(133)
#plt.imshow(phantom_tm[:,:,sliceSel],vmin=0, vmax=1)
#plt.title('3D Phantom, sagittal view')
#plt.show()

#%%

N = N_size
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_num_z=N)

n1 = random_noise(phantom_tm, mode = 'gaussian', mean=0, var = 0.001, seed=10)
noisy_data = ImageData(n1)
#plt.imshow(noisy_data.as_array()[:,:,32])

#%%

# Regularisation Parameter
alpha = 0.02

#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
method = '0'

if method == '0':

    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig)

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

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

opt = {'niter':2000}
opt1 = {'niter':2000, 'memopt': True}

#t1 = timer()
#res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
#t2 = timer()


t3 = timer()
res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) 
t4 = timer()

#import cProfile
#cProfile.run('res1, time1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt1) ')
###
print ("No memopt in {}s, memopt in  {}s ".format(t2-t1, t4 -t3))
#
##    
##%%
#
#plt.figure(figsize=(10,10)) 
#plt.subplot(311)
#plt.imshow(res1.as_array()[sliceSel,:,:])
#plt.colorbar()
#plt.title('3D Phantom, axial view')
#
#plt.subplot(312)
#plt.imshow(res1.as_array()[:,sliceSel,:])
#plt.colorbar()
#plt.title('3D Phantom, coronal view')
#
#plt.subplot(313)
#plt.imshow(res1.as_array()[:,:,sliceSel])
#plt.colorbar()
#plt.title('3D Phantom, sagittal view')
#plt.show()
#
#plt.figure(figsize=(10,10)) 
#plt.subplot(311)
#plt.imshow(res.as_array()[sliceSel,:,:])
#plt.colorbar()
#plt.title('3D Phantom, axial view')
#
#plt.subplot(312)
#plt.imshow(res.as_array()[:,sliceSel,:])
#plt.colorbar()
#plt.title('3D Phantom, coronal view')
#
#plt.subplot(313)
#plt.imshow(res.as_array()[:,:,sliceSel])
#plt.colorbar()
#plt.title('3D Phantom, sagittal view')
#plt.show()
#
#diff  =  (res1 - res).abs()
#
#plt.figure(figsize=(10,10)) 
#plt.subplot(311)
#plt.imshow(diff.as_array()[sliceSel,:,:])
#plt.colorbar()
#plt.title('3D Phantom, axial view')
#
#plt.subplot(312)
#plt.imshow(diff.as_array()[:,sliceSel,:])
#plt.colorbar()
#plt.title('3D Phantom, coronal view')
#
#plt.subplot(313)
#plt.imshow(diff.as_array()[:,:,sliceSel])
#plt.colorbar()
#plt.title('3D Phantom, sagittal view')
#plt.show()
#
#
#
#
##%%
#pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
#pdhg.max_iteration = 2000
#pdhg.update_objective_interval = 100
####
#pdhgo = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
#pdhgo.max_iteration = 2000
#pdhgo.update_objective_interval = 100
####
#steps = [timer()]
#pdhgo.run(2000)
#steps.append(timer())
#t1 = dt(steps)
##
#pdhg.run(2000)
#steps.append(timer())
#t2 = dt(steps)
#
#print ("Time difference {}s {}s {}s Speedup {:.2f}".format(t1,t2,t2-t1, t2/t1))
#res = pdhg.get_output()
#res1 = pdhgo.get_output()

#%%
#plt.figure(figsize=(15,15))
#plt.subplot(3,1,1)
#plt.imshow(res.as_array())
#plt.title('no memopt')
#plt.colorbar()
#plt.subplot(3,1,2)
#plt.imshow(res1.as_array())
#plt.title('memopt')
#plt.colorbar()
#plt.subplot(3,1,3)
#plt.imshow((res1 - res).abs().as_array())
#plt.title('diff')
#plt.colorbar()
#plt.show()


#plt.figure(figsize=(15,15))
#plt.subplot(3,1,1)
#plt.imshow(pdhg.get_output().as_array())
#plt.title('no memopt class')
#plt.colorbar()
#plt.subplot(3,1,2)
#plt.imshow(res.as_array())
#plt.title('no memopt')
#plt.colorbar()
#plt.subplot(3,1,3)
#plt.imshow((pdhg.get_output() - res).abs().as_array())
#plt.title('diff')
#plt.colorbar()
#plt.show()
#    
#    
#
#plt.figure(figsize=(15,15))
#plt.subplot(3,1,1)
#plt.imshow(pdhgo.get_output().as_array())
#plt.title('memopt class')
#plt.colorbar()
#plt.subplot(3,1,2)
#plt.imshow(res1.as_array())
#plt.title('no memopt')
#plt.colorbar()
#plt.subplot(3,1,3)
#plt.imshow((pdhgo.get_output() - res1).abs().as_array())
#plt.title('diff')
#plt.colorbar()
#plt.show()
    

    

    
#    print ("Time difference {}s {}s {}s Speedup {:.2f}".format(t1,t2,t2-t1, t2/t1))
#    res = pdhg.get_output()
#    res1 = pdhgo.get_output()
#    
#    diff = (res-res1)
#    print ("diff norm {} max {}".format(diff.norm(), diff.abs().as_array().max()))
#    print ("Sum ( abs(diff) )  {}".format(diff.abs().sum()))
#    
#    
#    plt.figure(figsize=(5,5))
#    plt.subplot(1,3,1)
#    plt.imshow(res.as_array())
#    plt.colorbar()
#    #plt.show()
#     
#    #plt.figure(figsize=(5,5))
#    plt.subplot(1,3,2)
#    plt.imshow(res1.as_array())
#    plt.colorbar()
    
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
