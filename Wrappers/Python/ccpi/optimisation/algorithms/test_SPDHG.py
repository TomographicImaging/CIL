#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:36:48 2020

@author: cd902
"""

# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%%
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData, BlockDataContainer
import numpy as np 
import numpy                          
import matplotlib.pyplot as plt
from ccpi.optimisation.algorithms import PDHG, SPDHG
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, IndicatorBox, TotalVariation
                      
from ccpi.astra.operators import AstraProjectorSimple
import os, sys
from ccpi.framework import TestData
from ccpi.utilities.display import plotter2D
# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 1
    
    
#
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
ig = data.geometry
ig.voxel_size_x = 0.1
ig.voxel_size_y = 0.1
    
#import os, sys
#import tomophantom
#from tomophantom import TomoP2D
## user supplied input
#if len(sys.argv) > 1:
#    which_noise = int(sys.argv[1])
#else:
#    which_noise = 1
#    
#model = 1 # select a model number from the library
#N = 128 # set dimension of the phantom
#path = os.path.dirname(tomophantom.__file__)
#path_library2D = os.path.join(path, "Phantom2DLibrary.dat")    
#phantom_2D = TomoP2D.Model(model, N, path_library2D)    
#data = ImageData(phantom_2D)
#ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_size_x = 0.1, voxel_size_y = 0.1)
# Create acquisition data and geometry
detectors = ig.shape[0]
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
# Select device
device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)
# Create noisy data. Apply Gaussian noise
noises = ['gaussian', 'poisson']
noise = noises[which_noise]
if noise == 'poisson':
    np.random.seed(10)
    scale = 5
    eta = 0
    noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
elif noise == 'gaussian':
    np.random.seed(10)
    n1 = np.random.normal(0, 0.1, size = ag.shape)
    noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
    
else:
    raise ValueError('Unsupported Noise ', noise)
# Show Ground Truth and Noisy Data
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,2)
# plt.imshow(data.as_array())
# plt.title('Ground Truth')
# plt.colorbar()
# plt.subplot(1,2,1)
# plt.imshow(noisy_data.as_array())
# plt.title('Noisy Data')
# plt.colorbar()
# plt.show()
#%% 'explicit' SPDHG, scalar step-sizes
subsets = 10
size_of_subsets = int(len(angles)/subsets)
# create Gradient operator
op1 = Gradient(ig)
# take angles and create uniform subsets in uniform+sequential setting
list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
# create acquisitioin geometries for each the interval of splitting angles
list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
for i in range(len(list_angles))]
# create with operators as many as the subsets
A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)] + [op1])
## number of subsets
#(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
#
## acquisisiton data
g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:]) for i in range(0, len(angles), size_of_subsets)])
alpha = 0.5
## block function
F = BlockFunction(*[*[KullbackLeibler(b=g[i]) for i in range(subsets)] + [alpha * MixedL21Norm()]]) 
G = IndicatorBox(lower=0)

prob = [1/(2*subsets)]*(len(A)-1) + [1/2]
spdhg = SPDHG(f=F,g=G,operator=A, 
              max_iteration = 1000,
              update_objective_interval=200, prob = prob)
spdhg.run(1000, very_verbose = True)


#%% with different probability choice
#prob = [1/len(A)]*(len(A))
#spdhg = SPDHG(f=F,g=G,operator=A, 
#              max_iteration = 1000,
#              update_objective_interval=200, prob = prob)
#spdhg.run(1000, very_verbose = True)
#plt.figure()
#plt.imshow(spdhg.get_output().as_array())
#plt.colorbar()
#plt.show()
#%% 'explicit' PDHG, scalar step-sizes
op1 = Gradient(ig)
op2 = Aop
# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 
f2 = KullbackLeibler(b=noisy_data)  
g =  IndicatorBox(lower=0)    
normK = operator.norm()
sigma = 1/normK
tau = 1/normK
    
f1 = alpha * MixedL21Norm() 
f = BlockFunction(f1, f2)   
# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000, very_verbose = True)

#%% show diff between PDHG and SPDHG
# plt.imshow(spdhg.get_output().as_array() -pdhg.get_output().as_array())
# plt.colorbar()
# plt.show()

from ccpi.utilities.quality_measures import mae, mse, psnr
qm = (mae(spdhg.get_output(), pdhg.get_output()),
    mse(spdhg.get_output(), pdhg.get_output()),
    psnr(spdhg.get_output(), pdhg.get_output())
    )
print ("Quality measures", qm)
# 0.0015075773699209094, 1.6859006791491993e-05
plotter2D([spdhg.get_output(), pdhg.get_output(), spdhg.get_output() - pdhg.get_output()], titles=['SPDHG', 'PDHG', 'diff'])

#%% 'implicit' PDHG, scalar step-sizes

# Fast Gradient Projection algorithm for Total Variation(TV)
from ccpi.optimisation.functions import TotalVariation

# Create BlockOperator
operator = Aop 
f = KullbackLeibler(b=noisy_data)        
g =  TotalVariation(alpha, 200, 1e-4, lower=0)   
normK = operator.norm()
sigma = 1/normK
tau = 1/normK
      
# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
#pdhg.run(200, very_verbose = True)

#%% 'implicit' PDHG, preconditioned step-sizes

# Fast Gradient Projection algorithm for Total Variation(TV)
from ccpi.optimisation.functions import TotalVariation

# Create BlockOperator
operator = Aop 
f = KullbackLeibler(b=noisy_data)        
g =  TotalVariation(alpha, 50, 1e-4, lower=0)   
#normK = operator.norm()
tau_tmp = 1
sigma_tmp = 1
tau = sigma_tmp / operator.adjoint(tau_tmp * operator.range_geometry().allocate(1.))
sigma = tau_tmp / operator.direct(sigma_tmp * operator.domain_geometry().allocate(1.))
x_init = operator.domain_geometry().allocate()

# for some reason these lines don't work, but pdhg does
#g.proximal(noisy_data, 1.)
#g.proximal(x_init, tau)
   
# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000, very_verbose = True)

#
#plt.imshow(pdhg.get_output().as_array())
#plt.colorbar()
#plt.show()

# %% 'implicit' SPDHG, scalar step-sizes
from ccpi.optimisation.functions import TotalVariation

subsets = 10
size_of_subsets = int(len(angles)/subsets)
# take angles and create uniform subsets in uniform+sequential setting
list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
# create acquisitioin geometries for each the interval of splitting angles
list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
                for i in range(len(list_angles))]
# create with operators as many as the subsets
A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)])
## number of subsets
#(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
#
## acquisisiton data
g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:])
                            for i in range(0, len(angles), size_of_subsets)])
alpha = 0.5
## block function
F = BlockFunction(*[KullbackLeibler(b=g[i]) for i in range(subsets)]) 
G = TotalVariation(alpha, 50, 1e-4, lower=0) 

prob = [1/len(A)]*len(A)
spdhg = SPDHG(f=F,g=G,operator=A, 
              max_iteration = 1000,
              update_objective_interval=200, prob = prob)
spdhg.run(1000, very_verbose = True)
#plt.figure()
#plt.imshow(spdhg.get_output().as_array())
#plt.colorbar()
#plt.show()
qm = (mae(spdhg.get_output(), pdhg.get_output()),
    mse(spdhg.get_output(), pdhg.get_output()),
    psnr(spdhg.get_output(), pdhg.get_output())
    )
print ("Quality measures", qm)
# 0.0028578834608197212, 3.885594196617603e-05
plotter2D([spdhg.get_output(), pdhg.get_output(), spdhg.get_output() - pdhg.get_output()], titles=['SPDHG', 'PDHG', 'diff'])


# # %% 'implicit' SPDHG,  preconditioned step-sizes
# from ccpi.optimisation.functions import TotalVariation

# subsets = 10
# size_of_subsets = int(len(angles)/subsets)
# # take angles and create uniform subsets in uniform+sequential setting
# list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
# # create acquisitioin geometries for each the interval of splitting angles
# list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
#                 for i in range(len(list_angles))]
# # create with operators as many as the subsets
# A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)])
# ## number of subsets
# #(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
# #
# ## acquisisiton data
# g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:])
#                             for i in range(0, len(angles), size_of_subsets)])
# alpha = 0.5
# ## block function
# F = BlockFunction(*[KullbackLeibler(b=g[i]) for i in range(subsets)]) 
# G = TotalVariation(alpha, 50, 1e-4, lower=0) 

# prob = [1/len(A)]*len(A)

# tau_tmp = 1
# sigma_tmp = 1
# tau = sigma_tmp / A.adjoint(tau_tmp * A.range_geometry().allocate(1.))
# sigma_tmp = tau_tmp / A.direct(sigma_tmp * A.domain_geometry().allocate(1.))


# spdhg = SPDHG(f=F,g=G,operator=A, 
#               max_iteration = 1000,
#               update_objective_interval=200, prob = prob, 
#               tau = tau, sigma = sigma)



# # %%





# import numpy as np 
# import numpy                          
# import matplotlib.pyplot as plt
# from ccpi.optimisation.algorithms import PDHG

# def PDHG_new_update(self):
#      """Modify the PDHG update to allow preconditioning"""
#      # save previous iteration
#      self.x_old.fill(self.x)
#      self.y_old.fill(self.y)
#      # Gradient ascent for the dual variable
#      self.operator.direct(self.xbar, out=self.y_tmp)
#      self.y_tmp *= self.sigma
#      self.y_tmp += self.y_old
#      self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)
#      # Gradient descent for the primal variable
#      self.operator.adjoint(self.y, out=self.x_tmp)
#      self.x_tmp *= -1*self.tau
#      self.x_tmp += self.x_old
#      self.g.proximal(self.x_tmp, self.tau, out=self.x)
#      # Update
#      self.x.subtract(self.x_old, out=self.xbar)
#      self.xbar *= self.theta    
#      self.xbar += self.x
    
# PDHG.update = PDHG_new_update


# #tau_tmp = 1
# #sigma_tmp = 1
# #tau = sigma_tmp/K.adjoint(tau_tmp*K.range_geometry().allocate(1.))
# #sigma = tau_tmp/ K.direct(sigma_tmp*K.domain_geometry().allocate(1.))



# # %% 'explicit' PDHG,  preconditioned step-sizes

# #from ccpi.optimisation.operators import  Gradient_numpy

# op1 = Gradient(ig)
# #op1.sum_abs_col()

# op2 = Aop
# # Create BlockOperator
# operator = BlockOperator(op1, op2, shape=(2,1) ) 
# f2 = KullbackLeibler(b=noisy_data)  
# g =  IndicatorBox(lower=0)    


# tau = 1. / (op2.adjoint(op2.range_geometry().allocate(1.)) + 4.)
# sigma2 = 1. / op2.direct(op2.domain_geometry().allocate(1.))
# print ("type tau", type(tau))
# sigma1 =  op1.range_geometry().allocate(2.)

# sigma = BlockDataContainer(sigma1, sigma2)

    
# f1 = alpha * MixedL21Norm() 
# f = BlockFunction(f1, f2)   
# # Setup and run the PDHG algorithm
# pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
# pdhg.max_iteration = 1000
# pdhg.update_objective_interval = 200
# pdhg.run(200, very_verbose = True)
