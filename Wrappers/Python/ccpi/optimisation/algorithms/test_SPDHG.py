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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ccpi.optimisation.algorithms import Algorithm
import numpy as np

class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient
    
    Problem: 
    
    .. math::
    
      \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)
        
    :param operator: BlockOperator of Linear Operators
    :param f: BlockFunction, each function with "simple" proximal of its conjugate 
    :param g: Convex function with "simple" proximal 
    :param sigma=(sigma_i): List of Step size parameters for Dual problem
    :param tau: Step size parameter for Primal problem
    :param x_init: Initial guess ( Default x_init = 0)
    :param prob: List of probabilities
        
    Remark: Convergence is guaranted provided that [2, eq. (12)]:
        
    .. math:: 
    
      \|\sigma[i]^{1/2} * K[i] * tau^{1/2} \|  <1 for all i
      
    Remark: Notation for primal and dual step-sizes are reversed with comparison
            to PDGH.py
            
    Remark: this code implements serial sampling only, as presented in [2]
            (to be extended to more general case of [1] as future work)             
            
    References:
        
        [1]"Stochastic primal-dual hybrid gradient algorithm with arbitrary 
        sampling and imaging applications",
        Chambolle, Antonin, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schonlieb,
        SIAM Journal on Optimization 28, no. 4 (2018): 2783-2808.   
         
        [2]"Faster PET reconstruction with non-smooth priors by randomization and preconditioning",
        Matthias J Ehrhardt, Pawel Markiewicz and Carola-Bibiane Schönlieb,
        Physics in Medicine & Biology, Volume 64, Number 22, 2019.
        
        
    '''
    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=None,
                 x_init=None, prob=None, **kwargs):
        '''SPDHG algorithm creator
        Optional parameters
        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate 
        :param g: Convex function with "simple" proximal 
        :param sigma=(sigma_i): List of Step size parameters for Dual problem
        :param tau: Step size parameter for Primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities
        '''
        super(SPDHG, self).__init__(**kwargs)
        
        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, 
                        x_init=x_init, prob=prob)
    def set_up(self, f, g, operator, tau=None, sigma=None, x_init=None, prob=None):
        '''initialisation of the algorithm
        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate.
        :param g: Convex function with "simple" proximal 
        :param sigma: list of Step size parameters for dual problem
        :param tau: Step size parameter for primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities'''
        print("{} setting up".format(self.__class__.__name__, ))
                    
        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator
        self.tau = tau
        self.sigma = sigma
        self.prob = prob
        self.ndual_subsets = len(self.operator)
        
        # Compute norm of each sub-operator       
        norms = [operator.get_item(i,0).norm() for i in range(self.ndual_subsets)]
        
        if self.sigma is None and self.tau is None:
            self.sigma = [1.] * self.ndual_subsets
            self.tau = 1 / sum([si * ni**2 for si, ni in zip(self.sigma, norms)])
            
        if self.prob is None:
            self.prob = [1/self.ndual_subsets] * self.ndual_subsets
        
        # initialize primal variable 
        if x_init is None:
            self.x = self.operator.domain_geometry().allocate()
        else:
            self.x = x_init.copy()
            
        self.x_tmp = self.operator.domain_geometry().allocate()
        
        # initialize dual variable to 0
        self.y = operator.range_geometry().allocate()
        self.y_old = operator.range_geometry().allocate()
        
        # initialize variable z corresponding to back-projected dual variable
        self.z = operator.domain_geometry().allocate()
        self.zbar= A.domain_geometry().allocate()
        # relaxation parameter
        self.theta = 1
        self.update_objective()
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
    def update(self):
        
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self.x.axpby(1., -self.tau, self.zbar, out=self.x_tmp)
        self.g.proximal(self.x_tmp, self.tau, out=self.x)
        
        # Choose subset
        i = int(np.random.choice(len(self.sigma), 1, p=self.prob))
        
        # save previous iteration
        self.y_old[i].fill(self.y[i])
        
        # Gradient ascent for the dual variable
        # y[i] = y_old[i] + sigma[i] * K[i] x
        self.operator.get_item(i,0).direct(self.x, out=self.y[i])
        self.y[i].axpby(self.sigma[i], 1., self.y_old[i], out=self.y[i])
        self.f[i].proximal_conjugate(self.y[i], self.sigma[i], out=self.y[i])
        
        # Back-project
        # x_tmp = K[i]^*(y[i] - y_old[i])
        self.operator.get_item(i,0).adjoint(self.y[i]-self.y_old[i], out = self.x_tmp)
        # Update backprojected dual variable and extrapolate
        # z = z + x_tmp
        self.z.add(self.x_tmp, out =self.z)
        # zbar = z + (1 + theta/p[i]) x_tmp
        self.z.axpby(1., (1 + self.theta / self.prob[i]), self.x_tmp, out = self.zbar)
        
        
    def update_objective(self):
         p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
         d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))
#
         self.loss.append([p1, d1, p1-d1])

        
    @property
    def objective(self):
         '''alias of loss'''
         return [x[0] for x in self.loss]
    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]
#%%
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData, BlockDataContainer
import numpy as np 
import numpy                          
import matplotlib.pyplot as plt
from ccpi.optimisation.algorithms import PDHG
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, IndicatorBox, TotalVariation
                      
from ccpi.astra.operators import AstraProjectorSimple
import os, sys
from ccpi.framework import TestData
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
plt.figure(figsize=(10,10))
plt.subplot(1,2,2)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,1)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()
#%%
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
#%%
prob = [1/20]*(len(A)-1) + [1/2]
spdhg = SPDHG(f=F,g=G,operator=A, 
              max_iteration = 1000,
              update_objective_interval=200, prob = prob)
spdhg.run(1000, very_verbose = True)
plt.figure()
plt.imshow(spdhg.get_output().as_array())
plt.colorbar()
plt.show()
#%%
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



