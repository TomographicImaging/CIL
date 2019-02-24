#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:31:47 2019

@author: evangelos
"""


from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.ops import PowerMethodNonsquare


import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG_testGeneric
from operators import CompositeOperator, Identity
from GradientOperators import Gradient
from functions import L1Norm, ZeroFun, L2NormSq

#%%###############################################################################
# Create phantom for TV
N = 100
#
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = (N,N)
ag = ig

noisy_data = ImageData(random_noise(data,'gaussian', mean = 0, var = 0.01))
alpha = 1
#%%############################################################################

# Create operators
op1 = Gradient(ig)
op2 = Identity(ig, ag)

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 

f = [ L1Norm(Gradient(ig),alpha), \
      L2NormSq(Identity(ig,ag), noisy_data, c = 0.5, memopt = False) ]
g = ZeroFun()

# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
#
opt = {'niter':500}
res, total_time, its = PDHG_testGeneric(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)



#%%
#Kdim = K.domain_dim()
#Kran = K.range_dim()
#
#x_old = operator.alloc_domain_dim()
#y_old = operator.alloc_range_dim()
#
#z = K.direct(x_old)





#%%
#K.alloc_range_dim()
#

# Compute K.direct, K.adjoint
#u = ImageData(np.random.randint(10, size=(N,N)))
#z1, z2 = op1.direct(u), op2.direct(u)
##w1 = op1.adjoint(z1)
##w2 = op2.adjoint(z2)
#
########################################################################
#
#w1 = K.direct([u]), 
#w2 = K.adjoint(z1,z2)
#

#w = [ImageData(np.random.randint(10, size=(N,N))),\
#     ImageData(np.random.randint(10, size=(N,N)))]

#xx = [w,u]
#out = [[None]*K.shape[1], [None]
#for i in range(K.shape[1]):
#    for j in range(K.shape[0]):
#        if j == 0:
#            ww = K.opMatrix()[i][j].adjoint(xx[j])
#        else:
#            ww += K.opMatrix()[i][j].adjoint(xx[j])
#    out[i] = ww







#%%############################################################################

# one approach g not 0
#f = [ L1Norm(gradient(ig), alpha ) ]
#g = Norm2sq_new(MyTomoIdentity(ig,ag), noisy_data, c = 0.5, memopt = False)
#operator = [ [gradient(ig)] ]



###############################################################################
#%%
###############################################################################
# Create a phantom for TGV
#N = 200
##
#data = np.zeros((N,N))
##
#x1 = np.linspace(0, int(N/2), N)
#x2 = np.linspace(int(N/2), 0., N)
#xv, yv = np.meshgrid(x1, x2)
##
#xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T
#data = xv/np.max(xv)
#
#
#noisy_data = ImageData(random_noise(data,'gaussian', mean = 0, var = 0.01))
#
#
#ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
#ag = ig
#
#ig_sym_grad = ImageGeometry(voxel_num_x=N, voxel_num_y=N, channels = 2)
#ig_sym_grad1 = ImageGeometry(voxel_num_x=N, voxel_num_y=N, channels = 3)
#ig_grad = ImageGeometry(voxel_num_x=N, voxel_num_y=N, channels = 2)

# Construct operator matrix 
#operator = [ [gradient(ig), MyTomoIdentity(ig_grad, ig_sym_grad, -1) ],\
#              [ZeroOp(ig, ig_sym_grad1), sym_gradient(ig_sym_grad) ],\
#              [MyTomoIdentity(ig, ag), ZeroOp(ig_grad, ag)] ]

#operator = CompositeOperator((3,2), gradient(ig), MyTomoIdentity(ig_grad, ig_sym_grad, -1), \
#                                    ZeroOp(ig, ig_sym_grad1), sym_gradient(ig_sym_grad),\
#                                    MyTomoIdentity(ig, ag), ZeroOp(ig_grad, ag) )
                             
#normK = operator.opNorm()                             

# Compute operator norm
#normK = compute_opNorm(operator)

#alpha = 0.2
#beta = 1
##
#f = [ L1Norm(gradient(ig), MyTomoIdentity(ig_grad, ig_sym_grad), alpha ), \
#      L1NormTGV(sym_gradient(ig_sym_grad), beta),\
#      Norm2sq_new(MyTomoIdentity(ig,ag), noisy_data, c = 0.5, memopt = False) ]
#g = ZeroFun()
#     

#%%

# Primal & dual stepsizes
#sigma = 1
#tau = 1/(sigma*normK**2)
#
#opt = {'niter':500}
#res, total_time, its = PDHG_testGeneric(noisy_data, f, g, operator, \
#                                  ig, ag, tau = tau, sigma = sigma, opt = opt)
#plt.imshow(res.as_array())
#plt.colorbar()
#plt.show()
##
#plt.imshow(noisy_data.as_array())
#plt.colorbar()
#plt.show()
#
#plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'Recon')
#plt.legend()

#%%





