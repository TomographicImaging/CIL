#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:16:44 2019

@author: evangelos
"""


from ccpi.framework import ImageData 
import numpy as np                           

from numpy import inf
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize


from Algorithms.PDHG import PDHG
from Operators.CompositeOperator_DataContainer import CompositeOperator, CompositeDataContainer
from Operators.IdentityOperator import *
from Operators.GradientOperator import Gradient
from Operators.SymmetrizedGradientOperator import SymmetrizedGradient
from Operators.ZeroOperator import ZeroOp

from Functions.FunctionComposition import FunctionComposition_new
from Functions.mixed_L12Norm import mixed_L12Norm
from Functions.L2NormSquared import L2NormSq
from Functions.ZeroFun import ZeroFun


from skimage.util import random_noise



#%%
# Create a phantom 
N = 100
#
phantom = np.zeros((N,N))
#
x1 = np.linspace(0, int(N/2), N)
x2 = np.linspace(int(N/2), 0., N)
xv, yv = np.meshgrid(x1, x2)
#
xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T
#
phantom = xv
## Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
phantom = phantom/phantom.max()

plt.imshow(phantom)
plt.show()

#%%
# Add noise
#np.random.seed(10)
#z = np.random.random((N,N))
#n1 = phantom + 0.25 * z
#n1 = phantom + 0.4 * phantom.std() * np.random.random(phantom.shape)

n1 = random_noise(phantom, mode='gaussian', seed=10)
noisy_data = ImageData(n1)

alpha = 0.2
beta = 1

ig = (N,N)
ag = ig

Grad = Gradient(ig)
SymGrad = SymmetrizedGradient( ((2,)+ig), ((3,)+ig))

Id1 = Identity( Grad.range_dim(), Grad.range_dim() )
ZeroOp1 = ZeroOp( ig, SymGrad.range_dim())

ZeroOp2 = ZeroOp( SymGrad.domain_dim(), ag)
Aop = Identity(ig, ag)

#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
method = '1'
if method == '0':
    # Define block operator
    operator = CompositeOperator( (3,2), Grad, -1*Id1,\
                                         ZeroOp1, SymGrad,\
                                         Aop, ZeroOp2)

    f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
                                          mixed_L12Norm(beta, sym_grad=True),\
                                          L2NormSq(0.5, b=noisy_data) )
    g = ZeroFun()
    
else:
    ###########################################################################
    #         No Composite #
    ###########################################################################    
    operator = CompositeOperator( (2,2), Grad, -1*Id1,\
                                         ZeroOp1, SymGrad)
    
    operator1 = CompositeOperator( (2,1), Aop, ZeroOp1)
    
    f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
                                          mixed_L12Norm(beta, sym_grad=True))  
    
    g = FunctionComposition_new(operator1, L2NormSq(0.5, b=noisy_data), \
                                           ZeroFun())      

        
normK = operator.norm()      
## Primal & dual stepsizes
sigma = 1.0/normK
tau = 1.0/normK
#
opt = {'niter':3000}
result, total_time, its = PDHG(f, g, operator, \
                                  tau = tau, sigma = sigma, opt = opt)



plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.title('Noisy')
plt.show()
#
plt.imshow(result.get_item(0).as_array())
plt.colorbar()
plt.title('Reconstruction')
plt.show()
#
plt.plot(np.linspace(0,N,N), phantom[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), result.get_item(0).as_array()[int(N/2),:], label = 'Recon')
plt.legend()
###############################################################################
#%%  Compare with CVX

try_cvx = input("Do you want CVX comparison (0/1)")

if try_cvx=='0':

    from cvxpy import *
    import sys
    sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
    from cvx_functions import tgv

    ###Construct problem
    u = Variable((N, N))
    w1 = Variable((N, N))
    w2 = Variable((N, N))

    constraints = []
    fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
    solver = MOSEK
    regulariser = tgv(u,w1,w2,alpha,beta)

    obj =  Minimize( regulariser +  fidelity )
    prob = Problem( obj, constraints)

    resCVX = prob.solve(verbose = True, solver = solver)
    print()
    print('Objective value is {} '.format(obj.value))

##%%
    # Show results
    plt.imshow(u.value)
    plt.title('CVX-denoising')
    plt.colorbar()
    plt.show()
#
    plt.imshow(result.get_item(0).as_array())
    plt.title('PDHG-denoising')
    plt.colorbar()
    plt.show()
#
    dif = np.abs( result.get_item(0).as_array() - u.value)
    plt.imshow(dif)
    plt.title('Difference')
    plt.colorbar()
    plt.show()
#
    plt.plot(np.linspace(0,N,N), result.get_item(0).as_array()[50,:], label = 'CVX')
    plt.plot(np.linspace(0,N,N), u[50,:].value, label = 'PDHG')
    plt.plot(np.linspace(0,N,N), phantom[50,:], label = 'phantom')
#    plt.plot(np.linspace(0,N,N), noisy_data.as_array()[50,:], label = 'noisy')
    plt.legend()
    plt.show()
    
else:
    print('No CVX solution available')    

