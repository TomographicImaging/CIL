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
from ccpi.optimisation.functions import ZeroFun, L1Norm, \
                      MixedL21Norm, FunctionOperatorComposition, BlockFunction
                 

from skimage.util import random_noise



# ############################################################################
# Create phantom for TV denoising

N = 100
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data, mode = 's&p', salt_vs_pepper = 0.9)
noisy_data = ImageData(n1)

plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.show()

#%%

# Regularisation Parameter
alpha = 10

#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
method = '1'
if method == '0':

    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)

    # Form Composite Operator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    #### Create functions
#    f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
#                                    L2NormSq(0.5, b = noisy_data) )    
    
    f1 = alpha * MixedL21Norm()
    f2 = L1Norm(b = noisy_data)
    
    f = BlockFunction(f1, f2 )                                        
    g = ZeroFun()
    
else:
    
    ###########################################################################
    #         No Composite #
    ###########################################################################
    operator = Gradient(ig)
    f = alpha *  FunctionOperatorComposition(operator, MixedL21Norm())
    g = L1Norm(b = noisy_data)
    ###########################################################################
#%%
    
# Compute operator Norm
normK = operator.norm()
print ("normK", normK)
# Primal & dual stepsizes
#sigma = 1
#tau = 1/(sigma*normK**2)

sigma = 1/normK
tau = 1/normK

opt = {'niter':2000}

res, time, primal, dual, pdgap = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
 
plt.figure(figsize=(5,5))
plt.imshow(res.as_array())
plt.colorbar()
plt.show()

#pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
#pdhg.max_iteration = 2000
#pdhg.update_objective_interval = 10
#
#pdhg.run(2000)

    

#sol = pdhg.get_output().as_array()
##sol = result.as_array()
##
#fig = plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(noisy_data.as_array())
##plt.colorbar()
#plt.subplot(1,2,2)
#plt.imshow(sol)
##plt.colorbar()
#plt.show()
##

##
#plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
#plt.legend()
#plt.show()


#%% Compare with cvx

try_cvx = input("Do you want CVX comparison (0/1)")

if try_cvx=='0':

    from cvxpy import *
    import sys
    sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
    from cvx_functions import TV_cvx

    u = Variable((N, N))
    fidelity = pnorm( u - noisy_data.as_array(),1)
    regulariser = alpha * TV_cvx(u)
    solver = MOSEK
    obj =  Minimize( regulariser +  fidelity)
    constraints = []
    prob = Problem(obj, constraints)

    # Choose solver (SCS is fast but less accurate than MOSEK)
    result = prob.solve(verbose = True, solver = solver)

    print('Objective value is {} '.format(obj.value))

    diff_pdhg_cvx = np.abs(u.value - res.as_array())
    plt.imshow(diff_pdhg_cvx)
    plt.colorbar()
    plt.title('|CVX-PDHG|')        
    plt.show()

    plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
    plt.plot(np.linspace(0,N,N), res.as_array()[int(N/2),:], label = 'PDHG')
    plt.legend()
    plt.show()

else:
    print('No CVX solution available')




