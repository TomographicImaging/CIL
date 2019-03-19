#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData
import numpy as np
from ccpi.optimisation.operators import FiniteDiff
from ccpi.framework import ImageGeometry, BlockGeometry
from ccpi.framework import BlockDataContainer
from operators import Gradient, Identity, BlockOperator

M, N = 2, 3
ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
ag = ig
u = ig.allocate('random_int')
op1 = Gradient(ig)
op2 = Identity(ig, ag)

operator = BlockOperator(op1, op2, shape=(2,1)) 

d1 = op1.direct(u)
d2 = op2.direct(u)

d = operator.direct(u)

dd = operator.domain_geometry()
ww = operator.range_geometry()

print(d.get_item(0).get_item(0).as_array())
print(d.get_item(0).get_item(1).as_array())
print(d.get_item(1).as_array())

c1 = d + d

c2 = 2*d

c3 = d / (d+0.0001)


c5 = d.get_item(0).power(2).sum()



#%%############################################################################
## Create phantom for TV denoising
#
#N = 200
#data = np.zeros((N,N))
#data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
#data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
#
#ig = (N,N)
#ag = ig
#
## Create noisy data. Add Gaussian noise
#n1 = random_noise(data, mode='gaussian', seed=10)
#noisy_data = ImageData(n1)
#
## Regularisation Parameter
#alpha = 2
#
#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
#
#if method == '0':
#
#    # Create operators
#    op1 = Gradient(ig)
#    op2 = Identity(ig, ag)
#
#    # Form Composite Operator
#    operator = CompositeOperator((2,1), op1, op2 ) 
#
#    #### Create functions
##    f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
##                                    L2NormSq(0.5, b = noisy_data) )    
#    
#    f1 = mixed_L12Norm(alpha)
#    f2 = L2NormSq(0.5, b = noisy_data)
#    
#    f = BlockFunction( operator, f1, f2 )                                        
#    g = ZeroFun()
#    
#else:
#    
#    ###########################################################################
#    #         No Composite #
#    ###########################################################################
#    operator = Gradient(ig)
#    f = FunctionOperatorComposition(operator, mixed_L12Norm(alpha))
#    g = L2NormSq(0.5, b=noisy_data)
#    ###########################################################################
##%%
#    
## Compute operator Norm
#normK = operator.norm()
#
## Primal & dual stepsizes
#sigma = 1
#tau = 1/(sigma*normK**2)
#
#
##%%
### Number of iterations
#opt = {'niter':1000}
###
#### Run algorithm
#result, total_time, objective = PDHG(f, g, operator, \
#                                  tau = tau, sigma = sigma, opt = opt)
##%%
####Show results
#if isinstance(result, CompositeDataContainer):
#    sol = result.get_item(0).as_array()
#else:
#    sol = result.as_array()
#    
##sol = result.as_array()
##
#plt.imshow(sol)
#plt.colorbar()
#plt.show()
##
####
#plt.imshow(noisy_data.as_array())
#plt.colorbar()
#plt.show()
###
#plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
#plt.legend()
#plt.show()
#
##%% 
#
#try_cvx = input("Do you want CVX comparison (0/1)")
#
#if try_cvx=='0':
#
#    from cvxpy import *
#    import sys
#    sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
#    from cvx_functions import TV_cvx
#
#    u = Variable((N, N))
#    fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
#    regulariser = alpha * TV_cvx(u)
#    solver = MOSEK
#    obj =  Minimize( regulariser +  fidelity)
#    constraints = []
#    prob = Problem(obj, constraints)
#
#    # Choose solver (SCS is fast but less accurate than MOSEK)
#    res = prob.solve(verbose = True, solver = solver)
#
#    print('Objective value is {} '.format(obj.value))
#
#
#    diff_pdhg_cvx = np.abs(u.value - sol)
#    plt.imshow(diff_pdhg_cvx)
#    plt.colorbar()
#    plt.title('|CVX-PDHG|')
#    plt.show()
#
#    plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
#    plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'PDHG')
#    plt.legend()
#    plt.show()
#
#else:
#    print('No CVX solution available')
