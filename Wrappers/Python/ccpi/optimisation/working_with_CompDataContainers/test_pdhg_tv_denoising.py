#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

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
from numpy import inf
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG
from operators import CompositeOperator, Identity, CompositeDataContainer
from GradientOperator import Gradient
#from functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq ,CompositeFunction
from test_functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq, CompositeFunction, FunctionComposition_new




from Sparse_GradMat import GradOper

#%%###############################################################################
# Create phantom for TV

N = 100
data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

ig = (N,N)
ag = ig

# Create noisy data. Add Gaussian noise
np.random.seed(10)
z = np.random.rand(N,N)
n1 = data + 0.25 * z
noisy_data = ImageData(n1)
alpha = 2

# Create operators
op1 = Gradient(ig)
op2 = Identity(ig, ag)

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 
###
#### Create functions
#f = CompositeFunction(operator, mixed_L12Norm(alpha), \
#                                L2NormSq(0.5, b = noisy_data) )
f = FunctionComposition_new(operator, mixed_L12Norm(alpha), \
                                L2NormSq(0.5, b = noisy_data) )
##
##
g = ZeroFun()

###############################################################################
#         No Composite #
###############################################################################
#operator = op1
#f = FunctionComposition_new(operator, mixed_L12Norm(alpha))
#g = L2NormSq(0.5, b=noisy_data)
###############################################################################
#%%
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes

sigma = 1
tau = 1/(sigma*normK**2)


#%%
## Number of iterations
opt = {'niter':1000}
##
### Run algorithm
result, total_time, objective = PDHG(f, g, operator, \
                                  tau = tau, sigma = sigma, opt = opt)
#%%
###Show results
sol = result.get_item(0).as_array()
#sol = result.as_array()
#
plt.imshow(sol)
plt.colorbar()
plt.show()
#
###
plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.show()
##
plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'Recon')
plt.legend()
plt.show()

#%%

from cvxpy import *
import sys
sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/cvx_scripts')
from cvx_functions import TV_cvx

u = Variable((N, N))
fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
regulariser = alpha * TV_cvx(u)
solver = MOSEK
obj =  Minimize( regulariser +  fidelity)
constraints = []
prob = Problem(obj, constraints)

# Choose solver (SCS is fast but less accurate than MOSEK)
res = prob.solve(verbose = True, solver = solver)
print()
print('Objective value is {} '.format(obj.value))


#%%

diff_pdhg_cvx = np.abs(u.value - sol)
plt.imshow(diff_pdhg_cvx)
plt.colorbar()
plt.show()

plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
plt.plot(np.linspace(0,N,N), sol[int(N/2),:], label = 'PDHG')
plt.legend()
plt.show()


#plt.imshow(diff_pdhg_cvx)
#plt.colorbar()
#plt.show()