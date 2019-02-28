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
import numpy
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG, PDHG_Composite
from operators import CompositeOperator, Identity, CompositeDataContainer
from GradientOperators import Gradient
from functions import L1Norm, ZeroFun, CompositeFunction, mixed_L12Norm, L2NormSq


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
noisy_data = ImageData(random_noise(data,'gaussian', mean = 0, var = 0.01))
alpha = 1

# Create operators
op1 = Gradient(ig)
op2 = Identity(ig, ag)

# Form Composite Operator
operator = CompositeOperator((2,1), op1, op2 ) 

# Create functions
f = CompositeFunction(mixed_L12Norm(op1,None,alpha), \
                      L2NormSq(op2, noisy_data, c = 0.5) )

#f = CompositeFunction(mixed_L12Norm(op1, None, alpha), \
#                      L1Norm(op2, noisy_data, c = 1) )
#
g = ZeroFun()

###############################################################################
#         No Composite #
###############################################################################
#operator = op1
#f = mixed_L12Norm(op1,None,alpha)
#g = L2NormSq(op2, noisy_data, c = 0.5)
###############################################################################

# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes

diagonal_preconditioning = False

if diagonal_preconditioning:
    
    GradSparse = GradOper(ig, [1]*len(ig), direction = 'for', order = '1', bndrs = 'Neumann')

    out = [None]*len(ig)
    out1 = [None]*len(ig)

    for i in range(len(GradSparse)): 
        out[i] = np.reshape(np.array(np.abs(GradSparse[i]).sum(axis=0)), ig, 'F')
        out1[i] = np.reshape(np.array(np.abs(GradSparse[i]).sum(axis=1)), ig, 'F')        
                        
    # TODO, Infinity values should replace by 0 or 1????    
    tmp_tau = 1/sum(out)
    tmp_tau[tmp_tau==inf]=0
    
    tmp_sigma1 = 1/out1[0]
    tmp_sigma1[tmp_sigma1==inf]=0
    
    tmp_sigma2 = 1/out1[1]
    tmp_sigma2[tmp_sigma2==inf]=0
    
    tau = CompositeDataContainer(ImageData(tmp_tau))
    sigma = CompositeDataContainer(ImageData(tmp_sigma1), ImageData(tmp_sigma2))
            
else:
    
    sigma = 10
    tau = 1/(sigma*normK**2)
    
    

#%%
# Number of iterations
opt = {'niter':1000}
#
## Run algorithm
res, total_time, objective = PDHG_Composite(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)
#
##Show results
solution = res.get_item(0).as_array()
#solution = res.as_array()

plt.imshow(solution)
plt.colorbar()
plt.show()

##
plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.show()
#
plt.plot(np.linspace(0,N,N), data[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), solution[int(N/2),:], label = 'Recon')
plt.legend()

#%%

