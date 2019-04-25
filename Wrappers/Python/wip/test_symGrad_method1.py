#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:53:03 2019

@author: evangelos
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Identity, \
                        Gradient, SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from skimage.util import random_noise

from timeit import default_timer as timer
#def dt(steps):
#    return steps[-1] - steps[-2]

# Create phantom for TGV Gaussian denoising

N = 3

data = np.zeros((N,N))

x1 = np.linspace(0, int(N/2), N)
x2 = np.linspace(int(N/2), 0., N)
xv, yv = np.meshgrid(x1, x2)

xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T

data = xv
data = data/data.max()

plt.imshow(data)
plt.show()

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
n1 = random_noise(data, mode = 'gaussian', mean=0, var = 0.005, seed=10)
noisy_data = ImageData(n1)


plt.imshow(noisy_data.as_array())
plt.title('Noisy data')
plt.show()

alpha, beta = 0.2, 1

#method = input("Enter structure of PDHG (0=Composite or 1=NotComposite): ")
method = '1'


# Create operators
op11 = Gradient(ig)
op12 = Identity(op11.range_geometry())

op22 = SymmetrizedGradient(op11.domain_geometry())

op21 = ZeroOperator(ig, op22.range_geometry())


op31 = Identity(ig, ag)
op32 = ZeroOperator(op22.domain_geometry(), ag)

operator1 = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
    
    
f1 = alpha * MixedL21Norm()
f2 = beta * MixedL21Norm() 
f3 = ZeroFunction()   
f_B3 = BlockFunction(f1, f2, f3)         
g_B3 = ZeroFunction()
    
    
    
# Create operators
op11 = Gradient(ig)
op12 = Identity(op11.range_geometry())

op22 = SymmetrizedGradient(op11.domain_geometry())

op21 = ZeroOperator(ig, op22.range_geometry())

operator2 = BlockOperator(op11, -1*op12, \
                         op21, op22, \
                         shape=(2,2) )  

#f1 = alpha * MixedL21Norm()
#f2 = beta * MixedL21Norm()     
f_B2 = BlockFunction(f1, f2)     
g_B2 =  0.5 * L2NormSquared(b = noisy_data)    
     

#%%

x_old1 = operator1.domain_geometry().allocate('random_int')
y_old1 = operator1.range_geometry().allocate()       
            
xbar1 = x_old1.copy()
x_tmp1 = x_old1.copy()
x1 = x_old1.copy()
    
y_tmp1 = y_old1.copy()
y1 = y_tmp1.copy()

x_old2 = x_old1.copy()
y_old2 = operator2.range_geometry().allocate()       
            
xbar2 = x_old2.copy()
x_tmp2 = x_old2.copy()
x2 = x_old2.copy()
    
y_tmp2 = y_old2.copy()
y2 = y_tmp2.copy()

sigma = 0.4
tau = 0.4

y_tmp1 = y_old1 +  sigma * operator1.direct(xbar1) 
y_tmp2 = y_old2 +  sigma * operator2.direct(xbar2)  

numpy.testing.assert_array_equal(y_tmp1[0][0].as_array(), y_tmp2[0][0].as_array()) 
numpy.testing.assert_array_equal(y_tmp1[0][1].as_array(), y_tmp2[0][1].as_array()) 
numpy.testing.assert_array_equal(y_tmp1[1][0].as_array(), y_tmp2[1][0].as_array()) 
numpy.testing.assert_array_equal(y_tmp1[1][1].as_array(), y_tmp2[1][1].as_array()) 


y1 = f_B3.proximal_conjugate(y_tmp1, sigma)
y2 = f_B2.proximal_conjugate(y_tmp2, sigma)

numpy.testing.assert_array_equal(y1[0][0].as_array(), y2[0][0].as_array()) 
numpy.testing.assert_array_equal(y1[0][1].as_array(), y2[0][1].as_array()) 
numpy.testing.assert_array_equal(y1[1][0].as_array(), y2[1][0].as_array()) 
numpy.testing.assert_array_equal(y1[1][1].as_array(), y2[1][1].as_array()) 


x_tmp1 = x_old1 - tau * operator1.adjoint(y1)
x_tmp2 = x_old2 - tau * operator2.adjoint(y2)

numpy.testing.assert_array_equal(x_tmp1[0].as_array(), x_tmp2[0].as_array()) 



                      







##############################################################################
#x_1 = operator1.domain_geometry().allocate('random_int')
#
#x_2 = BlockDataContainer(x_1[0], x_1[1])
#
#res1 = operator1.direct(x_1)
#res2 = operator2.direct(x_2)
#
#print(res1[0][0].as_array(), res2[0][0].as_array())
#print(res1[0][1].as_array(), res2[0][1].as_array())
#
#print(res1[1][0].as_array(), res2[1][0].as_array())
#print(res1[1][1].as_array(), res2[1][1].as_array())
#
##res1 = op11.direct(x1[0]) - op12.direct(x1[1])
##res2 = op21.direct(x1[0]) - op22.direct(x1[1])
