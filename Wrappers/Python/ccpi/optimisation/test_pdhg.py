#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 23:27:36 2019

@author: evangelos
"""

import scipy.misc
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

#%%

data = scipy.misc.ascent()
data = data/np.max(data)
data = resize(data, [100, 100], order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)

N, M = data.shape
np.random.seed(10)
        
x_old = np.zeros(data.shape)
y_old = np.zeros([len(data.shape), ] + list(data.shape) )
xbar = np.zeros(data.shape)
 
tau = 1/np.sqrt(8)
sigma = 1/np.sqrt(8)

theta = 1
alpha = 1

np.random.seed(10)
#noisy_data = random_noise(data,'gaussian', mean = 0, var = 0.01)
noisy_data = random_noise(data, 's&p', amount = 0.2)

plt.imshow(noisy_data)
plt.show()


#%%

def gradient(x):
    shape = [len(x.shape), ] + list(x.shape)
    gradient = np.zeros(shape)
    slice_all = [0, slice(None, -1),]
    for d in range(len(x.shape)):
        gradient[slice_all] = np.diff(x, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))  
    return gradient

def divergence(x):
    res = np.zeros(x.shape[1:])
    for d in range(x.shape[0]):
        this_grad = np.rollaxis(x[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2] 
    return -res   

def shrink(x, tau):
    return np.sign(x) * np.maximum(0, np.abs(x) - tau)
         

for i in range(5000):
    
    y_tmp = y_old + sigma * gradient(xbar)  
    n = np.maximum(np.sqrt(np.sum(y_tmp**2, 0))/alpha, 1.0)
    y = y_tmp/n
    
    x_tmp = x_old - tau * divergence(y)
    
    x = noisy_data + np.sign(x_tmp-noisy_data) * np.maximum(0, np.abs(x_tmp-noisy_data) - tau)
            
    xbar = x + theta * (x - x_old) 
      
    x_old = x
    y_old = y 
    xbar_old = xbar
    
    
        
    if (i+1)%500==0:
        print(i)
        plt.gray()
        plt.imshow(x)
        plt.show()
    
   
#%%    
    


#def PDHG(data, regulariser, fidelity, operator, tau = None, sigma = None, opt = None ):
#
#                              
#    if regulariser is None: regulariser = ZeroFun()
#    if fidelity is None: fidelity = ZeroFun()
#                
#    # algorithmic parameters
#    if opt is None: 
#        opt = {'tol': 1e-7, 'iter': 1000, 'memopt':False}
#    
#    max_iter = opt['iter'] if 'iter' in opt.keys() else 1000
#    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
#    memopt = opt['memopt'] if 'memopt' in opt.keys() else False  
#    show_iter = opt['show_iter'] if 'show_iter' in opt.keys() else False  
#            
#    obj_value = []
#    
#    # initialization
#    if memopt:
#        
#        x_init = DataContainer(np.zeros(data.shape))
#        y_init = DataContainer(np.zeros([len(data.shape), ] + list(data.shape) )) 
###        
#        x_old = x_init.copy()
#        y_old = y_init.copy()
###        
#        x_tilde = x_old.copy()
#        x = x_old.copy()
#        x_tmp = x_old.copy()        
#        y = y_old.copy()
#        y_tmp = y_old.copy()
##                       
#    else:

#            
#    # Start time
#    t = time.time()
#    
#    # Compute error
#    error_cmp = Norm2()
#    
#    # theta value
#    theta = 1
#                    
#    # Show results
#    print('Iter {:<5} || {:<5} PrimalObj {:<5} || {:<4} l2_error'.format(' ',' ',' ',' '))
#    
#    
#    for it in range(max_iter):
#        
#        if memopt:
#                    
#            operator.adjoint(y_old, out = x_tmp)
#            x_tmp *= -tau
#            x_tmp += x_old
#                        
#            x_tmp += tau * data
#            x_tmp *= 1/(1+tau)
#            x.fill(x_tmp)
#                        
#            x.subtract(x_old, out = x_tilde)
#            x_tilde *= theta
#            x_tilde.add(x, out = x_tilde)
#                                    
#            operator.direct(x_tilde, out = y_tmp)
#            y_tmp *= sigma
#            y_tmp += y_old
#                        
#            regulariser.proximal(y_tmp, sigma, out = y)
#                                                
#        else:
#            
#            x_tmp = x_old - tau * operator.adjoint(y_old)
#            
##            tmp1 = np.maximum(0, np.abs(x_tmp.as_array() - data.as_array()) - tau)
##            x = data + DataContainer(np.multiply(np.sign(x_tmp.as_array() - data.as_array()), tmp1))
##            x = data.as_array() + np.multiply(np.abs(x_tmp.as_array() - data.as_array()) , np.maximum(0, np.abs(x_tmp.as_array() - data.as_array()) - tau))
##            x = DataContainer(x)
##            y = np.maximum(0, np.abs())
##            y = y_tmp
#            
##            u1 = @ (aux,g,tau) g + (sign(aux - g) .* max(0,abs(aux - g) - tau ));            
#            
##            x = data + ((x_tmp.subtract(data)).sign()).multiply((x_tmp.subtract(data) - tau).maximum(0))
#            
#            
#            x = 0.5 * (x_tmp - tau) + ( (x_tmp - tau).power(2) + 4*tau*data).sqrt()
##            u1 = @ (aux,g,tau)  0.5 * ( (aux-tau)+sqrt( (aux-tau).^2 + 4*tau*g ) ) ;    
#        
#            
##            x = fidelity.proximal(x_tmp, data, tau)
#            
#            
#            x_tilde = x + theta * (x - x_old) 
#            y_tmp = y_old + sigma * operator.direct(x_tilde)                        
#            y = regulariser.proximal(y_tmp, sigma)
#            
#
# 
#        # Compute objective ( primal function ) 
##        obj_value.append(regulariser(x_old) + fidelity(x_old))
##         
#        error = error_cmp(x-x_old)
##        
#        if error < tol:
#           break
#       
#        if it%show_iter == 0:
#            plt.imshow(x.as_array())
#            plt.show()
##        
##        if it % show_iter==0:
##            print('{} {:<5} || {:<5} {:.4f} {:<5} || {:<5} {:.4g}'.format(it,' ',' ',obj_value[it],' ',' ',error))                 
#        if memopt:        
#            x_old.fill(x)
#            y_old.fill(y)
#        else:
#            x_old = x
#            y_old = y
#
#    # End time        
#    t_end = time.time()        
#        
#    return x, t_end - t, obj_value, error 

