#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:18:06 2019

@author: evangelos
"""

from ccpi.framework import ImageData
import numpy as np
import matplotlib.pyplot as plt
import time


def PDHG_testGeneric(data, f, g, operator, ig, ag, tau = None, sigma = None, opt = None):
        
    # algorithmic parameters
    if opt is None: 
        opt = {'tol': 1e-6, 'niter': 500, 'show_iter': 100, \
               'memopt': False} 
        
    if sigma is None and tau is None:
        raise ValueError('Need sigma*tau||K||^2<1') 
                
    niter = opt['niter'] if 'niter' in opt.keys() else 1000
    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
    memopt = opt['memopt'] if 'memopt' in opt.keys() else False  
    show_iter = opt['show_iter'] if 'show_iter' in opt.keys() else False 
    stop_crit = opt['stop_crit'] if 'stop_crit' in opt.keys() else False 
    
    x_old = operator.alloc_domain_dim()
    y_old = operator.alloc_range_dim()
        
    
    xbar = x_old
    x_tmp = x_old
    x = x_old
    
    y_tmp = y_old
    y = y_tmp
        
    # relaxation parameter
    theta = 1
    
    t = time.time()
    
    for i in range(niter):
        
        opDirect = operator.direct(xbar)
        for i in range(operator.shape[0]):
            y_tmp[i] = y_old[i] + sigma * opDirect[i]
            y[i] = f[i].proximal_conj(y_tmp[i], sigma)
            
        opAdjoint = operator.adjoint(y)
        for i in range(operator.shape[1]):
            x_tmp[i] = x_old[i] - tau * opAdjoint[i]    
            x[i] = g.proximal(x_tmp[i], tau) 

        # Update
        for i in range(operator.shape[1]):
            xbar[i] = x[i] + theta * (x[i] - x_old[i])             
                                                
        x_old = x
        y_old = y   

                           
    t_end = time.time()        
        
    return x[0], t_end - t, i

