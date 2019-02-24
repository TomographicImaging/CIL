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


def get_primal_dual_variable(operator):
    
    x_tmp = []
    y_tmp = []
        
#    for i in range(operator.d[1]):    
#        x_tmp.append(ImageData(np.zeros(*operator.get_item(0,i).domain_dim())))
        
#    for j in range(operator.shape[0]):
#        y_tmp.append(ImageData(np.zeros(*operator.get_item(j,0).range_dim())))    
#    
    for i in range(len(operator[0])):    
        x_tmp.append(ImageData(np.zeros(operator[0][i].domain_dim())))
#        
    for j in range(len(operator)):
        y_tmp.append(ImageData(np.zeros(operator[j][0].range_dim())))


    return x_tmp, y_tmp


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

    # Initialisation
    x_old, y_old = get_primal_dual_variable(operator) 
#    x_old = CompositeDataContainer()
        
    
    xbar = x_old
    x_tmp = x_old
    x = x_old
    
    y_tmp = y_old
    y = y_tmp
        
    # relaxation parameter
    theta = 1
    
    t = time.time()
    
    for i in range(niter):
        
        # Gradient ascent in the dual variable y for Grad/Aop operator        
#        opDirect = operator.direct(xbar)
#        for i in range(operator.dimension[0]):
#            y_tmp[i] = y_old[i] + sigma *  opDirect[i]
#            y[i] = f[i].proximal_conj(y_tmp[i], sigma)
            
#            f = CompFuntion(L1no,L2no):
                
#                def prox_conj:
#                    return CompoDataCont
                    
        for i in range(len(operator)):
            z1 = ImageData(np.zeros(operator[i][0].range_dim()))
            for j in range(len(operator[0])):
                z1 += operator[i][j].direct(xbar[j])
            y_tmp[i] = y_old[i] + sigma *  z1
            y[i] = f[i].proximal_conj(y_tmp[i], sigma) 
                                           
        # Gradient descent in the primal variable x for Grad/Aop operator 
#        opAdjoint = operator.adjoint(y)
#        for i in range(operator.dimension[1]):
#            x_tmp[i] = x_old[i] - tau * opAdjoint[i]    
#            x[i] = g.proximal(x_tmp[i], tau) 
                    
        for i in range(len(operator[0])):
            z2 = ImageData(np.zeros(operator[0][i].domain_dim()))
            for j in range(len(operator)):
                z2 += operator[j][i].adjoint(y[j])
            x_tmp[i] = x_old[i] - tau * z2    
            x[i] = g.proximal(x_tmp[i], tau)  
                        
        # Update
        for i in range(len(operator[0])):
            xbar[i] = x[i] + theta * (x[i] - x_old[i]) 
                
        x_old = x
        y_old = y   

                           
    t_end = time.time()        
        
    return x[0], t_end - t, i

def PDHG(data, f, g, operator, ig, ag, tau = None, sigma = None, opt = None):
        
    # algorithmic parameters
    if opt is None: 
        opt = {'tol': 1e-6, 'niter': 10, 'show_iter': 100, \
               'memopt': False} 
        
    if sigma is None and tau is None:
        raise ValueError('Need sigma*tau||K||^2<1') 
                
    niter = opt['niter'] if 'niter' in opt.keys() else 1000
    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
    memopt = opt['memopt'] if 'memopt' in opt.keys() else False  
    show_iter = opt['show_iter'] if 'show_iter' in opt.keys() else False 
    stop_crit = opt['stop_crit'] if 'stop_crit' in opt.keys() else False 

    # Initialisation
#    x_old = [None]*
    
    x_old = ImageData(geometry=ig)
    xbar = ImageData(geometry=ig)

    y_old1 = ImageData(np.zeros([len(x_old.shape), ] + list(x_old.shape) ))
    y_old2 = ImageData(np.zeros(data.shape), geometry = ag) 
    
    y_old = [y_old1,y_old2]
    y_tmp = [None]*len(operator)
    y = [None]*len(operator)
        
    # relaxation parameter
    theta = 1
    
    t = time.time()
    
    for i in range(niter):
        
        # Gradient ascent in the dual variable y for Grad/Aop operator
        for i_op in range(len(operator)):
            y_tmp[i_op] = y_old[i_op] + sigma *  operator[i_op].direct(xbar)
            y[i_op] = f[i_op].proximal_conj(y_tmp[i_op], sigma)   
           
        # Gradient descent in the primal variable x for Grad/Aop operator            
        x_tmp = x_old - tau * sum([operator[iii].adjoint(y[iii]) for iii in range(len(operator))])                                    
        x = g.proximal(x_tmp, tau)
            
#        print(cmp_L2norm(x, x_old))
#        if cmp_L2norm(x, x_old)<1e-3:
#            break
        # Update
        xbar = x + theta * (x - x_old) 
        x_old = x
        y_old = y   

                           
    t_end = time.time()        
        
    return x, t_end - t, i