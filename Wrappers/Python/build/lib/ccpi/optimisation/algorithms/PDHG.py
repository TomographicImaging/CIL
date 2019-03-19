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
from operators import BlockOperator
from ccpi.framework import BlockDataContainer

def PDHG(f, g, operator, tau = None, sigma = None, opt = None, **kwargs):
        
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

    if isinstance(operator, BlockOperator):
        x_old = operator.domain_geometry().allocate()
        y_old = operator.range_geometry().allocate()
    else:
        x_old = ImageData(np.zeros(operator.domain_dim()))
        y_old = ImageData(np.zeros(operator.range_dim()))        
        
    
    xbar = x_old
    x_tmp = x_old
    x = x_old
    
    y_tmp = y_old
    y = y_tmp
        
    # relaxation parameter
    theta = 1
    
    t = time.time()
    
    objective = []
    
    for i in range(niter):
        
        # Gradient descent, Dual problem solution
        y_tmp = y_old + sigma * operator.direct(xbar)
        y = f.proximal_conjugate(y_tmp, sigma)
        
        # Gradient ascent, Primal problem solution
        x_tmp = x_old - tau * operator.adjoint(y)
        x = g.proximal(x_tmp, tau)
        
        #Update
        xbar = x + theta * (x - x_old)
                                
        x_old = x
        y_old = y   
        
#        pdgap
        print(f(x) + g(x) + f.convex_conjugate(y) + g.convex_conjugate(-1*operator.adjoint(y)) )
        
        
        
        
            
#        # TV denoising, pdgap with composite
#        
#        primal_obj = f.get_item(0).alpha * ImageData(operator.compMat[0][0].direct(x.get_item(0)).power(2).sum(axis=0)).sqrt().sum() +\
#                     0.5*( (operator.compMat[1][0].direct(x.get_item(0)) - f.get_item(1).b).power(2).sum()) 
#        dual_obj =  0.5 * ((y.get_item(1).power(2)).sum()) + ( y.get_item(1)*f.get_item(1).b ).sum() 
        
        # TV denoising, pdgap with no composite
        
        
        
#        primal_obj = f.get_item(0).alpha * ImageData(operator.compMat[0][0].direct(x.get_item(0)).power(2).sum(axis=0)).sqrt().sum() +\
#                     0.5*( (operator.compMat[1][0].direct(x.get_item(0)) - f.get_item(1).b).power(2).sum()) 
#        dual_obj =  0.5 * ((y.get_item(1).power(2)).sum()) + ( y.get_item(1)*f.get_item(1).b ).sum()         
                    
                   
#        print(primal_obj)
#        objective = primal_obj
#       
   

                           
    t_end = time.time()        
        
    return x, t_end - t, objective

