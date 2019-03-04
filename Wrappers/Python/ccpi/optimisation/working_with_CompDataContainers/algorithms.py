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
from operators import CompositeOperator


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

    if isinstance(operator, CompositeOperator):
        x_old = operator.alloc_domain_dim()
        y_old = operator.alloc_range_dim()
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
            
#        a1 = ImageData(operator.direct(x).power(2).sum(axis=0)).sqrt().sum()
#        a2 = 0.5*(x - g.b).power(2).sum()      
#        a3 = 0.5*(-1*operator.adjoint(y).power(2).sum()) + (g.b * -1*operator.adjoint(y)).sum()
#        print(a3, g.convex_conjugate(-1*operator.adjoint(y)))
        
#        print(a1+a2, f(x) + g(x))
        
#        d = -1*operator.adjoint(y)
#        a3 = 0.5 * d.power(2).sum() + (d*g.b).sum()
        
#        print(a3, g.convex_conjugate(-1*operator.adjoint(y)))
        
#        print( f(operator.direct(x)) + g(x) + g.convex_conjugate(-1*operator.adjoint(y)))
        
#        print(g.convex_conjugate(-1*operator.adjoint(y)))
#        print( f(x) + g(x) + f.convex_conjugate(y) + g.convex_conjugate(-1*operator.adjoint(y)))
        
        
#        print(f(x) + g(x) + f.convex_conjugate(y) + g.convex_conjugate(-1*operator.adjoint(y)))
        
        # check gap with tomo
        
        tv_term = 30*ImageData(operator.compMat[0][0].direct(x.get_item(0)).power(2).sum(axis=0)).sqrt().sum()
        l2_norm_sq = 0.5*((operator.compMat[1][0].direct(x.get_item(0)) - f.get_item(1).b).power(2).sum()) 
        
        tv_term_conj = 0
        l2_norm_sq_conj = 0.5 * y.get_item(1).power(2).sum() + (f.get_item(1).b * y.get_item(1)).sum()
        
                
        print( (tv_term + l2_norm_sq + tv_term_conj + l2_norm_sq_conj) /(50*50) )
        
#        a = 30*ImageData(operator.compMat[0][0].direct(x.get_item(0)).power(2).sum(axis=0)).sqrt().sum()+\
#            0.5*(operator.compMat[1][0].direct(x.get_item(0)) - f.get_item(1).b).power(2).sum() 
            
#        d = 0.5 * y.get_item(1).power(2).sum() + (f.get_item(1).b*y.get_item(1)).sum()

#            
#        d1 = -1* operator.compMat[0][0].adjoint(y.get_item(0)) - \
#                 operator.compMat[1][0].adjoint(y.get_item(1))
            
                                        
#        print(a + d + d1.maximum(0).sum())
            
        
        
        
        
        
#        print( f.convex_conjugate(y) )
#        print( g.convex_conjugate(-1*operator.adjoint(y)))
        
        
        
#        print( f(x), g(x), f.convex_conjugate(y), g.convex_conjugate(-1*operator.adjoint(y)) )
                
#        
#        print(f(x) + g(x) + f.convex_conjugate(y) + g.convex_conjugate(-1*operator.adjoint(y)))
   

                           
    t_end = time.time()        
        
    return x, t_end - t, objective

