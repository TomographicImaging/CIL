#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:18:06 2019

@author: evangelos
"""
from ccpi.optimisation.algorithms import Algorithm
from ccpi.framework import ImageData, DataContainer
import numpy as np
import numpy
import time
from ccpi.optimisation.operators import BlockOperator
from ccpi.framework import BlockDataContainer
from ccpi.optimisation.functions import FunctionOperatorComposition

class PDHG(Algorithm):
    '''Primal Dual Hybrid Gradient'''

    def __init__(self, **kwargs):
        super(PDHG, self).__init__()
        self.f        = kwargs.get('f', None)
        self.operator = kwargs.get('operator', None)
        self.g        = kwargs.get('g', None)
        self.tau      = kwargs.get('tau', None)
        self.sigma    = kwargs.get('sigma', None)
        self.memopt   = kwargs.get('memopt', False)

        if self.f is not None and self.operator is not None and \
           self.g is not None:
            print ("Calling from creator")
            self.set_up(self.f,
                        self.operator,
                        self.g, 
                        self.tau, 
                        self.sigma)

    def set_up(self, f, g, operator, tau = None, sigma = None, opt = None, **kwargs):
        # algorithmic parameters
            
        if sigma is None and tau is None:
            raise ValueError('Need sigma*tau||K||^2<1') 
                    
    
        self.x_old = self.operator.domain_geometry().allocate()
        self.y_old = self.operator.range_geometry().allocate()
        
        self.xbar = self.x_old.copy()
        
        self.x = self.x_old.copy()
        self.y = self.y_old.copy()
        if self.memopt:
            self.y_tmp = self.y_old.copy()
            self.x_tmp = self.x_old.copy()
        #y = y_tmp
            
        # relaxation parameter
        self.theta = 1

    def update(self):
        if self.memopt:
            # Gradient descent, Dual problem solution
            # self.y_old += self.sigma * self.operator.direct(self.xbar)
            self.operator.direct(self.xbar, out=self.y_tmp)
            self.y_tmp *= self.sigma
            self.y_old += self.y_tmp

            #self.y = self.f.proximal_conjugate(self.y_old, self.sigma)
            self.f.proximal_conjugate(self.y_old, self.sigma, out=self.y)

            # Gradient ascent, Primal problem solution
            self.operator.adjoint(self.y, out=self.x_tmp)
            self.x_tmp *= self.tau
            self.x_old -= self.x_tmp

            self.g.proximal(self.x_old, self.tau, out=self.x)

            #Update
            self.x.subtract(self.x_old, out=self.xbar)
            self.xbar *= self.theta
            self.xbar += self.x

            self.x_old.fill(self.x)
            self.y_old.fill(self.y)

        else:
            # Gradient descent, Dual problem solution
            self.y_old += self.sigma * self.operator.direct(self.xbar)
            self.y = self.f.proximal_conjugate(self.y_old, self.sigma)
            
            # Gradient ascent, Primal problem solution
            self.x_old -= self.tau * self.operator.adjoint(self.y)
            self.x = self.g.proximal(self.x_old, self.tau)

            #Update
            #xbar = x + theta * (x - x_old)
            self.xbar.fill(self.x)
            self.xbar -= self.x_old 
            self.xbar *= self.theta
            self.xbar += self.x

            self.x_old = self.x
            self.y_old = self.y

    def update_objective(self):
        p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        d1 = -(self.f.convex_conjugate(self.y) + self.g(-1*self.operator.adjoint(self.y)))

        self.loss.append([p1,d1,p1-d1])



def PDHG_old(f, g, operator, tau = None, sigma = None, opt = None, **kwargs):
        
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

    x_old = operator.domain_geometry().allocate()
    y_old = operator.range_geometry().allocate()       
            
    xbar = x_old.copy()
    x_tmp = x_old.copy()
    x = x_old.copy()
    
    y_tmp = y_old.copy()
    y = y_tmp.copy()

        
    # relaxation parameter
    theta = 1
    
    t = time.time()
    
    primal = []
    dual = []
    pdgap = []
    
    
    for i in range(niter):

        
        if not memopt:
            
            y_tmp = y_old +  sigma * operator.direct(xbar)            
            y = f.proximal_conjugate(y_tmp, sigma)
                
            x_tmp = x_old - tau*operator.adjoint(y)
            x = g.proximal(x_tmp, tau)
                 
            x.subtract(x_old, out=xbar)
            xbar *= theta
            xbar += x
                                                  
            x_old.fill(x)
            y_old.fill(y)
            
            
            if i%100==0:
                
                p1 = f(operator.direct(x)) + g(x)
                d1 = - ( f.convex_conjugate(y) + g(-1*operator.adjoint(y)) )
                primal.append(p1)
                dual.append(d1)
                pdgap.append(p1-d1)      
                print(p1, d1, p1-d1)
                               
            
        else:
            
            operator.direct(xbar, out = y_tmp)             
            y_tmp *= sigma
            y_tmp += y_old                      
            f.proximal_conjugate(y_tmp, sigma, out=y)

            operator.adjoint(y, out = x_tmp)   
            x_tmp *= -tau
            x_tmp += x_old

            g.proximal(x_tmp, tau, out = x)
            
            x.subtract(x_old, out=xbar)
            xbar *= theta
            xbar += x
                                                
            x_old.fill(x)
            y_old.fill(y)
            
            if i%100==0:
                
                p1 = f(operator.direct(x)) + g(x)
                d1 = - ( f.convex_conjugate(y) + g(-1*operator.adjoint(y)) )
                primal.append(p1)
                dual.append(d1)
                pdgap.append(p1-d1)      
                print(p1, d1, p1-d1)
            
                         
    t_end = time.time()        
        
    return x, t_end - t, primal, dual, pdgap



