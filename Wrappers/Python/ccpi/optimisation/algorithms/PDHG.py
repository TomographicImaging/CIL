#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:18:06 2019

@author: evangelos
"""
from ccpi.optimisation.algorithms import Algorithm


from ccpi.framework import ImageData
import numpy as np
import matplotlib.pyplot as plt
import time
from ccpi.optimisation.operators import BlockOperator
from ccpi.framework import BlockDataContainer

class PDHG(Algorithm):
    '''Primal Dual Hybrid Gradient'''

    def __init__(self, **kwargs):
        super(PDHG, self).__init__()
        self.f        = kwargs.get('f', None)
        self.operator = kwargs.get('operator', None)
        self.g        = kwargs.get('g', None)
        self.tau      = kwargs.get('tau', None)
        self.sigma    = kwargs.get('sigma', None)

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
        #x_tmp = x_old
        self.x = self.x_old.copy()
        self.y = self.y_old.copy()
        #y_tmp = y_old
        #y = y_tmp
            
        # relaxation parameter
        self.theta = 1

    def update(self):
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
                        
        self.x_old.fill(self.x)
        self.y_old.fill(self.y)
        #self.y_old = y.copy()
        #self.y = self.y_old

    def update_objective(self):
        self.loss.append([self.f(self.operator.direct(self.x)) + self.g(self.x),
            -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(- 1 * self.operator.adjoint(self.y)))
        ])


