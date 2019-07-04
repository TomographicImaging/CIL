# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
        super(PDHG, self).__init__(max_iteration=kwargs.get('max_iteration',0))
        self.f        = kwargs.get('f', None)
        self.operator = kwargs.get('operator', None)
        self.g        = kwargs.get('g', None)
        self.tau      = kwargs.get('tau', None)
        self.sigma    = kwargs.get('sigma', 1.)
        
        
        if self.f is not None and self.operator is not None and \
           self.g is not None:
            if self.tau is None:
                # Compute operator Norm
                normK = self.operator.norm()
                # Primal & dual stepsizes
                self.tau = 1/(self.sigma*normK**2)
            print ("Calling from creator")
            self.set_up(self.f,
                        self.g,
                        self.operator,
                        self.tau, 
                        self.sigma)

    def set_up(self, f, g, operator, tau = None, sigma = None, opt = None, **kwargs):
        # algorithmic parameters
        self.operator = operator
        self.f = f
        self.g = g
        self.tau = tau
        self.sigma = sigma
        self.opt = opt
        if sigma is None and tau is None:
            raise ValueError('Need sigma*tau||K||^2<1') 

        self.x_old = self.operator.domain_geometry().allocate()
        self.x_tmp = self.x_old.copy()
        self.x = self.x_old.copy()
    
        self.y_old = self.operator.range_geometry().allocate()
        self.y_tmp = self.y_old.copy()
        self.y = self.y_old.copy()

        self.xbar = self.x_old.copy()

        # relaxation parameter
        self.theta = 1
        self.update_objective()
        self.configured = True

    def update(self):
        
        # Gradient descent, Dual problem solution
        self.operator.direct(self.xbar, out=self.y_tmp)
        self.y_tmp *= self.sigma
        self.y_tmp += self.y_old

        #self.y = self.f.proximal_conjugate(self.y_old, self.sigma)
        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)
        
        # Gradient ascent, Primal problem solution
        self.operator.adjoint(self.y, out=self.x_tmp)
        self.x_tmp *= -1*self.tau
        self.x_tmp += self.x_old

            
        self.g.proximal(self.x_tmp, self.tau, out=self.x)

        #Update
        self.x.subtract(self.x_old, out=self.xbar)
        self.xbar *= self.theta
        self.xbar += self.x

        self.x_old.fill(self.x)
        self.y_old.fill(self.y)

    def update_objective(self):

        p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))

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
        

        
        if memopt:
            operator.direct(xbar, out = y_tmp)             
            y_tmp *= sigma
            y_tmp += y_old 
        else:
            y_tmp = y_old +  sigma * operator.direct(xbar)                        
        
        f.proximal_conjugate(y_tmp, sigma, out=y)
        
        if memopt:
            operator.adjoint(y, out = x_tmp)   
            x_tmp *= -1*tau
            x_tmp += x_old
        else:
            x_tmp = x_old - tau*operator.adjoint(y)
            
        g.proximal(x_tmp, tau, out=x)
             
        x.subtract(x_old, out=xbar)
        xbar *= theta
        xbar += x
                                              
        x_old.fill(x)
        y_old.fill(y)
                    
        if i%10==0:
            
            p1 = f(operator.direct(x)) + g(x)
            d1 = - ( f.convex_conjugate(y) + g.convex_conjugate(-1*operator.adjoint(y)) )            
            primal.append(p1)
            dual.append(d1)
            pdgap.append(p1-d1) 
            print(p1, d1, p1-d1)
            
        
                         
    t_end = time.time()        
        
    return x, t_end - t, primal, dual, pdgap



