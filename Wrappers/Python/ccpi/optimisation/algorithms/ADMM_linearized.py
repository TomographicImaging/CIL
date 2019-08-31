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

class ADMM_linearized(Algorithm):
        
    ''' 
        Quick comments:
    
        ADMM :   min_{x} f(x) + g(y), subject to Ax + By = b
    
        Case: A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)  
                
        (Linearized) ADMM : The quadratic term in the augmented Lagrangian is linearized
                            for the x-update
                            
        Main algorithmic diff is that in ADMM computes two proximal, where in the PDHG
        one proxi, one prox conjugate                            
    
    
    '''

    def __init__(self, **kwargs):
        
        super(ADMM_linearized, self).__init__(max_iteration=kwargs.get('max_iteration',0))
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

        self.x = self.operator.domain_geometry().allocate()
        self.z0 = self.operator.range_geometry().allocate()
        self.u0 = self.z0.copy()
        
        self.x1 = self.x.copy() 
        self.z1 = self.z0.copy() 
        self.u1 = self.u0.copy() 
        
        self.tmp1 = self.z0.copy()
        self.tmp2 = self.x.copy()
        self.tmp3 = self.z0.copy()
    
        self.update_objective()
        self.configured = True        
        
    def update(self):
        
        self.operator.direct(self.x, out = self.tmp1)
        self.tmp1 += self.u0
        self.tmp1 += -1 * self.z0
        
        self.operator.adjoint(self.tmp1, out = self.tmp2)
        
        self.tmp2 *= -1 * (self.tau/self.sigma)
        self.tmp2 += self.x
        self.f.proximal( self.tmp2, self.tau, out = self.x1)
        
        self.operator.direct(self.x1, out = self.tmp3)
        self.tmp3 += self.u0
        
        self.g.proximal(self.tmp3, self.sigma, out = self.z1)

        self.operator.direct(self.x1, out = self.u1)
        self.u1 += self.u0
        self.u1 -= self.z1

        self.u0.fill(self.u1)
        self.z0.fill(self.z1)
        self.x.fill(self.x1)      
                                
    def update_objective(self):

        self.loss.append(self.f(self.x) + self.g(self.operator.direct(self.x)))