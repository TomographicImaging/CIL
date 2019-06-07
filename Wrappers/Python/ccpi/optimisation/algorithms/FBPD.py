# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2019 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Created on Thu Feb 21 11:09:03 2019

@author: ofn77899
"""

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
        
class FBPD(Algorithm):
    '''FBPD Algorithm
    
    Parameters:
      x_init: initial guess
      f: constraint
      g: data fidelity
      h: regularizer
      opt: additional algorithm 
    '''
    def __init__(self, **kwargs):
        super(FBPD, self).__init__()
        self.f = kwargs.get('f', None)
        self.g = kwargs.get('g', ZeroFunction())
        self.g = kwargs.get('h', ZeroFunction())
        self.operator = kwargs.get('operator', None)
        self.x_init = kwargs.get('x_init',None)
        if self.x_init is not None and self.operator is not None:
            self.set_up(self.x_init, self.operator, self.f, self.g, self.h)

    def set_up(self, x_init, operator, constraint, data_fidelity, 
               regulariser, opt=None):

        
        # algorithmic parameters
        
        
        # step-sizes
        self.tau   = 2 / (data_fidelity.L + 2)
        self.sigma = (1/self.tau - data_fidelity.L/2) / regulariser.L
        
        self.inv_sigma = 1/self.sigma
    
        # initialization
        self.x = x_init
        self.y = operator.direct(self.x)
        self.update_objective()
        self.configured = True
        
    
    def update(self):
    
        # primal forward-backward step
        x_old = self.x
        self.x = self.x - self.tau * ( self.g.gradient(self.x) + self.operator.adjoint(self.y) )
        self.x = self.f.proximal(self.x, self.tau)
    
        # dual forward-backward step
        self.y += self.sigma * self.operator.direct(2*self.x - x_old);
        self.y -= self.sigma * self.h.proximal(self.inv_sigma*self.y, self.inv_sigma)   

        # time and criterion

    def update_objective(self):
        self.loss.append(self.f(self.x) + self.g(self.x) + self.h(self.operator.direct(self.x)))
