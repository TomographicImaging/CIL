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
from ccpi.optimisation.functions import ZeroFun
        
class FBPD(Algorithm):
    '''FBPD Algorithm
    
    Parameters:
      x_init: initial guess
      f: constraint
      g: data fidelity
      h: regularizer
      opt: additional algorithm 
    '''
    constraint = None
    data_fidelity = None
    regulariser = None
    def __init__(self, **kwargs):
        pass
    def set_up(self, x_init, operator=None, constraint=None, data_fidelity=None,\
         regulariser=None, opt=None):

        # default inputs
        if constraint    is None: 
            self.constraint    = ZeroFun()
        else:
            self.constraint = constraint
        if data_fidelity is None:
            data_fidelity = ZeroFun()
        else:
            self.data_fidelity = data_fidelity
        if regulariser   is None:
            self.regulariser   = ZeroFun()
        else:
            self.regulariser = regulariser
        
        # algorithmic parameters
        
        
        # step-sizes
        self.tau   = 2 / (self.data_fidelity.L + 2)
        self.sigma = (1/self.tau - self.data_fidelity.L/2) / self.regulariser.L
        
        self.inv_sigma = 1/self.sigma
    
        # initialization
        self.x = x_init
        self.y = operator.direct(self.x)
        
    
    def update(self):
    
        # primal forward-backward step
        x_old = self.x
        self.x = self.x - self.tau * ( self.data_fidelity.grad(self.x) + self.operator.adjoint(self.y) )
        self.x = self.constraint.prox(self.x, self.tau);
    
        # dual forward-backward step
        self.y = self.y + self.sigma * self.operator.direct(2*self.x - x_old);
        self.y = self.y - self.sigma * self.regulariser.prox(self.inv_sigma*self.y, self.inv_sigma);   

        # time and criterion
        self.loss = self.constraint(self.x) + self.data_fidelity(self.x) + self.regulariser(self.operator.direct(self.x))
