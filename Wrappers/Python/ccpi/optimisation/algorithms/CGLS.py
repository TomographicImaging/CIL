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
from ccpi.optimisation.functions import Norm2Sq
import numpy

class CGLS(Algorithm):

    '''Conjugate Gradient Least Squares algorithm

    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
      tolerance: tolerance to stop algorithm
      
    Reference:
        https://web.stanford.edu/group/SOL/software/cgls/
      
    '''
    
    
    
    def __init__(self, **kwargs):
        
        super(CGLS, self).__init__()
        self.x        = kwargs.get('x_init', None)
        self.operator = kwargs.get('operator', None)
        self.data     = kwargs.get('data', None)
        self.tolerance     = kwargs.get('tolerance', 1e-6)
        if self.x is not None and self.operator is not None and \
           self.data is not None:
            print (self.__class__.__name__ , "set_up called from creator")
            self.set_up(x_init  =kwargs['x_init'],
                               operator=kwargs['operator'],
                               data    =kwargs['data'])
            
                                    
    def set_up(self, x_init, operator , data ):

        self.x = x_init * 0.
        self.r = data - self.operator.direct(self.x)
        self.s = self.operator.adjoint(self.r)
        
        self.p = self.s
        self.norms0 = self.s.norm()
        
        ##
        self.norms = self.s.norm()
        ##
        
        
        self.gamma = self.norms0**2
        self.normx = self.x.norm()
        self.xmax = self.normx   
        
        self.loss.append(self.r.squared_norm())
        self.configured = True         

        
    def update(self):
        
        self.q = self.operator.direct(self.p)
        delta = self.q.squared_norm()
        alpha = self.gamma/delta
                        
        self.x += alpha * self.p
        self.r -= alpha * self.q
        
        self.s = self.operator.adjoint(self.r)
        
        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        self.p = self.s + self.beta * self.p   
        
        self.normx = self.x.norm()
        self.xmax = numpy.maximum(self.xmax, self.normx)
                    

    def update_objective(self):
        a = self.r.squared_norm()
        if a is numpy.nan:
            raise StopIteration()
        self.loss.append(a)
        
    def should_stop(self):
        return self.flag() or self.max_iteration_stop_cryterion()
 
    def flag(self):
        flag  = (self.norms <= self.norms0 * self.tolerance) or (self.normx * self.tolerance >= 1)

        if flag:
            self.update_objective()
            if self.iteration > self._iteration[-1]:
                print (self.verbose_output())
            print('Tolerance is reached: {}'.format(self.tolerance))

        return flag
 
