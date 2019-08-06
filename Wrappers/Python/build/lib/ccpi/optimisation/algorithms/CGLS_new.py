# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Edoardo Pasca

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
Created on Thu Feb 21 11:11:23 2019

@author: ofn77899
"""

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import Norm2Sq
import numpy

class CGLS_new(Algorithm):

    '''Conjugate Gradient Least Squares algorithm

    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
    '''
    def __init__(self, **kwargs):
        super(CGLS_new, self).__init__()
        self.x        = kwargs.get('x_init', None)
        self.operator = kwargs.get('operator', None)
        self.data     = kwargs.get('data', None)
        if self.x is not None and self.operator is not None and \
           self.data is not None:
            print ("Calling from creator")
            self.set_up(x_init  =kwargs['x_init'],
                               operator=kwargs['operator'],
                               data    =kwargs['data'])
            
        if self.x.shape != self.operator.domain_geometry().allocate().shape:
            raise ValueError('Initial x0 not in the operator domain')

    def set_up(self, x_init, operator , data ):

        self.x = x_init
        self.r = data - self.operator.direct(self.x)
        self.s = self.operator.adjoint(self.r)
        
        self.p = self.s
        self.norms0 = self.s.norm()
        self.gamma = self.norms0**2
        self.normx = self.x.norm()
        self.xmax = self.normx        
        
#        self.r = data.copy()
#        self.x = x_init * 0
#
#        self.operator = operator
#        self.d = operator.adjoint(self.r)
#
#        
#        self.normr2 = self.d.squared_norm()
#        
#        self.s = self.operator.domain_geometry().allocate()
        #if isinstance(self.normr2, Iterable):
        #    self.normr2 = sum(self.normr2)
        #self.normr2 = numpy.sqrt(self.normr2)
        #print ("set_up" , self.normr2)
        
#        n = Norm2Sq(operator, self.data)
#        self.loss.append(n(x_init))
#        self.configured = True

    def update(self):
        self.update_new()
            

    def update_new(self):
        
        self.q = self.operator.direct(self.p)
        delta = self.q.squared_norm()
        alpha = self.gamma/delta
        
        self.x += alpha * p
        self.r -= alpha * q
        
        s = operator.adjoint(self.r)
        
        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        self.p = self.s + self.beta * self.p   
        
        self.normx = self.x.norm()
        self.xmax = self.xmax.maximum(self.normx)
        
        
        if self.gamma<=1e-6:
            raise StopIteration()        
        
        
#        Ad = self.operator.direct(self.d)
#        norm = Ad.squared_norm()
#        
#        if norm <= 1e-36:
#            print ('norm = 0, cannot update solution')
#            #print ("self.d norm", self.d.squared_norm(), self.d.as_array())
#            raise StopIteration()
#        alpha = self.normr2/norm
#        if alpha == 0.:
#            print ('alpha = 0, cannot update solution')
#            raise StopIteration()
#        self.d *= alpha
#        Ad *= alpha
#        self.r -= Ad
#        
#        self.x += self.d
#        
#        self.operator.adjoint(self.r, out=self.s)
#        s = self.s
#
#        normr2_new = s.squared_norm()
#        
#        beta = normr2_new/self.normr2
#        self.normr2 = normr2_new
#        self.d *= (beta/alpha) 
#        self.d += s

    def update_objective(self):
        a = self.r.squared_norm()
        if a is numpy.nan:
            raise StopIteration()
        self.loss.append(a)
        
#    def should_stop(self):
#        if self.iteration > 0:
#            x = self.get_last_objective()
#            a = x > 0
#            return self.max_iteration_stop_cryterion() or (not a)
#        else:
#            return False
