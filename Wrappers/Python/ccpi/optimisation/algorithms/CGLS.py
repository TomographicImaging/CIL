# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.optimisation.algorithms import Algorithm
import numpy

class CGLS(Algorithm):

    r'''Conjugate Gradient Least Squares algorithm 
    
    Problem:  

    .. math::

      \min || A x - b ||^2_2
    
    |

    Parameters :
        
      :parameter operator : Linear operator for the inverse problem
      :parameter x_init : Initial guess ( Default x_init = 0)
      :parameter data : Acquired data to reconstruct       
      :parameter tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
      
    Reference:
        https://web.stanford.edu/group/SOL/software/cgls/
    '''
    def __init__(self, x_init=None, operator=None, data=None, tolerance=1e-6, **kwargs):
        '''initialisation of the algorithm

        :param operator : Linear operator for the inverse problem
        :param x_init : Initial guess ( Default x_init = 0)
        :param data : Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        super(CGLS, self).__init__(**kwargs)
        

        if x_init is not None and operator is not None and data is not None:
            self.set_up(x_init=x_init, operator=operator, data=data, tolerance=tolerance)

    def set_up(self, x_init, operator, data, tolerance=1e-6):
        '''initialisation of the algorithm

        :param operator: Linear operator for the inverse problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param data: Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        print("{} setting up".format(self.__class__.__name__, ))
        
        self.x = x_init * 0.
        self.operator = operator
        self.tolerance = tolerance

        self.r = data - self.operator.direct(self.x)
        self.s = self.operator.adjoint(self.r)
        
        self.p = self.s
        self.norms0 = self.s.norm()
        
        self.norms = self.s.norm()

        self.gamma = self.norms0**2
        self.normx = self.x.norm()
        self.xmax = self.normx   
        
        self.loss.append(self.r.squared_norm())
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
     

        
    def update(self):
        '''single iteration'''
        
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
        '''stopping criterion'''
        return self.flag() or self.max_iteration_stop_cryterion()
 
    def flag(self):
        '''returns whether the tolerance has been reached'''
        flag  = (self.norms <= self.norms0 * self.tolerance) or (self.normx * self.tolerance >= 1)

        if flag:
            self.update_objective()
            if self.iteration > self._iteration[-1]:
                print (self.verbose_output())
            print('Tolerance is reached: {}'.format(self.tolerance))

        return flag
 
