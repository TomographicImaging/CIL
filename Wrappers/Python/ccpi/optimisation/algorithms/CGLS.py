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

from ccpi.optimisation.operators import Gradient, BlockOperator
from ccpi.framework import BlockDataContainer

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
        
        if x_init is None and operator is not None:
            x_init = operator.domain_geometry().allocate(0)
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
        
        self.p = self.s.copy()
        self.q = self.operator.range_geometry().allocate()
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
        
        self.operator.direct(self.p, out=self.q)
        delta = self.q.squared_norm()
        alpha = self.gamma/delta
         
        self.x.axpby(1, alpha, self.p, out=self.x)
        #self.x += alpha * self.p
        self.r.axpby(1, -alpha, self.q, out=self.r)
        #self.r -= alpha * self.q
        
        self.operator.adjoint(self.r, out=self.s)
        
        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        #self.p = self.s + self.beta * self.p   
        self.p.axpby(self.beta, 1, self.s, out=self.p)
        
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
 
class RCGLS(CGLS):
    '''Regularised CGLS

    Tikhonov regularisation
    '''
    def __init__(self, x_init=None, operator=None, data=None, tolerance=1e-6, **kwargs):
        '''initialisation of the algorithm

        :param operator : Linear operator for the inverse problem
        :param x_init : Initial guess ( Default x_init = 0)
        :param data : Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        super(CGLS, self).__init__(**kwargs)
        
        alpha = kwargs.get('alpha', None)
        if alpha is None:
            raise ValueError('Please specify alpha')

        if x_init is None and operator is not None:
            x_init = operator.domain_geometry().allocate(0)
        if x_init is not None and operator is not None and data is not None:
            self.set_up(x_init=x_init, operator=operator, data=data, tolerance=tolerance, alpha=alpha)

    def set_up(self, x_init, operator, data, tolerance=1e-6, alpha=1e-2):
        '''initialisation of the algorithm

        :param operator: Linear operator for the inverse problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param data: Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        :param alpha: regularisation parameter
        '''
        print("{} setting up".format(self.__class__.__name__, ))
        
        self.x = x_init * 0.
        data = BlockDataContainer(data, operator.domain_geometry().allocate(0))
        gradient = Gradient(operator.domain_geometry(), backend='c')
        # store the regularisation parameter into the gradient operator
        gradient.scalar = alpha
        self.operator = BlockOperator(operator, gradient)
        self.tolerance = tolerance

        # self.r = data - self.operator.direct(self.x)
        self.r = data - BlockDataContainer(operator.direct(self.x), gradient.scalar * gradient.direct(self.x))
        #self.s = self.operator.adjoint(self.r)
        self.s = gradient.adjoint(self.r.get_item(1))
        self.s.multiply(gradient.scalar, out=self.s)
        self.s.add(operator.adjoint(self.r.get_item(0)), out=self.s)

        self.p = self.s.copy()
        # don't need to preallocate as we reallocate at each iteration
        self.q = self.operator.range_geometry().allocate()
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
        gradient = self.operator.get_item(0,1)

        term0 = self.operator.get_item(0,0).direct(self.p)
        delta0 = term0.squared_norm()
        # self.operator.get_item(0,0).direct(self.p, out=self.q.get_item(0))
        # delta0 = self.q.get_item(0).squared_norm()
        delta1, term1 = self.operator.get_item(0,1).direct_L21norm(self.p)
        #term1.multiply(gradient.scalar, out=self.q.get_item(1))
        term1.multiply(gradient.scalar, out=term1)
        self.q = BlockDataContainer(term0, term1)

        delta = delta0 + delta1
        # print ("delta", delta, "delta0", delta0, "delta1", delta1)
        # delta = self.q.squared_norm()

        alpha = self.gamma/delta
        # print ("alpha", alpha)                        
        # self.x += alpha * self.p
        # self.r -= alpha * self.q
        
        self.x.axpby(1, alpha, self.p, out=self.x)
        #self.x += alpha * self.p
        self.r.axpby(1, -alpha, self.q, out=self.r)
        #self.r -= alpha * self.q

        # adjoint of block operator is sum of adjoints
        # sumsq, self.s = self.operator.adjoint(self.r)
        term0 = self.operator.get_item(0,0).adjoint(self.r.get_item(0))
        delta0 = term0.norm()
        delta1, term1 = gradient.adjoint_L21norm(self.r.get_item(1))
        self.norms = delta0 + numpy.sqrt(delta1)
        # self.norms = self.s.norm()

        #term0.add(term1, out=self.s)
        term0.axpby(1,gradient.scalar, term1, out=self.s)
        # print ("self.norms {}".format(self.norms))
        
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        self.p.axpby(self.beta, 1., self.s , out=self.p)
        
        self.normx = self.x.norm()
        self.xmax = numpy.maximum(self.xmax, self.normx)