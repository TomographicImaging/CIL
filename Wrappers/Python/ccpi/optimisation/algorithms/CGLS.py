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
from ccpi.optimisation.operators import Gradient, BlockOperator
from ccpi.framework import BlockDataContainer
import numpy, psutil, functools, time

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
        preallocate = kwargs.get('preallocate', False)

        if alpha is None:
            raise ValueError('Please specify alpha')

        if x_init is None and operator is not None:
            x_init = operator.domain_geometry().allocate(0)
        if operator is not None and data is not None:
            self.set_up(x_init=x_init, operator=operator, data=data, tolerance=tolerance, alpha=alpha)
            
    


    def set_up(self, x_init, operator, data, tolerance=1e-6, alpha=1e-2):
        '''initialisation of the algorithm

        :param operator: Linear operator for the inverse problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param data: Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        :param alpha: regularisation parameter
        '''
        try:
            print("{} setting up".format(self.__class__.__name__, ))
            gradient = Gradient(operator.domain_geometry(), backend='c')
            gradient.scalar = alpha
            # # required memory is 
            # def prod(X):
            #     return functools.reduce(lambda x,y:x*y, X, 1)
            # # 3 in operator domain (x, p s)
            # domain_size = 3 * x_init.size
            # # 2 in operator range (q r)
            # org = operator.range_geometry()
            # or_size = 2 * functools.reduce(lambda x, y: x*y, org.shape, 1)
            # # 3 in gradient range
            # grg = gradient.range_geometry()
            # gr_size = 3 * sum(prod(gradient.range_geometry().get_item(i).shape) for i in range(gradient.range_geometry().shape[0]))

            # required_mem = (domain_size + or_size + gr_size) * 32 / 8 / 1024**3
            # print ("Required memory: {:.3f} Gb".format(required_mem))
            # print ("Available memory {:.3f} Gb".format(psutil.virtual_memory().available/1024**3))
            # if required_mem > psutil.virtual_memory().available/1024**3:
            #     raise MemoryError('Insufficient Memory for process')
            # x_init.size + 
            # don't create a copy
            self.x = x_init
            data = BlockDataContainer(data, operator.domain_geometry().allocate(0))
            #data = BlockDataContainer(data, 0)
            
            self.operator = BlockOperator(operator, gradient)
            self.tolerance = tolerance

            print ("Allocating q")
            self.q = self.operator.range_geometry().allocate(value=None)

            # self.r = data - self.operator.direct(self.x)
            self.r = data - BlockDataContainer(operator.direct(self.x), gradient.scalar * gradient.direct(self.x))
            # residuals
            print ("Allocating and setting up r")
            #self.r = BlockDataContainer(- operator.direct(self.x), - gradient.scalar * gradient.direct(self.x)) 
            #self.r.add(data, out=self.r)
            #self.s = self.operator.adjoint(self.r)
            
            print ("Allocating and setting up p")
            self.p = gradient.adjoint(self.r.get_item(1))
            self.p.multiply(gradient.scalar, out=self.p)
            self.p.add(operator.adjoint(self.r.get_item(0)), out=self.p)

            print ("Allocating s")
            self.s = self.p.copy()
            print ("Allocating s1")
            self.s1 = gradient.domain_geometry().allocate(None)

            self.norms = self.p.norm()
            self.norms0 = self.norms
            self.gamma = self.norms**2
            self.normx = self.x.norm()
            
            
            self.loss.append(self.r.squared_norm())
            self.configured = True
            print("{} configured".format(self.__class__.__name__, ))
        except MemoryError as me:
            print (me)
            print ("Cannot allocate all this memory!")
     
    def update(self):
        '''single iteration'''
        operator = self.operator.get_item(0,0)
        gradient = self.operator.get_item(0,1)
        operator.direct(self.p, out=self.q.get_item(0))
        delta0 = self.q.get_item(0).squared_norm()
        # self.operator.get_item(0,0).direct(self.p, out=self.q.get_item(0))
        # delta0 = self.q.get_item(0).squared_norm()
        delta1, a = gradient.direct_L21norm(self.p, out=self.q.get_item(1))
        #term1.multiply(gradient.scalar, out=self.q.get_item(1))
        self.q.get_item(1).multiply(gradient.scalar, out=self.q.get_item(1))
        
        delta = delta0 + delta1 * numpy.abs(gradient.scalar)
        # print ("delta", delta, "delta0", delta0, "delta1", delta1)
        # delta = self.q.squared_norm()

        alpha = self.gamma/delta
        self.r.axpby(1, -alpha, self.q, out=self.r)
        
        # print ("alpha", alpha)                        
        # self.x += alpha * self.p
        # self.r -= alpha * self.q
        
        self.x.axpby(1, alpha, self.p, out=self.x)
        #self.x += alpha * self.p
        #self.r -= alpha * self.q

        # adjoint of block operator is sum of adjoints
        # sumsq, self.s = self.operator.adjoint(self.r)
        
        operator.adjoint(self.r.get_item(0), out=self.s)
        delta0 = self.s.norm()

        delta1, a = gradient.adjoint_L21norm(self.r.get_item(1), out=self.s1)
        self.norms = delta0 + numpy.sqrt(delta1) * gradient.scalar
        # self.norms = self.s.norm()

        #term0.add(term1, out=self.s)
        self.s.axpby(1,gradient.scalar, self.s1, out=self.s)
        
        # print ("self.norms {}".format(self.norms))
        
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        self.p.axpby(self.beta, 1., self.s , out=self.p)
        
        self.normx = self.x.norm()
        # self.xmax = numpy.maximum(self.xmax, self.normx)

