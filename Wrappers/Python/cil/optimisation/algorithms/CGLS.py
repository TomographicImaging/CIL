# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.algorithms import Algorithm
import numpy
import warnings

class CGLS(Algorithm):

    r'''Conjugate Gradient Least Squares algorithm 
    
    Problem:  

    .. math::

      \min || A x - b ||^2_2
    
    |

    Parameters :
        
      :parameter operator : Linear operator for the inverse problem
      :parameter initial : Initial guess ( Default initial = 0)
      :parameter data : Acquired data to reconstruct       
      :parameter tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
      
    Reference:
        https://web.stanford.edu/group/SOL/software/cgls/
    '''
    def __init__(self, initial=None, operator=None, data=None, tolerance=1e-6, **kwargs):
        '''initialisation of the algorithm

        :param operator : Linear operator for the inverse problem
        :param initial : Initial guess ( Default initial = 0)
        :param data : Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        super(CGLS, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data, tolerance=tolerance)

    def set_up(self, initial, operator, data, tolerance=1e-6):
        '''initialisation of the algorithm

        :param operator: Linear operator for the inverse problem
        :param initial: Initial guess ( Default initial = 0)
        :param data: Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        print("{} setting up".format(self.__class__.__name__, ))
        
        self.x = initial * 0.
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
        
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
     

        
    def update(self):
        '''single iteration'''
        
        self.operator.direct(self.p, out=self.q)
        delta = self.q.squared_norm()
        alpha = self.gamma/delta
         
        self.x.sapyb(1, self.p, alpha, out=self.x)
        #self.x += alpha * self.p
        self.r.sapyb(1, self.q, -alpha, out=self.r)
        #self.r -= alpha * self.q
        
        self.operator.adjoint(self.r, out=self.s)
        
        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        #self.p = self.s + self.beta * self.p   
        self.p.sapyb(self.beta, self.s, 1, out=self.p)

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
 
