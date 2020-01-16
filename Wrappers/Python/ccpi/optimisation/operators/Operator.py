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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numbers import Number
import numpy as np

class Operator(object):
        
    '''Operator that maps from a space X -> Y'''
    
    
    def __init__(self, domain_gm=None, range_gm=None):
        
        '''
        An operator should have domain geometry and range geometry
                
        '''
                         
        self.domain_gm = domain_gm
        self.range_gm = range_gm          
        self.__norm = None
        
        
    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    
        
    def direct(self, x, out=None):
        '''Returns the application of the Operator on x'''
        raise NotImplementedError
        
                
    def norm(self, **kwargs):
        '''Returns the norm of the Operator'''
        if self.__norm is None:
            self.__norm = self.calculate_norm(**kwargs)            
        return self.__norm
    
#    def calculate_norm(self, **kwargs):
#        '''Returns the norm of the LinearOperator as calculated by the PowerMethod'''
#        x0 = kwargs.get('x0', None)
#        iterations = kwargs.get('iterations', 25)
#        s1, sall, svec = PowerMethod(self, iterations, x_init=x0)
#        return s1
    
    def calculate_norm(self, **kwargs):
        '''Calculates the norm of the Operator'''
        raise NotImplementedError
        
         
    def range_geometry(self):
        '''Returns the range of the Operator: Y space'''
        return self.range_gm
        
        
    def domain_geometry(self):
        '''Returns the domain of the Operator: X space'''
        return self.domain_gm
        
        
    # ALGEBRA for operators, use same structure as Function class
    
        # Add operators
        # Subtract operator
        # - Operator style        
        # Multiply with Scalar    
    
    def __add__(self, other):
        
        if isinstance(other, Operator):
            return SumOperator(self, other)
        
    def __radd__(self, other):        
        """ Making addition commutative. """
        return self + other     
    
    def __rmul__(self, scalar):
        """Returns a operator multiplied by a scalar."""               
        return ScaledOperator(self, scalar)    
    
    def __mul__(self, scalar):
        return self.__rmul__(scalar)    
    
    def __neg__(self):
        """ Return -self """
        return -1 * self    
        
    def __sub__(self, other):
        """ Returns the subtraction of the operators."""
        return self + (-1) * other   
    
    def compose(self, other):
                
#        if self.operator2.range_geometry != self.operator1.domain_geometry:
#            raise ValueError('Cannot compose operators, check domain geometry of {} and range geometry of {}'.format(self.operato1,self.operator2))    
        
        return CompositionOperator(self, other) 
        
class ScaledOperator(Operator):
    
    '''ScaledOperator

    A class to represent the scalar multiplication of an Operator with a scalar.
    It holds an operator and a scalar. Basically it returns the multiplication
    of the result of direct and adjoint of the operator with the scalar.
    For the rest it behaves like the operator it holds.
    
    Args:
       :param operator (Operator): a Operator or LinearOperator
       :param scalar (Number): a scalar multiplier
    
    Example:
       The scaled operator behaves like the following:

    .. code-block:: python

      sop = ScaledOperator(operator, scalar)
      sop.direct(x) = scalar * operator.direct(x)
      sop.adjoint(x) = scalar * operator.adjoint(x)
      sop.norm() = operator.norm()
      sop.range_geometry() = operator.range_geometry()
      sop.domain_geometry() = operator.domain_geometry()

    '''
    
    def __init__(self, operator, scalar):
        
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))   
            
        self.scalar = scalar
        self.operator = operator            
        
        super(ScaledOperator, self).__init__(self.operator.domain_geometry(),
                                             self.operator.range_geometry())

        
    def direct(self, x, out=None):
        if out is None:
            return self.scalar * self.operator.direct(x, out=out)
        else:
            self.operator.direct(x, out=out)
            out *= self.scalar
            
    def adjoint(self, x, out=None):
        if self.operator.is_linear():
            if out is None:
                return self.scalar * self.operator.adjoint(x, out=out)
            else:
                self.operator.adjoint(x, out=out)
                out *= self.scalar
        else:
            raise TypeError('No adjoint operation with non-linear operators')
            
    def norm(self, **kwargs):
        return np.abs(self.scalar) * self.operator.norm(**kwargs)
      
    def is_linear(self):
        return self.operator.is_linear()   
    
class SumOperator(Operator):
    
    
    def __init__(self, operator1, operator2):
                
        self.operator1 = operator1
        self.operator2 = operator2
        
#        if self.operator1.domain_geometry() != self.operator2.domain_geometry():
#            raise ValueError('Domain geometry of {} is not equal with domain geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
#        if self.operator1.range_geometry() != self.operator2.range_geometry():
#            raise ValueError('Range geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
            
        self.linear_flag = self.operator1.is_linear() and self.operator2.is_linear()            
        
        super(SumOperator, self).__init__(self.operator1.domain_geometry(),
                                          self.operator1.range_geometry()) 
                                  
    def direct(self, x, out=None):
        
        if out is None:
            return self.operator1.direct(x) + self.operator2.direct(x)
        else:
            #TODO check if allcating with tmp is faster            
            self.operator1.direct(x, out=out)
            out.add(self.operator2.direct(x), out=out)     

    def adjoint(self, x, out=None):
        
        if self.linear_flag:        
            if out is None:
                return self.operator1.adjoint(x) + self.operator2.adjoint(x)
            else:
                #TODO check if allcating with tmp is faster            
                self.operator1.adjoint(x, out=out)
                out.add(self.operator2.adjoint(x), out=out) 
        else:
            raise ValueError('No adjoint operation with non-linear operators')
                                        
    def is_linear(self):
        return self.linear_flag            
    
class CompositionOperator(Operator):
    
    def __init__(self, operator1, operator2):
        
        self.operator1 = operator1
        self.operator2 = operator2
        
        if self.operator2.range_geometry() != self.operator1.domain_geometry():
            raise ValueError('Domain geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
        super(CompositionOperator, self).__init__(self.operator1.domain_geometry(),
                                          self.operator2.range_geometry()) 
        
    def direct(self, x, out = None):

        if out is None:
            return self.operator1.direct(self.operator2.direct(x))
        else:
            tmp = self.operator2.range_geometry().allocate()
            self.operator2.direct(x, out = tmp)
            self.operator1.direct(tmp, out = out)
            
    def adjoint(self, x, out = None):
        
        if self.linear_flag: 
            
            if out is None:
                return self.operator2.adjoint(self.operator1.adjoint(x))
            else:
                
                tmp = self.operator1.domain_geometry().allocate()
                self.operator1.adjoint(x, out=tmp)
                self.operator2.adjoint(tmp, out=out)
        else:
            raise ValueError('No adjoint operation with non-linear operators')
                          

def PowerMethod(operator, iterations, x_init=None):
        '''Power method to calculate iteratively the Lipschitz constant
        
        :param operator: input operator
        :type operator: :code:`LinearOperator`
        :param iterations: number of iterations to run
        :type iteration: int
        :param x_init: starting point for the iteration in the operator domain
        :returns: tuple with: L, list of L at each iteration, the data the iteration worked on.
        '''
        
        # Initialise random
        if x_init is None:
            x0 = operator.domain_geometry().allocate('random')
        else:
            x0 = x_init.copy()
            
        x1 = operator.domain_geometry().allocate()
        y_tmp = operator.range_geometry().allocate()
        s = np.zeros(iterations)
        # Loop
        for it in np.arange(iterations):
            operator.direct(x0,out=y_tmp)
            operator.adjoint(y_tmp,out=x1)
            x1norm = x1.norm()
            if hasattr(x0, 'squared_norm'):
                s[it] = x1.dot(x0) / x0.squared_norm()
            else:
                x0norm = x0.norm()
                s[it] = x1.dot(x0) / (x0norm * x0norm) 
            x1.multiply((1.0/x1norm), out=x0)
        return np.sqrt(s[-1]), np.sqrt(s), x0        
    
