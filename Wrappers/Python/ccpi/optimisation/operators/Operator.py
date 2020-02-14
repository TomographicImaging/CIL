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
import numpy
import functools

class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def __init__(self, domain_geometry, **kwargs):
        r'''
        Creator

        :param domain_geometry: domain of the operator
        :param range_geometry: range of the operator
        :type range_geometry: optional, default None
        '''
        self._norm = None
        self._domain_geometry = domain_geometry
        self._range_geometry = kwargs.get('range_geometry', None)

    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    def direct(self,x, out=None):
        '''Returns the application of the Operator on x'''
        raise NotImplementedError
    def norm(self, **kwargs):
        '''Returns the norm of the Operator
        
        Calling norm triggers the calculation of the norm of the operator. Normally this
        is a computationally expensive task, therefore we store the result of norm into 
        a member of the class. If the calculation has already run, following calls to 
        norm just return the saved member. 
        It is possible to force recalculation by setting the optional force parameter. Notice that
        norm doesn't take notice of how many iterations or of the initialisation of the PowerMethod, 
        so in case you want to recalculate by setting a higher number of iterations or changing the
        starting point or both you need to set :code:`force=True`

        :param iterations: number of iterations to run
        :type iterations: int, optional, default = 25
        :param x_init: starting point for the iteration in the operator domain
        :type x_init: same type as domain, a subclass of :code:`DataContainer`, optional, default None
        :parameter force: forces the recalculation of the norm
        :type force: boolean, default :code:`False`
        '''
        if self._norm is None or kwargs.get('force', False):
            self._norm = self.calculate_norm(**kwargs)
        return self._norm
    def calculate_norm(self, **kwargs):
        '''Calculates the norm of the Operator'''
        raise NotImplementedError
    def range_geometry(self):
        '''Returns the range of the Operator: Y space'''
        return self._range_geometry
    def domain_geometry(self):
        '''Returns the domain of the Operator: X space'''
        return self._domain_geometry
    def __rmul__(self, scalar):
        '''Defines the multiplication by a scalar on the left

        returns a ScaledOperator'''
        return ScaledOperator(self, scalar)
    
    def compose(self, *other, **kwargs):
        # TODO: check equality of domain and range of operators        
        #if self.operator2.range_geometry != self.operator1.domain_geometry:
        #    raise ValueError('Cannot compose operators, check domain geometry of {} and range geometry of {}'.format(self.operato1,self.operator2))    
        
        return CompositionOperator(self, *other, **kwargs) 

    def __add__(self, other):
        return SumOperator(self, other)

    def __mul__(self, scalar):
        return self.__rmul__(scalar)    
    
    def __neg__(self):
        """ Return -self """
        return -1 * self    
        
    def __sub__(self, other):
        """ Returns the subtraction of the operators."""
        return self + (-1) * other   


class LinearOperator(Operator):
    '''A Linear Operator that maps from a space X <-> Y'''
    def __init__(self, domain_geometry, **kwargs):
        super(LinearOperator, self).__init__(domain_geometry, **kwargs)
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        '''returns the adjoint/inverse operation
        
        only available to linear operators'''
        raise NotImplementedError
    
    @staticmethod
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
        s = numpy.zeros(iterations)
        # Loop
        for it in numpy.arange(iterations):
            operator.direct(x0,out=y_tmp)
            operator.adjoint(y_tmp,out=x1)
            x1norm = x1.norm()
            if hasattr(x0, 'squared_norm'):
                s[it] = x1.dot(x0) / x0.squared_norm()
            else:
                x0norm = x0.norm()
                s[it] = x1.dot(x0) / (x0norm * x0norm) 
            x1.multiply((1.0/x1norm), out=x0)
        return numpy.sqrt(s[-1]), numpy.sqrt(s), x0

    def calculate_norm(self, **kwargs):
        '''Returns the norm of the LinearOperator as calculated by the PowerMethod
        
        :param iterations: number of iterations to run
        :type iterations: int, optional, default = 25
        :param x_init: starting point for the iteration in the operator domain
        :type x_init: same type as domain, a subclass of :code:`DataContainer`, optional, None
        :parameter force: forces the recalculation of the norm
        :type force: boolean, default :code:`False`
        '''
        x0 = kwargs.get('x_init', None)
        iterations = kwargs.get('iterations', 25)
        s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
        return s1

    @staticmethod
    def dot_test(operator, domain_init=None, range_init=None, verbose=False, **kwargs):
        r'''Does a dot linearity test on the operator
        
        Evaluates if the following equivalence holds
        
        .. math::
        
          Ax\times y = y \times A^Tx
        
        The equivalence is tested within a user specified precision

        .. code::
        
          abs(desired-actual) < 1.5 * 10**(-decimal)

        :param operator: operator to test
        :param range_init: optional initialisation container in the operator range 
        :param domain_init: optional initialisation container in the operator domain 
        :returns: boolean, True if the test is passed.
        :param decimal: desired precision
        :type decimal: int, optional, default 4
        '''
        if range_init is None:
            y = operator.range_geometry().allocate('random_int')
        else:
            y = range_init
        if domain_init is None:
            x = operator.domain_geometry().allocate('random_int')
        else:
            x = domain_init
            
        fx = operator.direct(x)
        by = operator.adjoint(y)
        a = fx.dot(y)
        b = by.dot(x)
        if verbose:
            print ('Left hand side  {}, \nRight hand side {}'.format(a, b))
            print ("decimal ", kwargs.get('decimal', 4))
        try:
            numpy.testing.assert_almost_equal(abs((a-b)/a), 0., decimal=kwargs.get('decimal',4))
            return True
        except AssertionError as ae:
            print (ae)
            return False
        
        
class ScaledOperator(Operator):
    
    
    '''ScaledOperator

    A class to represent the scalar multiplication of an Operator with a scalar.
    It holds an operator and a scalar. Basically it returns the multiplication
    of the result of direct and adjoint of the operator with the scalar.
    For the rest it behaves like the operator it holds.
    
    :param operator: a Operator or LinearOperator
    :param scalar: a scalar multiplier
    
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
    
    def __init__(self, operator, scalar, **kwargs):
        '''creator

        :param operator: a Operator or LinearOperator
        :param scalar: a scalar multiplier
        :type scalar: float'''

        super(ScaledOperator, self).__init__(domain_geometry=operator.domain_geometry(), 
                                             range_geometry=operator.range_geometry())
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        self.scalar = scalar
        self.operator = operator
    def direct(self, x, out=None):
        '''direct method'''
        if out is None:
            return self.scalar * self.operator.direct(x, out=out)
        else:
            self.operator.direct(x, out=out)
            out *= self.scalar
    def adjoint(self, x, out=None):
        '''adjoint method'''
        if self.operator.is_linear():
            if out is None:
                return self.scalar * self.operator.adjoint(x, out=out)
            else:
                self.operator.adjoint(x, out=out)
                out *= self.scalar
        else:
            raise TypeError('Operator is not linear')
    def norm(self, **kwargs):
        '''norm of the operator'''
        return numpy.abs(self.scalar) * self.operator.norm(**kwargs)
    # def range_geometry(self):
    #     '''range of the operator'''
    #     return self.operator.range_geometry()
    # def domain_geometry(self):
    #     '''domain of the operator'''
    #     return self.operator.domain_geometry()
    def is_linear(self):
        '''returns whether the operator is linear
        
        :returns: boolean '''
        return self.operator.is_linear()


###############################################################################
################   SumOperator  ###########################################
###############################################################################      
    
class SumOperator(Operator):
    
    
    def __init__(self, operator1, operator2):
                
        self.operator1 = operator1
        self.operator2 = operator2
        
        # if self.operator1.domain_geometry() != self.operator2.domain_geometry():
        #     raise ValueError('Domain geometry of {} is not equal with domain geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
        # if self.operator1.range_geometry() != self.operator2.range_geometry():
        #     raise ValueError('Range geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
            
        self.linear_flag = self.operator1.is_linear() and self.operator2.is_linear()            
        
        super(SumOperator, self).__init__(domain_geometry=self.operator1.domain_geometry(),
                                          range_geometry=self.operator1.range_geometry()) 
                                  
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
    
    def calculate_norm(self, **kwargs):
        # TODO
        # find a way to not repeat this code. This is a fallback in case the 
        # operator is linear.
        if self.is_linear():
            x0 = kwargs.get('x0', None)
            iterations = kwargs.get('iterations', 25)
            s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
            return s1

###############################################################################
################   Composition  ###########################################
###############################################################################             
    
class Composition2Operator(Operator):
    
    def __init__(self, operator1, operator2):
        
        self.operator1 = operator1
        self.operator2 = operator2
        
        self.linear_flag = self.operator1.is_linear() and self.operator2.is_linear()        
        
        if self.operator2.range_geometry() != self.operator1.domain_geometry():
            raise ValueError('Domain geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
        super(Composition2Operator, self).__init__(self.operator1.domain_geometry(),
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
            

    def is_linear(self):
        return self.linear_flag             
            
    def calculate_norm(self, **kwargs):
        # TODO
        # find a way to not repeat this code. This is a fallback in case the 
        # operator is linear.
        if self.is_linear():
            x0 = kwargs.get('x0', None)
            iterations = kwargs.get('iterations', 25)
            s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
            return s1

class CompositionOperator(Operator):
    
    def __init__(self, *operators, **kwargs):
        
        # get a reference to the operators
        self.operators = operators
        
        self.linear_flag = functools.reduce(lambda x,y: x and y.is_linear(),
                                            self.operators, True)
        self.preallocate = kwargs.get('preallocate', False)
        if self.preallocate:
            self.tmp_domain = [op.domain_geometry().allocate() for op in self.operators[:-1]]
            self.tmp_range = [op.range_geometry().allocate() for op in self.operators[1:]]
            # pass
        
        # TODO address the equality of geometries
        # if self.operator2.range_geometry() != self.operator1.domain_geometry():
        #     raise ValueError('Domain geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
        super(CompositionOperator, self).__init__(
            domain_geometry=self.operators[-1].domain_geometry(),
            range_geometry=self.operators[0].range_geometry()) 
        
    def direct(self, x, out = None):

        if out is None:
            #return self.operator1.direct(self.operator2.direct(x))
            # return functools.reduce(lambda X,operator: operator.direct(X), 
            #                        self.operators[::-1][1:],
            #                        self.operators[-1].direct(x))
            for i,operator in enumerate(self.operators[::-1]):
                if i == 0:
                    step = operator.direct(x)
                else:
                    step = operator.direct(step)
            return step

        else:
            # tmp = self.operator2.range_geometry().allocate()
            # self.operator2.direct(x, out = tmp)
            # self.operator1.direct(tmp, out = out)
            
            # out.fill (
            #     functools.reduce(lambda X,operator: operator.direct(X), 
            #                        self.operators[::-1][1:],
            #                        self.operators[-1].direct(x))
            # )
            
            # TODO this is a bit silly but will handle the pre allocation later
            
            for i,operator in enumerate(self.operators[::-1]):
                if i == 0:
                    operator.direct(x, out=self.tmp_range[i])
                elif i == len(self.operators) - 1:
                    operator.direct(self.tmp_range[i-1], out=out)
                else:
                    operator.direct(self.tmp_range[i-1], out=self.tmp_range[i])
            
            
    def adjoint(self, x, out = None):
        
        if self.linear_flag: 
            
            if out is not None:
                # return self.operator2.adjoint(self.operator1.adjoint(x))
                # return functools.reduce(lambda X,operator: operator.adjoint(X), 
                #                    self.operators[1:],
                #                    self.operators[0].adjoint(x))

                for i,operator in enumerate(self.operators):
                    if i == 0:
                        operator.adjoint(x, out=self.tmp_domain[i])
                    elif i == len(self.operators) - 1:
                        step = operator.adjoint(self.tmp_domain[i-1], out=out)
                    else:
                        operator.adjoint(self.tmp_domain[i-1], out=self.tmp_domain[i])
                return

            else:
                for i,operator in enumerate(self.operators):
                    if i == 0:
                        step = operator.adjoint(x)
                    else:
                        step = operator.adjoint(step)
                
                return step
        else:
            raise ValueError('No adjoint operation with non-linear operators')
            

    def is_linear(self):
        return self.linear_flag             
            
    def calculate_norm(self, **kwargs):
        # TODO
        # find a way to not repeat this code. This is a fallback in case the 
        # operator is linear.
        if self.is_linear():
            x0 = kwargs.get('x0', None)
            iterations = kwargs.get('iterations', 25)
            s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
            return s1



