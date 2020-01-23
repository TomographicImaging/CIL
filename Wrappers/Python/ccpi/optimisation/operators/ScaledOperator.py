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

class ScaledOperator(object):
    
    
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
    
    def __init__(self, operator, scalar):
        '''creator

        :param operator: a Operator or LinearOperator
        :param scalar: a scalar multiplier
        :type scalar: float'''
        super(ScaledOperator, self).__init__()
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
    def range_geometry(self):
        '''range of the operator'''
        return self.operator.range_geometry()
    def domain_geometry(self):
        '''domain of the operator'''
        return self.operator.domain_geometry()
    def is_linear(self):
        '''returns whether the operator is linear
        
        :returns: boolean '''
        return self.operator.is_linear()

