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
from numbers import Number
import numpy
from ccpi.optimisation.operators import ScaledOperator
import functools

class BlockScaledOperator(ScaledOperator):
    '''ScaledOperator

    A class to represent the scalar multiplication of an Operator with a scalar.
    It holds an operator and a scalar. Basically it returns the multiplication
    of the result of direct and adjoint of the operator with the scalar.
    For the rest it behaves like the operator it holds.

    Args:
       operator (Operator): a Operator or LinearOperator
       scalar (Number): a scalar multiplier
    Example:
       The scaled operator behaves like the following:
       sop = ScaledOperator(operator, scalar)
       sop.direct(x) = scalar * operator.direct(x)
       sop.adjoint(x) = scalar * operator.adjoint(x)
       sop.norm() = operator.norm()
       sop.range_geometry() = operator.range_geometry()
       sop.domain_geometry() = operator.domain_geometry()
    '''
    def __init__(self, operator, scalar, shape=None):
        if shape is None:
            shape = operator.shape
        
        if isinstance(scalar, (list, tuple, numpy.ndarray)):
            size = functools.reduce(lambda x,y:x*y, shape, 1)
            if len(scalar) != size:
                raise ValueError('Scalar and operators size do not match: {}!={}'
                .format(len(scalar), len(operator)))
            self.scalar = scalar[:]
            print ("BlockScaledOperator ", self.scalar)
        elif isinstance (scalar, Number):
            self.scalar = scalar
        else:
            raise TypeError('expected scalar to be a number of an iterable: got {}'.format(type(scalar)))
        self.operator = operator
        self.shape = shape
    def direct(self, x, out=None):
        print ("BlockScaledOperator self.scalar", self.scalar)
        #print ("self.scalar", self.scalar[0]* x.get_item(0).as_array())
        return self.scalar * (self.operator.direct(x, out=out))
    def adjoint(self, x, out=None):
        if self.operator.is_linear():
            return self.scalar * self.operator.adjoint(x, out=out)
        else:
            raise TypeError('Operator is not linear')
    def norm(self, **kwargs):
        return numpy.abs(self.scalar) * self.operator.norm(**kwargs)
    def range_geometry(self):
        return self.operator.range_geometry()
    def domain_geometry(self):
        return self.operator.domain_geometry()
    @property
    def T(self):
        '''Return the transposed of self'''
        #print ("transpose before" , self.shape)
        #shape = (self.shape[1], self.shape[0])
        ##self.shape = shape
        ##self.operator.shape = shape
        #print ("transpose" , shape)
        #return self
        return type(self)(self.operator.T, self.scalar)