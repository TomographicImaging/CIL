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

from ccpi.optimisation.operators import Operator
import numpy


class LinearOperator(Operator):
    '''A Linear Operator that maps from a space X <-> Y'''
    def __init__(self):
        super(LinearOperator, self).__init__()
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        '''returns the adjoint/inverse operation
        
        only available to linear operators'''
        raise NotImplementedError
    
    @staticmethod
    def PowerMethod(operator, iterations, x_init=None):
        '''Power method to calculate iteratively the Lipschitz constant'''
        
        # Initialise random
        if x_init is None:
            x0 = operator.domain_geometry().allocate(type(operator.domain_geometry()).RANDOM_INT)
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
            s[it] = x1.dot(x0) / x0.squared_norm()
            x1.multiply((1.0/x1norm), out=x0)
        return numpy.sqrt(s[-1]), numpy.sqrt(s), x0

    def calculate_norm(self, **kwargs):
        '''Returns the norm of the LinearOperator as calculated by the PowerMethod'''
        x0 = kwargs.get('x0', None)
        iterations = kwargs.get('iterations', 25)
        s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
        return s1


