#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy
from cil.optimisation.functions import Function
from cil.framework import VectorData, VectorGeometry

class Rosenbrock(Function):
    r'''Rosenbrock function

    .. math::

    F(x,y) = (\alpha - x)^2 + \beta(y-x^2)^2

    The function has a global minimum at .. math:: (x,y)=(\alpha, \alpha^2)

    '''
    def __init__(self, alpha, beta):
        super(Rosenbrock, self).__init__()

        self.alpha = alpha
        self.beta = beta

    def __call__(self, x):
        if not isinstance(x, VectorData):
            raise TypeError('Rosenbrock function works on VectorData only')
        vec = x.as_array()
        a = (self.alpha - vec[0])
        b = (vec[1] - (vec[0]*vec[0]))
        return a * a + self.beta * b * b

    def gradient(self, x, out=None):
        r'''Gradient of the Rosenbrock function

        .. math::

        \nabla f(x,y) = \left[ 2*((x-\alpha) - 2\beta x(y-x^2)) ; 2\beta (y - x^2)  \right]

        '''
        if not isinstance(x, VectorData):
            raise TypeError('Rosenbrock function works on VectorData only')

        vec = x.as_array()
        a = (vec[0] - self.alpha)
        b = (vec[1] - (vec[0]*vec[0]))

        res = numpy.empty_like(vec)
        res[0] = 2 * ( a - 2 * self.beta * vec[0] * b)
        res[1] = 2 * self.beta * b

        if out is not None:
            out.fill (res)
        else:
            return VectorData(res)
