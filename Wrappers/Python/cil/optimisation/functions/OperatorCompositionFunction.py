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

from cil.optimisation.functions import Function
from cil.optimisation.operators import Operator, ScaledOperator

import warnings

class OperatorCompositionFunction(Function):

    """ Composition of a function with an operator as : :math:`(F \otimes A)(x) = F(Ax)`

            :parameter function: :code:`Function` F
            :parameter operator: :code:`Operator` A


        For general operator, we have no explicit formulas for convex_conjugate,
        proximal and proximal_conjugate

    """

    def __init__(self, function, operator):
        '''creator

    :param A: operator
    :type A: :code:`Operator`
    :param f: function
    :type f: :code:`Function`
    '''

        super(OperatorCompositionFunction, self).__init__()

        self.function = function
        self.operator = operator

    @property
    def L(self):
        if self._L is None:
            try:
                self._L = self.function.L  * (self.operator.norm() ** 2)
            except ValueError as ve:
                self._L = None
        return self._L

    def __call__(self, x):

        """ Returns :math:`F(Ax)`
        """

        return self.function(self.operator.direct(x))

    def gradient(self, x, out=None):

        """ Return the gradient of F(Ax),

        ..math ::  (F(Ax))' = A^{T}F'(Ax)

        """

        tmp = self.operator.range_geometry().allocate()
        self.operator.direct(x, out=tmp)
        self.function.gradient(tmp, out=tmp)
        if out is None:
            return self.operator.adjoint(tmp)
        else:
            self.operator.adjoint(tmp, out=out)
            return out

