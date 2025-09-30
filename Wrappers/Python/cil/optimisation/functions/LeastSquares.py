#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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

from cil.optimisation.operators import LinearOperator, DiagonalOperator
from cil.optimisation.functions import Function
from cil.framework import DataContainer
import warnings
from numbers import Number
import numpy as np


class LeastSquares(Function):
    r""" (Weighted) Least Squares function

    .. math:: F(x) = c\|Ax-b\|_2^2

    or if weighted

    .. math:: F(x) = c\|Ax-b\|_{2,W}^{2}

    where :math:`W=\text{diag}(weight)`.

    Parameters
    -----------
    A : LinearOperator
    b : Data, DataContainer
    c : Scaling Constant, float, default 1.0
    weight: DataContainer with all positive elements of size of the range of operator A, default None

    Note
    --------

    L is the  Lipshitz Constant of the gradient of :math:`F` which is :math:`2 c ||A||_2^2 = 2 c \sigma_1(A)^2`, or :math:`2 c ||W|| ||A||_2^2 = 2c||W|| \sigma_1(A)^2`, where :math:`\sigma_1(A)` is the largest singular value of :math:`A` and :math:`W=\text{diag}(weight)`.

    """

    def __init__(self, A, b, c=1.0, weight = None):
        super(LeastSquares, self).__init__()

        self.A = A  # Should be a LinearOperator
        self.b = b
        self.c = c  # Default 1.

        # weight
        self.weight = weight
        self._weight_norm = None

        if weight is not None:
            if (self.weight<0).any():
                raise ValueError('Weight contains negative values')


    def __call__(self, x):

        r""" Returns the value of :math:`F(x) = c\|Ax-b\|_2^2` or :math:`c\|Ax-b\|_{2,W}^2`, where :math:`W=\text{diag}(weight)`:

        """
        # c * (A.direct(x)-b).dot((A.direct(x) - b))
        y = self.A.direct(x)
        y.subtract(self.b, out = y)

        if self.weight is None:
            return y.dtype.type(self.c) * y.dot(y)
        else:
            wy = self.weight.multiply(y)
            return y.dtype.type(self.c) * y.dot(wy)

    def gradient(self, x, out=None):

        r""" Returns the value of the gradient of :math:`F(x)`:

        .. math:: F'(x) = 2cA^T(Ax-b)

        or

        .. math:: F'(x) = 2cA^T(W(Ax-b))

        where :math:`W=\text{diag}(weight)`.

        """
        if out is None:
            out = x * 0.0

        tmp = self.A.direct(x)
        tmp.subtract(self.b , out=tmp)
        if self.weight is not None:
            tmp.multiply(self.weight, out=tmp)
        self.A.adjoint(tmp, out = out)
        out.multiply(self.c * 2.0, out=out)
        return out

    @property
    def L(self):
        if self._L is None:
            self.calculate_Lipschitz()
        return self._L
    @L.setter
    def L(self, value):
        warnings.warn("You should set the Lipschitz constant with calculate_Lipschitz().")
        if isinstance(value, (Number,)) and value >= 0:
            self._L = value
        else:
            raise TypeError('The Lipschitz constant is a real positive number')

    def calculate_Lipschitz(self):
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        try:
            self._L = 2 * np.abs(self.c) * (self.A.norm()**2)
        except AttributeError as ae:
            if self.A.is_linear():
                Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                self._L = 2 * np.abs(self.c) * (Anorm*Anorm)
            else:
                warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, ae))
        except NotImplementedError as noe:
            warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, noe))
        if self.weight is not None:
                self._L *= self.weight_norm
    @property
    def weight_norm(self):
        if self.weight is not None:
            if self._weight_norm is None:
                D = DiagonalOperator(self.weight)
                self._weight_norm = D.norm()
        else:
            self._weight_norm = 1.0
        return self._weight_norm

    def __rmul__(self, other):
        '''defines the right multiplication with a number'''

        if not isinstance (other, Number):
            raise NotImplemented
        constant = self.c * other

        return LeastSquares(A=self.A, b=self.b, c=constant, weight=self.weight)
