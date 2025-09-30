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
from cil.framework import DataContainer
from cil.optimisation.operators import DiagonalOperator


class L2NormSquared(Function):

    r""" L2NormSquared function: :math:`F(x) = \| x\|^{2}_{2} = \underset{i}{\sum}x_{i}^{2}`

    Following cases are considered:

        a) :math:`F(x) = \|x\|^{2}_{2}`
        b) :math:`F(x) = \|x - b\|^{2}_{2}`

    Parameters
    ----------

    b:`DataContainer`, optional
        Translation of the function


    Note
    -----

    For case b) we can use :code:`F = L2NormSquared().centered_at(b)`, see *TranslateFunction*.

    Example
    -------

        >>> F = L2NormSquared()
        >>> F = L2NormSquared(b=b)
        >>> F = L2NormSquared().centered_at(b)

    """

    def __init__(self, **kwargs):
        super(L2NormSquared, self).__init__(L=2)
        self.b = kwargs.get('b', None)

    def __call__(self, x):
        y = x
        if self.b is not None:
            y = x - self.b
        try:
            return y.squared_norm()
        except AttributeError as ae:
            # added for compatibility with SIRF
            return (y.norm()**2)

    def gradient(self, x, out=None):
        r"""Returns the value of the gradient of the L2NormSquared function at x.

        Following cases are considered:

            a) :math:`F'(x) = 2x`
            b) :math:`F'(x) = 2(x-b)`
        """
        if self.b is None:
            return x.multiply(2, out=out)
        else:
            return x.sapyb(2, self.b, -2, out=out)

    def convex_conjugate(self, x):
        r"""Returns the value of the convex conjugate of the L2NormSquared function at x.

        Consider the following cases:

                a) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2}
                b) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} + \langle x^{*}, b\rangle

        """
        tmp = 0

        if self.b is not None:
            tmp = x.dot(self.b)

        return 0.25 * x.squared_norm() + tmp

    def proximal(self, x, tau, out=None):
        r"""Returns the value of the proximal operator of the L2NormSquared function at x.


        Consider the following cases:

                a) .. math:: \text{prox}_{\tau F}(x) = \frac{x}{1+2\tau}
                b) .. math:: \text{prox}_{\tau F}(x) = \frac{x-b}{1+2\tau} + b

        """

        mult = 1/(1+2*tau)

        if self.b is None:
            return x.multiply(mult, out=out)
        else:
            return x.sapyb(mult, self.b, (1-mult), out=out)


class WeightedL2NormSquared(Function):

    r""" WeightedL2NormSquared function: :math:`F(x) = \|x\|_{W,2}^2 = \Sigma_iw_ix_i^2 = \langle x, Wx\rangle = x^TWx`
    where :math:`W=\text{diag}(weight)` if `weight` is a `DataContainer` or :math:`W=\text{weight} I` if `weight` is a scalar.

    Parameters
    -----------
    **kwargs

    weight: a `scalar` or a `DataContainer` with the same shape as the intended domain of this `WeightedL2NormSquared` function
    b: a `DataContainer` with the same shape as the intended domain of this `WeightedL2NormSquared` function
        A shift so that the function becomes  :math:`F(x) = \| x-b\|_{W,2}^2 = \Sigma_iw_i(x_i-b_i)^2 = \langle x-b, W(x-b) \rangle = (x-b)^TW(x-b)`


    """

    def __init__(self, **kwargs):

        # Weight can be either a scalar or a DataContainer
        # Lispchitz constant L = 2 *||weight||

        self.weight = kwargs.get('weight', 1.0)
        self.b = kwargs.get('b', None)
        tmp_norm = 1.0
        self.tmp_space = self.weight*0.

        if isinstance(self.weight, DataContainer):
            self.operator_weight = DiagonalOperator(self.weight)
            tmp_norm = self.operator_weight.norm()
            self.tmp_space = self.operator_weight.domain_geometry().allocate()

            if (self.weight < 0).any():
                raise ValueError('Weight contains negative values')

        super(WeightedL2NormSquared, self).__init__(L=2 * tmp_norm)

    def __call__(self, x):
        self.operator_weight.direct(x, out=self.tmp_space)
        y = x.dot(self.tmp_space)

        if self.b is not None:
            self.operator_weight.direct(x - self.b, out=self.tmp_space)
            y = (x - self.b).dot(self.tmp_space)
        return y

    def gradient(self, x, out=None):
        r""" Returns the value of :math:`F'(x) = 2Wx` or, if `b` is defined,  :math:`F'(x) = 2W(x-b)`
        where :math:`W=\text{diag}(weight)` if `weight` is a `DataContainer` or :math:`\text{weight}I` if `weight` is a scalar.

        """

        if out is not None:
            out.fill(x)
            if self.b is not None:
                out -= self.b
            self.operator_weight.direct(out, out=out)
            out *= 2
            return out
        else:
            y = x
            if self.b is not None:
                y = x - self.b
            return 2*self.weight*y

    def convex_conjugate(self, x):
        r"""Returns the value of the convex conjugate of the WeightedL2NormSquared function at x."""
        return 0.25 * (x/self.weight.sqrt()).squared_norm() + (x.dot(self.b) if self.b is not None else 0)

    def proximal(self, x, tau, out=None):
        r"""Returns the value of the proximal operator of the WeightedL2NormSquared function at x."""
        if self.b is not None:
            ret = x.subtract(self.b, out=out)
            ret /= (1+2*tau*self.weight)
            ret += self.b
        else:
            ret = x.divide((1+2*tau*self.weight), out=out)
        return ret
