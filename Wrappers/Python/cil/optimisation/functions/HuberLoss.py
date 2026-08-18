#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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
# Martin Sæbye Carøe (Technical University of Denmark, DTU Compute)

import numpy as np
import warnings
from numbers import Number

from cil.optimisation.functions import Function
from cil.optimisation.operators import DiagonalOperator, LinearOperator

class HuberLoss(Function):
    r"""
    (Weighted) Huber loss

    For residual :math:`r = Ax - b`:

    .. math::
        \phi_\delta(r) =
        \begin{cases}
            0.5 * r^2 & \text{if } |r| \leq \delta \\
            \delta * (|r| - 0.5*\delta) & \text{otherwise}
        \end{cases}

    .. math::
    The (Weighted) Huber loss acts element wise on the residual :math:`r = Ax - b`. For small values of the residual it acts like a least squares loss and for larger values it acts like an absolute error loss. The idea is that the resulting loss is differentiable and stongly convex close to the minimum while also being robust to outliers far from the minimum.  A positive scalar :math:`\delta` controls the change point between the least squares and absolute error loss. 

    First define a function, acting on  :math:`d\in\mathbb{R}`

    .. math::
        \phi_\delta(d) =
        \begin{cases}
            0.5 * r^2 & \text{if } |d| \leq \delta \\
            \delta * (|d| - 0.5*\delta) & \text{otherwise.}
        \end{cases}
        
    This is then applied element wise to give the :code: `HuberLoss`:

    .. math::
        \mathtt{HuberLoss}_\delta(x) = c * \sum_i w_i \phi_\delta([Ax - b]_i).

    Note that :math:`c\in\mathbb{R}` is an optional scalar constant and :math:`w` is an optional weighting vector in range of the operator, :math:`A`, which defaults to a vector of 1s. 


    Parameters
    ----------
    A : LinearOperator
    b : Data, DataContainer
    huber_delta : float
        Transition point between L2 and L1 behaviour. Must be positive. 
    c : float, default 1.0
        Scaling constant
    weight : DataContainer, optional
        DataContainer with all positive elements of size of the range of operator A, default None
    """

    def __init__(self, A, b, huber_delta, c=1.0, weight=None):
        super(HuberLoss, self).__init__()

        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")

        self.A = A
        self.b = b
        self.c = c
        self.huber_delta = huber_delta

        self.weight = weight
        self._weight_norm = None

        if weight is not None:
            if (self.weight < 0).any():
                raise ValueError("weight contains negative values")

    def __call__(self, x):

        r = self.A.direct(x)
        r.subtract(self.b, out=r)

        abs_r = r.abs()

        # m = min(|r|, delta)
        m = abs_r.copy()
        m.minimum(self.huber_delta, out=m)

        # 0.5 * m^2
        val = m.power(2)
        val.multiply(0.5, out=val)

        # delta * (|r| - m)
        lin = abs_r.copy()
        lin.subtract(m, out=lin)
        lin.multiply(self.huber_delta, out=lin)

        val.add(lin, out=val)

        if self.weight is not None:
            val.multiply(self.weight, out=val)

        return self.c * val.sum()



    def gradient(self, x, out=None):
        r"""
        Returns the gradient of the Huber loss.

        For the residual

        .. math::
            r = Ax - b,

        the derivative of the Huber function is

        .. math::
            \phi_\delta'(r) =
            \begin{cases}
                r & \text{if } |r| \leq \delta \\
                \delta \operatorname{sign}(r) & \text{otherwise}.
            \end{cases}

        Therefore the gradient with respect to :math:`x` is

        .. math::
            \nabla f(x) =
            cA^T\left(
            w \odot
            \phi_\delta'(Ax-b)
            \right),

        where :math:`w` denotes the optional weights and
        :math:`\odot` denotes element-wise multiplication. If no
        weights are supplied, :math:`w=1`.

        Parameters
        ----------
        x : DataContainer
            Point at which the gradient is evaluated.
        out : DataContainer, optional
            Container in which to store the result.

        Returns
        -------
        DataContainer
            The gradient of the Huber loss evaluated at ``x``.
        """
        if out is None:
            out = x * 0.0

        r = self.A.direct(x)
        r.subtract(self.b, out=r)

        abs_r = r.abs()

        # m = min(|r|, delta)
        m = abs_r.copy()
        m.minimum(self.huber_delta, out=m)

        # grad wrt residual: sign(r) * m
        grad_r = r.sign()
        grad_r.multiply(m, out=grad_r)

        if self.weight is not None:
            grad_r.multiply(self.weight, out=grad_r)

        self.A.adjoint(grad_r, out=out)
        out.multiply(self.c, out=out)

        return out



    @property
    def L(self):
        if self._L is None:
            self.calculate_Lipschitz()
        return self._L

    @L.setter
    def L(self, value):
        warnings.warn("You should set the Lipschitz constant with calculate_Lipschitz().")
        if isinstance(value, Number) and value >= 0:
            self._L = value
        else:
            raise TypeError("The Lipschitz constant must be non-negative")

    def calculate_Lipschitz(self):
        r"""
        Calculate the Lipschitz constant of the gradient.

        For the Huber function

        .. math::

            \max_r \phi_\delta''(r) = 1.

        Therefore, for

        .. math::

            f(x) =
            c \sum_i w_i\,\phi_\delta((Ax-b)_i),

        the Hessian satisfies

        .. math::

            \nabla^2 f(x)
            =
            c\,A^T W D(x) A,

        where :math:`D(x)` is a diagonal operator with entries
        :math:`\phi_\delta''((Ax-b)_i)`.

        It follows that a Lipschitz constant for the gradient is

        .. math::

            L = |c|\,\|A\|^2,

        or, in the weighted case,

        .. math::

            L = |c|\,\|W\|\,\|A\|^2,

        where :math:`W` is the diagonal operator defined by
        ``weight``.

        """
        try:
            self._L = np.abs(self.c) * (self.A.norm() ** 2)
        except AttributeError:
            if self.A.is_linear():
                Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                self._L = np.abs(self.c) * (Anorm * Anorm)
            else:
                warnings.warn(
                    f"{self.__class__.__name__} could not calculate Lipschitz Constant, "
                    "likely because it was unable to calculate the norm of operator A."
                )

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
        if not isinstance(other, Number):
            raise NotImplemented

        return HuberLoss(
            A=self.A,
            b=self.b,
            huber_delta=self.huber_delta,
            c=self.c * other,
            weight=self.weight
        )
