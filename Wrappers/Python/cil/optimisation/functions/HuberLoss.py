import numpy as np
import warnings
from numbers import Number

from cil.optimisation.functions import Function
from cil.optimisation.operators import DiagonalOperator, LinearOperator
from cil.framework import DataContainer

class HuberLoss(Function):
    r"""
    (Weighted) Huber loss

    For residual r = Ax - b:

    phi_delta(r) =
        0.5 * r^2                  if |r| <= delta
        delta * (|r| - 0.5*delta)  otherwise

    Parameters
    ----------
    A : LinearOperator
    b : Data, DataContainer
    huber_delta : float
        Transition point between L2 and L1 behaviour
    c : float, default 1.0
        Scaling constant
    weight : DataContainer, optional
        Positive diagonal weights
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
                raise ValueError("Weight contains negative values")

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
        """
        Lipschitz constant of gradient.

        For Huber:
            max phi'' = 1
        so:
            L = c * ||A||^2
        (weighted: multiplied by ||W||)
        """
        try:
            self._L = np.abs(self.c) * (self.A.norm() ** 2)
        except AttributeError:
            if self.A.is_linear():
                Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                self._L = np.abs(self.c) * (Anorm * Anorm)
            else:
                warnings.warn(
                    f"{self.__class__.__name__} could not calculate Lipschitz Constant."
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
