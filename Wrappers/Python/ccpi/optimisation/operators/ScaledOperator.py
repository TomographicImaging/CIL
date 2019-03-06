from ccpi.optimisation.operators import LinearOperator
from numbers import Number

class ScaledOperator(LinearOperator):
    def __init__(self, operator, scalar):
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar))
        self.scalar = scalar
        self.operator = operator
    def direct(self, x, out=None):
        return self.scalar * self.operator.direct(x, out=out)
    def adjoint(self, x, out=None):
        if self.operator.is_linear():
            return self.scalar * self.operator.adjoint(x, out=out)
