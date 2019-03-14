from numbers import Number
import numpy

class ScaledFunction(object):
    '''ScaledFunction

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
    def __init__(self, function, scalar):
        super(ScaledFunction, self).__init__()
        self.L = None
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        self.scalar = scalar
        self.function = function

    def __call__(self,x, out=None):
        return self.scalar * self.function(x)

    def call_adjoint(self, x, out=None):
        return self.scalar * self.function.call_adjoint(x, out=out)

    def convex_conjugate(self, x, out=None):
        return self.scalar * self.function.convex_conjugate(x, out=out)

    def proximal_conjugate(self, x, tau, out = None):
        '''TODO check if this is mathematically correct'''
        return self.function.proximal_conjugate(x, tau, out=out)

    def grad(self, x):
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use gradient instead''', DeprecationWarning)
        return self.gradient(x, out=None)

    def prox(self, x, tau):
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use proximal instead''', DeprecationWarning)
        return self.proximal(x, out=None)

    def gradient(self, x, out=None):
        return self.scalar * self.function.gradient(x, out=out)

    def proximal(self, x, tau, out=None):
        '''TODO check if this is mathematically correct'''
        return self.function.proximal(x, tau, out=out)
