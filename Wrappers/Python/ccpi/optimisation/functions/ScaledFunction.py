from numbers import Number
import numpy

class ScaledFunction(object):
    '''ScaledFunction

    A class to represent the scalar multiplication of an Function with a scalar.
    It holds a function and a scalar. Basically it returns the multiplication
    of the product of the function __call__, convex_conjugate and gradient with the scalar.
    For the rest it behaves like the function it holds.

    Args:
       function (Function): a Function or BlockOperator
       scalar (Number): a scalar multiplier
    Example:
       The scaled operator behaves like the following:
       
    '''
    def __init__(self, function, scalar):
        super(ScaledFunction, self).__init__()
        self.L = None
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        self.scalar = scalar
        self.function = function

    def __call__(self,x, out=None):
        '''Evaluates the function at x '''
        return self.scalar * self.function(x)

    def convex_conjugate(self, x):
        '''returns the convex_conjugate of the scaled function '''
        # if out is None:
        #     return self.scalar * self.function.convex_conjugate(x/self.scalar)
        # else:
        #     out.fill(self.function.convex_conjugate(x/self.scalar))
        #     out *= self.scalar
        return self.scalar * self.function.convex_conjugate(x/self.scalar)

    def proximal_conjugate(self, x, tau, out = None):
        '''This returns the proximal operator for the function at x, tau
        
        TODO check if this is mathematically correct'''
        return self.function.proximal_conjugate(x, tau, out=out)

    def grad(self, x):
        '''Alias of gradient(x,None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use gradient instead''', DeprecationWarning)
        return self.gradient(x, out=None)

    def prox(self, x, tau):
        '''Alias of proximal(x, tau, None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use proximal instead''', DeprecationWarning)
        return self.proximal(x, out=None)

    def gradient(self, x, out=None):
        '''Returns the gradient of the function at x, if the function is differentiable'''
        return self.scalar * self.function.gradient(x, out=out)

    def proximal(self, x, tau, out=None):
        '''This returns the proximal operator for the function at x, tau
        
        TODO check if this is mathematically correct'''
        return self.function.proximal(x, tau, out=out)
