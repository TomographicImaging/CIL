from numbers import Number
import numpy
from ccpi.optimisation.operators import ScaledOperator
import functools

class BlockScaledOperator(ScaledOperator):
    '''ScaledOperator

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
    def __init__(self, operator, scalar, shape=None):
        if shape is None:
            shape = operator.shape
        
        if isinstance(scalar, (list, tuple, numpy.ndarray)):
            size = functools.reduce(lambda x,y:x*y, shape, 1)
            if len(scalar) != size:
                raise ValueError('Scalar and operators size do not match: {}!={}'
                .format(len(scalar), len(operator)))
            self.scalar = scalar[:]
            print ("BlockScaledOperator ", self.scalar)
        elif isinstance (scalar, Number):
            self.scalar = scalar
        else:
            raise TypeError('expected scalar to be a number of an iterable: got {}'.format(type(scalar)))
        self.operator = operator
        self.shape = shape
    def direct(self, x, out=None):
        print ("BlockScaledOperator self.scalar", self.scalar)
        #print ("self.scalar", self.scalar[0]* x.get_item(0).as_array())
        return self.scalar * (self.operator.direct(x, out=out))
    def adjoint(self, x, out=None):
        if self.operator.is_linear():
            return self.scalar * self.operator.adjoint(x, out=out)
        else:
            raise TypeError('Operator is not linear')
    def calculate_norm(self):
        return numpy.abs(self.scalar) * self.operator.norm()
    def range_geometry(self):
        return self.operator.range_geometry()
    def domain_geometry(self):
        return self.operator.domain_geometry()
    @property
    def T(self):
        '''Return the transposed of self'''
        #print ("transpose before" , self.shape)
        #shape = (self.shape[1], self.shape[0])
        ##self.shape = shape
        ##self.operator.shape = shape
        #print ("transpose" , shape)
        #return self
        return type(self)(self.operator.T, self.scalar)