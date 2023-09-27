# -*- coding: utf-8 -*-
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

from numbers import Number
import numpy
import functools
import warnings


class Operator(object):
    """
    Operator that maps from a space X -> Y

    Parameters
    ----------

    domain_geometry : ImageGeometry or AcquisitionGeometry
        domain of the operator

    range_geometry : ImageGeometry or AcquisitionGeometry, optional, default None
        range of the operator 
    """

    def __init__(self, domain_geometry, **kwargs):

        self._norm = None
        self._domain_geometry = domain_geometry
        self._range_geometry = kwargs.get('range_geometry', None)

    def is_linear(self):
        '''Returns if the operator is linear'''
        return False

    def direct(self, x, out=None):
        '''Returns the application of the Operator on x'''
        raise NotImplementedError

    def norm(self, **kwargs):
        '''Returns the norm of the Operator. On first call the norm will be calculated using the operator's calculate_norm
        method. Subsequent calls will return the cached norm.

        Returns
        -------
        norm: positive:`float`
        '''

        if len(kwargs) != 0:
            warnings.warn('norm: the norm method does not use any parameters.\n\
                For LinearOperators you can use PowerMethod to calculate the norm with non-default parameters and use set_norm to set it')

        if self._norm is None:
            self._norm = self.calculate_norm()

        return self._norm

    def set_norm(self, norm=None):
        '''Sets the norm of the operator to a custom value.
        '''
        self._norm = norm

    def calculate_norm(self):
        '''Calculates the norm of the Operator'''
        raise NotImplementedError

    def range_geometry(self):
        '''Returns the range of the Operator: Y space'''
        return self._range_geometry

    def domain_geometry(self):
        '''Returns the domain of the Operator: X space'''
        return self._domain_geometry

    @property
    def domain(self):
        return self.domain_geometry()

    @property
    def range(self):
        return self.range_geometry()

    def __rmul__(self, scalar):
        '''Defines the multiplication by a scalar on the left

        returns a ScaledOperator'''
        return ScaledOperator(self, scalar)

    def compose(self, *other, **kwargs):
        # TODO: check equality of domain and range of operators
        # if self.operator2.range_geometry != self.operator1.domain_geometry:
        #    raise ValueError('Cannot compose operators, check domain geometry of {} and range geometry of {}'.format(self.operato1,self.operator2))

        return CompositionOperator(self, *other, **kwargs)

    def __add__(self, other):
        return SumOperator(self, other)

    def __mul__(self, scalar):
        return self.__rmul__(scalar)

    def __neg__(self):
        """ Return -self """
        return -1 * self

    def __sub__(self, other):
        """ Returns the subtraction of the operators."""
        return self + (-1) * other


class LinearOperator(Operator):
    """
    Linear operator that maps from a space X <-> Y

    Parameters
    ----------

    domain_geometry : ImageGeometry or AcquisitionGeometry
        domain of the operator

    range_geometry : ImageGeometry or AcquisitionGeometry, optional, default None
        range of the operator 
    """

    def __init__(self, domain_geometry, **kwargs):
        super(LinearOperator, self).__init__(domain_geometry, **kwargs)

    def is_linear(self):
        '''Returns if the operator is linear'''
        return True

    def adjoint(self, x, out=None):
        '''returns the adjoint/inverse operation

        only available to linear operators'''
        raise NotImplementedError

    @staticmethod
    def PowerMethod(operator, max_iteration=10, initial=None, tolerance=1e-5,  return_all=False, method='auto'):
        r"""Power method or Power iteration algorithm 

        The Power method computes the largest (dominant) eigenvalue of a matrix in magnitude, e.g.,
        absolute value in the real case and modulus in the complex case.        
        
        Parameters
        ----------

        operator: LinearOperator
        max_iteration: positive:`int`, default=10
            Number of iterations for the Power method algorithm.
        initial: DataContainer, default = None
            Starting point for the Power method.
        tolerance: positive:`float`, default = 1e-5
            Stopping criterion for the Power method. Check if two consecutive eigenvalue evaluations are below the tolerance.                    
        return_all: `boolean`, default = False
            Toggles the verbosity of the return
        method: `string` one of `auto`, `composed_with_adjoint` and `direct_only`, default = `auto` 
            The default `auto` lets the code choose the method, this can be specified with `direct_only` or `compose_with_adjoint`
            

        Returns
        -------
        dominant eigenvalue: positive:`float`
        number of iterations: positive:`int`
            Number of iterations run. Only returned if return_all is True.
        eigenvector: DataContainer
            Corresponding eigenvector of the dominant eigenvalue. Only returned if return_all is True.
        list of eigenvalues: :obj:`list`
            List of eigenvalues. Only returned if return_all is True.
        convergence: `boolean'
            Check on wether the difference between the last two iterations is less than tolerance. Only returned if return_all is True.
        
            
        Note
        -----
        The power method contains two different algorithms chosen by the `method` flag. 

        In the case `method="direct_only"`, for operator, :math:`A`, the power method computes the iterations 
        .. math::
          x_{k+1} = A (\frac{x_k}{\|x_{k}\|}) \; (*)

        initialised with a random vector :math:`x_0` and returning the largest (dominant) eigenvalue in magnitude given by :math:`\|x_k\|`. 

        In the case `method="composed_with_adjoint"`, the algorithm computes the largest (dominant) eigenvalue of :math:`A^{T}A` 
          returning the square root of this value, i.e. the iterations:
          .. math::
          x_{k+1} = A^*A (\frac{x_k}{\|x_{k}\|})

        and returning  :math:`\sqrt{\|x_k\|}`.

        The default flag is `method="auto"`, the algorithm checks to see if the `operator.domain_geometry() == operator.range_geometry()` and if so
        uses the method "direct_only" and if not the method "composed_with_adjoint".

        Examples
        --------    

        >>> M = np.array([[1.,0],[1.,2.]])
        >>> Mop = MatrixOperator(M)
        >>> Mop_norm = Mop.PowerMethod(Mop)
        >>> Mop_norm
        2.0000654846240296

        `PowerMethod` is called when we compute the norm of a matrix or a `LinearOperator`.

        >>> Mop_norm = Mop.norm()
        2.0005647295658866

        """
        
        
        allowed_methods = ["auto","direct_only","composed_with_adjoint"]
        
        if method not in allowed_methods:
            raise ValueError("The argument 'method' can be set to one of {0} got {1}".format(allowed_methods, method))

        apply_adjoint=True
        if method == "direct_only":
            apply_adjoint=False
        if method=="auto":
            try:
                geometries_match = operator.domain_geometry() == operator.range_geometry()

            except AssertionError:
                # catch AssertionError for SIRF objects https://github.com/SyneRBI/SIRF-SuperBuild/runs/5110228626?check_suite_focus=true#step:8:972
                    pass
            else:
                if geometries_match:
                    apply_adjoint=False
        

        if initial is None:
            x0 = operator.domain_geometry().allocate('random')
        else:
            x0 = initial.copy()

        y_tmp = operator.range_geometry().allocate()

        # Normalize first eigenvector
        x0_norm = x0.norm()
        x0 /= x0_norm

        # initial guess for dominant eigenvalue
        eig_old = 1.
        if return_all:
            eig_list = []
            convergence_check = True
        diff = numpy.finfo('d').max
        i = 0
        while (i < max_iteration and diff > tolerance):
            operator.direct(x0, out=y_tmp)

            if not apply_adjoint:
                # swap datacontainer references
                tmp = x0
                x0 = y_tmp
                y_tmp = tmp
            else:
                operator.adjoint(y_tmp, out=x0)

            # Get eigenvalue using Rayleigh quotient: denominator=1, due to normalization
            x0_norm = x0.norm()
            if x0_norm < tolerance:
                Warning(
                    'The operator has at least one zero eigenvector and is likely to be nilpotent')
                eig_new = 0.
                break
            x0 /= x0_norm

            eig_new = numpy.abs(x0_norm)
            if apply_adjoint:
                eig_new = numpy.sqrt(eig_new)
            diff = numpy.abs(eig_new - eig_old)
            if return_all:
                eig_list.append(eig_new)
            eig_old = eig_new
            i += 1

        if return_all and i == max_iteration:
            convergence_check = False

        if return_all:
            return eig_new, i, x0, eig_list, convergence_check
        else:
            return eig_new

    def calculate_norm(self):
        r""" Returns the norm of the LinearOperator calculated by the PowerMethod with default values.
                """
        return LinearOperator.PowerMethod(self, method="composed_with_adjoint")

    @staticmethod
    def dot_test(operator, domain_init=None, range_init=None, tolerance=1e-6, **kwargs):
        r'''Does a dot linearity test on the operator
        Evaluates if the following equivalence holds
        .. math::
          Ax\times y = y \times A^Tx
        :param operator: operator to test the dot_test
        :param range_init: optional initialisation container in the operator range 
        :param domain_init: optional initialisation container in the operator domain 
        :param seed: Seed random generator
        :type : int, default = 1
        :param tolerance: Check if the following expression is below the tolerance
        .. math:: 

            |Ax\times y - y \times A^Tx|/(\|A\|\|x\|\|y\| + 1e-12) < tolerance

        :type : float, default 1e-6
        :returns: boolean, True if the test is passed.       
        '''

        seed = kwargs.get('seed', 1)

        if range_init is None:
            y = operator.range_geometry().allocate('random', seed=seed + 10)
        else:
            y = range_init
        if domain_init is None:
            x = operator.domain_geometry().allocate('random', seed=seed)
        else:
            x = domain_init

        fx = operator.direct(x)
        by = operator.adjoint(y)
        a = fx.dot(y)
        b = by.dot(x).conjugate()

        # Check relative tolerance but normalised with respect to
        # operator, x and y norms and avoid zero division
        error = numpy.abs(a - b) / (operator.norm()*x.norm()*y.norm() + 1e-12)

        if error < tolerance:
            return True
        else:
            print('Left hand side  {}, \nRight hand side {}'.format(a, b))
            return False


class ScaledOperator(Operator):

    '''ScaledOperator

    A class to represent the scalar multiplication of an Operator with a scalar.
    It holds an operator and a scalar. Basically it returns the multiplication
    of the result of direct and adjoint of the operator with the scalar.
    For the rest it behaves like the operator it holds.

    :param operator: a Operator or LinearOperator
    :param scalar: a scalar multiplier

    Example:
       The scaled operator behaves like the following:

    .. code-block:: python

      sop = ScaledOperator(operator, scalar)
      sop.direct(x) = scalar * operator.direct(x)
      sop.adjoint(x) = scalar * operator.adjoint(x)
      sop.norm() = operator.norm()
      sop.range_geometry() = operator.range_geometry()
      sop.domain_geometry() = operator.domain_geometry()

    '''

    def __init__(self, operator, scalar, **kwargs):
        '''creator

        :param operator: a Operator or LinearOperator
        :param scalar: a scalar multiplier
        :type scalar: Number'''

        super(ScaledOperator, self).__init__(domain_geometry=operator.domain_geometry(),
                                             range_geometry=operator.range_geometry())
        if not isinstance(scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        self.scalar = scalar
        self.operator = operator

    def direct(self, x, out=None):
        '''direct method'''
        if out is None:
            tmp = self.operator.direct(x)
            tmp *= self.scalar
            return tmp
        else:
            self.operator.direct(x, out=out)
            out *= self.scalar

    def adjoint(self, x, out=None):
        '''adjoint method'''
        if self.operator.is_linear():
            if out is None:
                tmp = self.operator.adjoint(x)
                tmp *= self.scalar
                return tmp
            else:
                self.operator.adjoint(x, out=out)
                out *= self.scalar
        else:
            raise TypeError('Operator is not linear')

    def norm(self, **kwargs):
        '''norm of the operator'''
        return numpy.abs(self.scalar) * self.operator.norm(**kwargs)

    def is_linear(self):
        '''returns whether the operator is linear

        :returns: boolean '''
        return self.operator.is_linear()


###############################################################################
################   SumOperator  ###########################################
###############################################################################

class SumOperator(Operator):

    def __init__(self, operator1, operator2):

        self.operator1 = operator1
        self.operator2 = operator2

        # if self.operator1.domain_geometry() != self.operator2.domain_geometry():
        #     raise ValueError('Domain geometry of {} is not equal with domain geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))

        # if self.operator1.range_geometry() != self.operator2.range_geometry():
        #     raise ValueError('Range geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))

        self.linear_flag = self.operator1.is_linear() and self.operator2.is_linear()

        super(SumOperator, self).__init__(domain_geometry=self.operator1.domain_geometry(),
                                          range_geometry=self.operator1.range_geometry())

    def direct(self, x, out=None):

        if out is None:
            return self.operator1.direct(x) + self.operator2.direct(x)
        else:
            self.operator1.direct(x, out=out)
            out.add(self.operator2.direct(x), out=out)

    def adjoint(self, x, out=None):

        if self.linear_flag:
            if out is None:
                return self.operator1.adjoint(x) + self.operator2.adjoint(x)
            else:
                self.operator1.adjoint(x, out=out)
                out.add(self.operator2.adjoint(x), out=out)
        else:
            raise ValueError('No adjoint operation with non-linear operators')

    def is_linear(self):
        return self.linear_flag

    def calculate_norm(self):
        if self.is_linear():
            return LinearOperator.calculate_norm(self)

        return super().calculate_norm(self)

###############################################################################
################   Composition  ###########################################
###############################################################################


class CompositionOperator(Operator):

    def __init__(self, *operators, **kwargs):

        # get a reference to the operators
        self.operators = operators

        self.linear_flag = functools.reduce(lambda x, y: x and y.is_linear(),
                                            self.operators, True)
        # self.preallocate = kwargs.get('preallocate', False)
        self.preallocate = False
        if self.preallocate:
            self.tmp_domain = [op.domain_geometry().allocate()
                               for op in self.operators[:-1]]
            self.tmp_range = [op.range_geometry().allocate()
                              for op in self.operators[1:]]
            # pass

        # TODO address the equality of geometries
        # if self.operator2.range_geometry() != self.operator1.domain_geometry():
        #     raise ValueError('Domain geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))

        super(CompositionOperator, self).__init__(
            domain_geometry=self.operators[-1].domain_geometry(),
            range_geometry=self.operators[0].range_geometry())

    def direct(self, x, out=None):

        if out is None:
            # return self.operator1.direct(self.operator2.direct(x))
            # return functools.reduce(lambda X,operator: operator.direct(X),
            #                        self.operators[::-1][1:],
            #                        self.operators[-1].direct(x))
            if self.preallocate:
                pass
            else:
                for i, operator in enumerate(self.operators[::-1]):
                    if i == 0:
                        step = operator.direct(x)
                    else:
                        step = operator.direct(step)
                return step

        else:
            # tmp = self.operator2.range_geometry().allocate()
            # self.operator2.direct(x, out = tmp)
            # self.operator1.direct(tmp, out = out)

            # out.fill (
            #     functools.reduce(lambda X,operator: operator.direct(X),
            #                        self.operators[::-1][1:],
            #                        self.operators[-1].direct(x))
            # )

            # TODO this is a bit silly but will handle the pre allocation later
            if self.preallocate:

                for i, operator in enumerate(self.operators[::-1]):
                    if i == 0:
                        operator.direct(x, out=self.tmp_range[i])
                    elif i == len(self.operators) - 1:
                        operator.direct(self.tmp_range[i-1], out=out)
                    else:
                        operator.direct(
                            self.tmp_range[i-1], out=self.tmp_range[i])
            else:
                for i, operator in enumerate(self.operators[::-1]):
                    if i == 0:
                        step = operator.direct(x)
                    else:
                        step = operator.direct(step)
                out.fill(step)

    def adjoint(self, x, out=None):

        if self.linear_flag:

            if out is not None:
                # return self.operator2.adjoint(self.operator1.adjoint(x))
                # return functools.reduce(lambda X,operator: operator.adjoint(X),
                #                    self.operators[1:],
                #                    self.operators[0].adjoint(x))
                if self.preallocate:
                    for i, operator in enumerate(self.operators):
                        if i == 0:
                            operator.adjoint(x, out=self.tmp_domain[i])
                        elif i == len(self.operators) - 1:
                            step = operator.adjoint(
                                self.tmp_domain[i-1], out=out)
                        else:
                            operator.adjoint(
                                self.tmp_domain[i-1], out=self.tmp_domain[i])
                    return
                else:
                    for i, operator in enumerate(self.operators):
                        if i == 0:
                            step = operator.adjoint(x)
                        else:
                            step = operator.adjoint(step)
                    out.fill(step)

            else:
                if self.preallocate:
                    pass
                else:
                    for i, operator in enumerate(self.operators):
                        if i == 0:
                            step = operator.adjoint(x)
                        else:
                            step = operator.adjoint(step)

                    return step
        else:
            raise ValueError('No adjoint operation with non-linear operators')

    def is_linear(self):
        return self.linear_flag             
            

    def calculate_norm(self):
        '''Returns the norm of the CompositionOperator, that is the product of the norms
        of its operators.'''
        norm = 1.
        for operator in self.operators:
                norm *= operator.norm()
        return norm
