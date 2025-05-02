#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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

from abc import ABC, abstractmethod
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.utilities import ConstantStepSize, StepSizeRule
import numpy
import logging
from numbers import Real, Number
import warnings

log = logging.getLogger(__name__)


class ScalarMomentumCoefficient(ABC):
    '''Abstract base class for MomentumCoefficient objects. The `__call__` method of this class returns the momentum coefficient for the given iteration.

    The call method of the ScalarMomentumCoefficient returns a scalar gradient value. Given access to the algorithm object, the momentum coefficient can be a function of the algorithm state.

    The `apply_momentum_in_APGD" function,  is  used to update the solution in the APGD algorithm as follows: x_{k+1} = y_{k+1} + M(y_{k+1} - y_{k}). The momentum coefficient M is returned by the ScalarMomentumCoefficient object.


    '''

    def __init__(self):
        '''Initialises the momentum coefficient object.
        '''
        pass

    @abstractmethod
    def __call__(self, algorithm):
        '''Returns the momentum coefficient for the given iteration.

        Parameters
        ----------
        algorithm: CIL Algorithm
            The algorithm object.
        '''

        pass

    def apply_momentum_in_APGD(self, algorithm, out=None):
        '''Calculates the momentum cofficient, applies a scalar momentum update in the APGD algorithm and returns the next iterate .

        Parameters
        ----------
        algorithm: instantiated CIL Algorithm
            The algorithm object.
        out: DataContainer, default is None
            Object to contain the next iterate. 
        '''
        momentum = self.__call__(algorithm)
        return algorithm.y.sapyb(momentum, algorithm.x, 1.0, out=out)


class ConstantMomentum(ScalarMomentumCoefficient):

    '''MomentumCoefficient object that returns a constant momentum coefficient.

    Parameters
    ----------
    momentum: float
        The constant momentum coefficient.
    '''

    def __init__(self, momentum):
        self.momentum = momentum

    def __call__(self, algorithm):
        return self.momentum


class NesterovMomentum(ScalarMomentumCoefficient):
    '''MomentumCoefficient object that returns the Nesterov momentum coefficient.

    Starting with t=1, the Nesterov momentum coefficient is updated as follows: :math:`t_{k+1} = 0.5(1 + \sqrt(1 + 4(t_{k}^2)))`. The momentum coefficient is then returned as :math:`(t_{k}-1)/t_{k}`.

    '''

    def __init__(self):
        self.t = 1

    def __call__(self, algorithm):
        self.t_old = self.t
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        return (self.t_old-1)/self.t


class APGD(Algorithm):

    r""" The Accelerated Proximal Gradient Descent (APGD) algorithm is used to solve:

    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, and :math:`g` has a *simple* proximal operator.

    In each update, the algorithm computes:

    .. math:: y_{k+1} = \mathrm{prox}_{\alpha g}(y_{k} - \alpha\nabla f(y_{k}))

    where :math:`\alpha` is the :code:`step_size`.

    A momentum term is then added to the update. Currently, we have implemented options for a scalar momentum coefficient. In this case, the momentum term is added as follows:

    .. math:: x_{k+1} = y_{k+1} + M(y_{k+1} - y_{k}). 

    The default momentum coefficient is the Nesterov momentum coefficient which varies with each iteration. Users can also set a constant momentum coefficient or implement their own momentum coefficient using the :class:`cil.optimisation.algorithms.APGD.ScalarMomentumCoefficient` class.


    Parameters
    ----------
    initial : DataContainer
              Initial guess of ISTA. :math:`x_{0}`
    f : Function
        Differentiable function. If `None` is passed, the algorithm will use the ZeroFunction.
    g : Function or `None`
        Convex function with *simple* proximal operator. If `None` is passed, the algorithm will use the ZeroFunction.
    step_size : positive :obj:`float` or child class of :meth:`cil.optimisation.utilities.StepSizeRule`',  default = None
                Step size for the gradient step of APGD. If a float is passed, this is used as a constant step size.  If a child class of :meth:`cil.optimisation.utilities.StepSizeRule` is passed then its method :meth:`get_step_size` is called for each update. 
                The default :code:`step_size` is a constant :math:`\frac{1}{L}` or 1 if `f=None`.
    preconditioner: class with an `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditoner`. If None is passed then `self.gradient_update` will remain unmodified. 
    momentum : float or child class of :meth:`cil.optimisation.algorithms.APGD.ScalarMomentumCoefficient`, default is Nesterov Momentum. 
            Class with an internal function `apply_momentum_in_APGD` that takes in an initiated algorithm and returns the next iterate.  


    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.


    Note
    ----
    The APGD algorithm with (default) Nesterov momentum is equivalent to the FISTA algorithm. 


    """

    def __init__(self, initial, f, g, step_size=None, preconditioner=None, momentum=None,  **kwargs):

        super(APGD, self).__init__(**kwargs)

        self.set_up(initial=initial, f=f, g=g, step_size=step_size,
                    preconditioner=preconditioner, momentum=momentum)

    def set_up(self, initial, f, g, step_size, preconditioner, momentum):
        """Set up of the algorithm"""
        log.info("%s setting up", self.__class__.__name__)

        self.initial = initial
        self.y = initial.copy()
        self.x_old = initial.copy()
        self.x = initial.copy()
        self.gradient_update = initial.copy()

        if f is None:
            f = ZeroFunction()

        self.f = f

        if g is None:
            g = ZeroFunction()

        self.g = g

        if isinstance(f, ZeroFunction) and isinstance(g, ZeroFunction):
            raise ValueError(
                'You set both f and g to be the ZeroFunction and thus the iterative method will not update and will remain fixed at the initial value.')

        # set step_size
        if step_size is None:
            self.step_size_rule = ConstantStepSize(
                self._calculate_default_step_size())
        elif isinstance(step_size, Real):
            self.step_size_rule = ConstantStepSize(step_size)
        elif isinstance(step_size, StepSizeRule):
            self.step_size_rule = step_size
        else:
            raise TypeError(
                "step_size must be a real number or a child class of :meth:`cil.optimisation.utilities.StepSizeRule`")

        self.preconditioner = preconditioner
        self._set_momentum(momentum)

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def _set_momentum(self, momentum):

        if momentum is None:
            self._momentum = NesterovMomentum()
        else:
            if isinstance(momentum, Number):
                self._momentum = ConstantMomentum(momentum)
            elif isinstance(momentum, ScalarMomentumCoefficient):
                self._momentum = momentum
            else:
                raise TypeError(
                    "Momentum must be a number or a child class of ScalarMomentumCoefficient")

    def _calculate_default_step_size(self):
        """Calculate the default step size if a step size rule or step size is not provided 
        """
        return 1./self.f.L

    def update(self):
        r"""Performs a single iteration of APGD. For :math:`k\geq 1`:

        .. math::

            \begin{cases}
                x_{k} = \mathrm{prox}_{\alpha g}(y_{k} - \alpha\nabla f(y_{k}))\\
                y_{k+1} = x_{k} + M(x_{k} - x_{k-1})
            \end{cases}

        where :math:`\alpha` is the :code:`step_size` and :math:`M` is a scalar momentum coefficient.
        """

        self.f.gradient(self.y, out=self.gradient_update)

        if self.preconditioner is not None:
            self.preconditioner.apply(
                self, self.gradient_update, out=self.gradient_update)

        self._step_size = self.step_size_rule.get_step_size(self)

        self.y.sapyb(1., self.gradient_update, -self._step_size, out=self.y)

        self.g.proximal(self.y, self._step_size, out=self.x)

        self.x.subtract(self.x_old, out=self.y)

        self.momentum.apply_momentum_in_APGD(self, out=self.y)

    def _update_previous_solution(self):
        """ Swaps the references to current and previous solution based on the :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def update_objective(self):
        """ Updates the objective

        .. math:: f(x) + g(x)

        """
        self.loss.append(
            self.calculate_objective_function_at_point(self.x_old))

    def calculate_objective_function_at_point(self, x):
        """ Calculates the objective at a given point x

        .. math:: f(x) + g(x)

        Parameters
        ----------
        x : DataContainer

        """
        return self.f(x) + self.g(x)

    def get_output(self):
        " Returns the current solution. "
        return self.x_old

    @property
    def step_size(self):
        if isinstance(self.step_size_rule, ConstantStepSize):
            return self.step_size_rule.step_size
        else:
            try:
                return self._step_size
            except NameError:
                return NotImplementedError("Note the step-size is set by a step-size rule and could change with each iteration. After running one update step, a step-size will be returned")

    @property
    def momentum(self):
        return self._momentum

    def _provable_convergence_condition(self):
        if self.preconditioner is not None:
            raise NotImplementedError(
                "Can't check convergence criterion if a preconditioner is used ")

        if isinstance(self.step_size_rule, ConstantStepSize) and isinstance(self.momentum, NesterovMomentum):
            return self.step_size_rule.step_size <= 1./self.f.L
        else:
            raise TypeError(
                "Can't check convergence criterion for non-constant step size or non-Nesterov momentum coefficient")
