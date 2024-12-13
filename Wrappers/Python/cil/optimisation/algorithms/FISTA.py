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

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.utilities import ConstantStepSize, StepSizeRule
import numpy
import logging
from numbers import Real, Number
import warnings

log = logging.getLogger(__name__)


class ISTA(Algorithm):

    r"""Iterative Shrinkage-Thresholding Algorithm (ISTA), see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.

    .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

    is used to solve

    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha^{k}`
    is the :code:`step_size`.

    Note
    ----

    For a constant step size, :math:`\alpha`, convergence of ISTA
    is guaranteed if

    .. math:: \alpha\in(0, \frac{2}{L}),

    where :math:`L` is the Lipschitz constant of :math:`f`, see :cite:`CombettesValerie`.

    Parameters
    ----------

    initial : DataContainer
              Initial guess of ISTA. :math:`x_{0}`
    f : Function
        Differentiable function. If `None` is passed, the algorithm will use the ZeroFunction.
    g : Function or `None`
        Convex function with *simple* proximal operator. If `None` is passed, the algorithm will use the ZeroFunction.
    step_size : positive :obj:`float` or child class of :meth:`cil.optimisation.utilities.StepSizeRule`',  default = None
                Step size for the gradient step of ISTA. If a float is passed, this is used as a constant step size.  If a child class of :meth:`cil.optimisation.utilities.StepSizeRule`' is passed then it's method `get_step_size` is called for each update. 
                The default :code:`step_size` is a constant :math:`\frac{0.99*2}{L}` or 1 if `f=None`.
     preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditoner`. If None is passed  then `self.gradient_update` will remain unmodified. 


    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.

    Note
    -----
    If the function `g` is set to `None` or to the `ZeroFunction` then the ISTA algorithm is equivalent to Gradient Descent.

    If the function `f` is set to `None` or to the `ZeroFunction` then the ISTA algorithm is equivalent to a Proximal Point Algorithm.

    Examples
    --------

    .. math:: \underset{x}{\mathrm{argmin}}\|A x - b\|^{2}_{2}

    >>> f = LeastSquares(A, b=b, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> ista = ISTA(initial = ig.allocate(), f = f, g = g)
    >>> ista.run()


    See also
    --------

    :class:`.FISTA`
    :class:`.GD`


    """

    def _provable_convergence_condition(self):
        if self.preconditioner is not None:
            raise NotImplementedError(
                "Can't check convergence criterion if a preconditioner is used ")

        if isinstance(self.step_size_rule, ConstantStepSize):
            return self.step_size_rule.step_size <= 0.99*2.0/self.f.L
        else:
            raise TypeError(
                "Can't check convergence criterion for non-constant step size")

    @property
    def step_size(self):
        if isinstance(self.step_size_rule, ConstantStepSize):
            return self.step_size_rule.step_size
        else:
            warnings.warn(
                "Note the step-size is set by a step-size rule and could change wit each iteration")
            return self.step_size_rule.get_step_size()

    # Set default step size

    def _calculate_default_step_size(self):
        """ Calculates the default step size if a step size rule or a step size is not provided. 
        """

        
        return 0.99*2.0/self.f.L

        

    def __init__(self, initial, f, g, step_size=None, preconditioner=None, **kwargs):

        super(ISTA, self).__init__(**kwargs)
        self._step_size = step_size
        self.set_up(initial=initial, f=f, g=g, step_size=step_size,
                    preconditioner=preconditioner, **kwargs)

    def set_up(self, initial, f, g, step_size, preconditioner, **kwargs):
        """Set up of the algorithm"""
        log.info("%s setting up", self.__class__.__name__)
        # set up ISTA
        self.initial = initial
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
        
        self.preconditioner = preconditioner

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def update(self):
        r"""Performs a single iteration of ISTA

        .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

        """

        # gradient step
        self.f.gradient(self.x_old, out=self.gradient_update)
        if self.preconditioner is not None:
            self.preconditioner.apply(
                self, self.gradient_update, out=self.gradient_update)

        try:
            step_size = self.step_size_rule.get_step_size(self)
        except NameError:
            raise NameError(msg='`step_size` must be `None`, a real float or a child class of :meth:`cil.optimisation.utilities.StepSizeRule`')

        self.x_old.sapyb(1., self.gradient_update, -step_size, out=self.x_old)

        # proximal step
        self.g.proximal(self.x_old, step_size, out=self.x)

    def _update_previous_solution(self):
        """ Swaps the references to current and previous solution based on the :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def get_output(self):
        " Returns the current solution. "
        return self.x_old

    def update_objective(self):
        """ Updates the objective

        .. math:: f(x) + g(x)

        """
        self.loss.append(self.calculate_objective_function_at_point(self.x_old))

    def calculate_objective_function_at_point(self, x):
        """ Calculates the objective at a given point x

        .. math:: f(x) + g(x)
        
        Parameters
        ----------
        x : DataContainer
        
        """
        return self.f(x) + self.g(x)

class FISTA(ISTA):

    r"""Fast Iterative Shrinkage-Thresholding Algorithm (FISTA), see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.

    .. math::

        \begin{cases}
                x_{k} = \mathrm{prox}_{\alpha g}(y_{k} - \alpha\nabla f(y_{k}))\\
                t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
                y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k+1}}(x_{k} - x_{k-1})
        \end{cases}

    is used to solve

    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha`
    is the :code:`step_size`.

    Note that the above applies for :math:`k\geq 1`. :math:`x_{0}` and :math:`y_{0}` are initialised to `initial` and :math:`t_{1}=1`


    Parameters
    ----------

    initial : DataContainer
            Starting point of the algorithm
    f : Function
        Differentiable function.  If `None` is passed, the algorithm will use the ZeroFunction.
    g : Function or `None`
        Convex function with *simple* proximal operator. If `None` is passed, the algorithm will use the ZeroFunction.
    step_size : positive :obj:`float` or child class of :meth:`cil.optimisation.utilities.StepSizeRule`',  default = None
                Step size for the gradient step of ISTA. If a float is passed, this is used as a constant step size. If a child class of :meth:`cil.optimisation.utilities.StepSizeRule`' is passed then it's method `get_step_size` is called for each update. 
                The default :code:`step_size` is a constant :math:`\frac{1}{L}` or 1 if `f=None`.
    preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditoner`. If None is passed  then `self.gradient_update` will remain unmodified. 

    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.

    Note
    -----
    If the function `g` is set to `None` or to the `ZeroFunction` then the FISTA algorithm is equivalent to Accelerated Gradient Descent by Nesterov (:cite:`nesterov2003introductory` algorithm 2.2.9).

    If the function `f` is set to `None` or to the `ZeroFunction` then the FISTA algorithm is equivalent to Guler's First Accelerated Proximal Point Method  (:cite:`guler1992new` sec 2).

    Examples
    --------

    .. math:: \underset{x}{\mathrm{argmin}}\|A x - b\|^{2}_{2}


    >>> f = LeastSquares(A, b=b, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> fista = FISTA(initial = ig.allocate(), f = f, g = g)
    >>> fista.run()

    See also
    --------
    :class:`.FISTA`
    :class:`.GD`

    """

    def _calculate_default_step_size(self):
        """Calculate the default step size if a step size rule or step size is not provided 
        """
        return 1./self.f.L



    def _provable_convergence_condition(self):
        if self.preconditioner is not None:
            raise NotImplementedError(
                "Can't check convergence criterion if a preconditioner is used ")

        if isinstance(self.step_size_rule, ConstantStepSize):
            return self.step_size_rule.step_size <= 1./self.f.L
        else:
            raise TypeError(
                "Can't check convergence criterion for non-constant step size")

    def __init__(self, initial, f, g, step_size=None,  preconditioner=None, **kwargs):

        self.y = initial.copy()
        self.t = 1
        super(FISTA, self).__init__(initial=initial, f=f, g=g,
                                    step_size=step_size,  preconditioner=preconditioner, **kwargs)

    def update(self):
        r"""Performs a single iteration of FISTA. For :math:`k\geq 1`:

        .. math::

            \begin{cases}
                x_{k} = \mathrm{prox}_{\alpha g}(y_{k} - \alpha\nabla f(y_{k}))\\
                t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
                y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k+1}}(x_{k} - x_{k-1})
            \end{cases}

        """

        self.t_old = self.t

        self.f.gradient(self.y, out=self.gradient_update)

        if self.preconditioner is not None:
            self.preconditioner.apply(
                self, self.gradient_update, out=self.gradient_update)

        step_size = self.step_size_rule.get_step_size(self)

        self.y.sapyb(1., self.gradient_update, -step_size, out=self.y)

        self.g.proximal(self.y, step_size, out=self.x)

        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))

        self.x.subtract(self.x_old, out=self.y)
        self.y.sapyb(((self.t_old-1)/self.t), self.x, 1.0, out=self.y)
