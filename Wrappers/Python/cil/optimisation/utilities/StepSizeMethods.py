#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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
# - CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from abc import ABC, abstractmethod
import numpy
from numbers import Number
import logging

log = logging.getLogger(__name__)


def _resolve_ladmm_step_sizes(tau, sigma, operator):
    """Fills in the LADMM default step sizes: ``sigma = 1.0`` and ``tau = sigma/||K||**2``."""
    if sigma is None:
        sigma = 1.0
    if tau is None:
        tau = sigma / operator.norm() ** 2
    _validate_ladmm_step_sizes(tau, sigma, operator)
    return tau, sigma


def _validate_ladmm_step_sizes(tau, sigma, operator):
    """Checks ``tau`` and ``sigma`` are positive numbers, or arrays matching the domain/range geometry."""
    for name, value, shape in (
            ("tau", tau, operator.domain_geometry().shape),
            ("sigma", sigma, operator.range_geometry().shape)):
        if isinstance(value, Number):
            if value <= 0:
                raise ValueError(
                    "The step-sizes of LADMM must be positive, got {0} = {1}.".format(name, value))
        elif hasattr(value, "shape"):
            if value.shape != shape:
                raise ValueError(
                    "The shape of {0} = {1} is not the same as the expected shape = {2}. "
                    "This step-size rule may not be compatible with LADMM.".format(name, value.shape, shape))
        else:
            raise ValueError(
                "The step-sizes of LADMM must be a positive number or an array-like object (e.g. a DataContainer, "
                "BlockDataContainer or numpy array) of the expected shape = {0}, got {1} = {2!r} of type {3}. "
                "This step-size rule may not be compatible with LADMM.".format(
                    shape, name, value, type(value).__name__))


class StepSizeRule(ABC):
    """
    Abstract base class for a step size rule. The abstract method, `get_step_size` takes in an algorithm and thus can access all parts of the algorithm (e.g. current iterate, current gradient, objective functions etc) and from this  should return a float as a step size.

    Notes
    -----
    Gradient-based algorithms (:class:`~cil.optimisation.algorithms.GD`, :class:`~cil.optimisation.algorithms.ISTA`,
    :class:`~cil.optimisation.algorithms.FISTA`) expect :meth:`get_step_size` to return a single scalar and need
    nothing else. Primal-dual algorithms (:class:`~cil.optimisation.algorithms.LADMM`) additionally require a
    ``get_initial_step_size(self, algorithm)`` method, called once during set-up, returning the initial
    ``(tau, sigma)`` pair; their :meth:`get_step_size` is called at the end of each iteration and returns the
    updated ``(tau, sigma)`` pair. The algorithms check for ``get_initial_step_size`` with ``hasattr`` and raise
    if an incompatible, gradient-only, rule is passed.
    """

    def __init__(self):
        '''Initialises the step size rule
        '''
        pass

    @abstractmethod
    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the calculated step size:float
        """
        pass


class ConstantStepSize(StepSizeRule):
    """
    Step-size rule that always returns a constant step-size.

    Parameters
    ----------
    step_size: float
        The step-size to be returned with each call.
    """

    def __init__(self, step_size):
        '''Initialises the constant step size rule

         Parameters:
         -------------
         step_size : float, the constant step size
        '''
        self.step_size = step_size

    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the calculated step size:float
        """
        return self.step_size


class ArmijoStepSizeRule(StepSizeRule):

    r""" Applies the Armijo rule to calculate the step size (step_size).

    The Armijo rule runs a while loop to find the appropriate step_size by starting from a very large number (`alpha`). The step_size is found by reducing the step size (by a factor `beta`) in an iterative way until a certain criterion is met. To avoid infinite loops, we add a maximum number of times (`max_iterations`) the while loop is run.

    Reference
    ---------
    - Algorithm 3.1 in Nocedal, J. and Wright, S.J. eds., 1999. Numerical optimization. New York, NY: Springer New York. https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

    - https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080


    Parameters
    ----------
    alpha: float, optional, default=1e6
        The starting point for the step size iterations
    beta: float between 0 and 1, optional, default=0.5
        The amount the step_size is reduced if the criterion is not met
    max_iterations: integer, optional, default is numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
        The maximum number of iterations to find a suitable step size
    warmstart: Boolean, default is True
        If `warmstart = True` the initial step size at each Armijo iteration is the calculated step size from the last iteration. If `warmstart = False` at each  Armijo iteration, the initial step size is reset to the original, large `alpha`.
        In the case of *well-behaved* convex functions, `warmstart = True` is likely to be computationally less expensive. In the case of non-convex functions, or particularly tricky functions, setting `warmstart = False` may be beneficial.

    """

    def __init__(self, alpha=1e6, beta=0.5, max_iterations=None, warmstart=True):
        '''Initialises the step size rule
        '''

        self.alpha_orig = alpha
        self.alpha = alpha
        self.beta = beta 
        self.max_iterations = max_iterations
        if self.max_iterations is None:
            self.max_iterations = numpy.ceil(2 * numpy.log10(self.alpha_orig) / numpy.log10(2))

        self.warmstart=warmstart

    def get_step_size(self, algorithm):
        """
        Applies the Armijo rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        k = 0
        if not self.warmstart:
            self.alpha = self.alpha_orig

        f_x = algorithm.calculate_objective_function_at_point(algorithm.solution)

        self.x_armijo = algorithm.solution.copy()

        log.debug("Starting Armijo backtracking with initial step size: %f", self.alpha)

        while k < self.max_iterations:

            algorithm.gradient_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.solution.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.calculate_objective_function_at_point(self.x_armijo)
            sqnorm = algorithm.gradient_update.squared_norm()
            if f_x_a - f_x <= - (self.alpha/2) * sqnorm:
                break
            k += 1.
            self.alpha *= self.beta

        log.info("Armijo rule took %d iterations to find step size", k)

        if k == self.max_iterations:
            raise ValueError(
                'Could not find a proper step_size in {} loops. Consider increasing alpha or max_iterations.'.format(self.max_iterations))

        return self.alpha


class BarzilaiBorweinStepSizeRule(StepSizeRule):

    r""" Applies the Barzilai- Borwein rule to calculate the step size (step_size).

    Let :math:`\Delta x=x_k-x_{k-1}` and :math:`\Delta g=g_k-g_{k-1}`. Where :math:`x_k` is the :math:`k` th iterate (current solution after iteration :math:`k` ) and :math:`g_k` is the gradient calculation in the :math:`k` th iterate, found in :code:`algorithm.gradient_update`.  A Barzilai-Borwein (BB) iteration is :math:`x_{k+1}=x_k-\alpha_kg_k` where the step size :math:`\alpha _k` is either

    - :math:`\alpha_k^{LONG}=\frac{\Delta x\cdot\Delta x}{\Delta x\cdot\Delta g}`, or

    - :math:`\alpha_k^{SHORT}=\frac{\Delta x \cdot\Delta g}{\Delta g \cdot\Delta g}`.

    Where the operator :math:`\cdot` is the standard inner product between two vectors.

    This is suitable for use with gradient based iterative methods where the calculated gradient is stored as `algorithm.gradient_update`.

    Parameters
    ----------
    initial: float, greater than zero
        The step-size for the first iteration. We recommend something of the order :math:`1/f.L` where :math:`f` is the (differentiable part of) the objective you wish to minimise.
    mode: One of 'long', 'short' or 'alternate', default is 'short'.
        This calculates the step-size based on the LONG, SHORT or alternating between the two, starting with short.
    stabilisation_param: 'auto', float or 'off', default is 'auto'
        In order to add stability the step-size has an upper limit of :math:`\Delta/\|g_k\|` where by 'default', the `stabilisation_param`, :math:`\Delta` is  determined automatically to be the minimium of :math:`\Delta x` from the first 3 iterations. The user can also pass a fixed constant or turn "off" the stabilisation, equivalently passing `np.inf`.


    Reference
    ---------
    - Barzilai, Jonathan; Borwein, Jonathan M. (1988). "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis. 8: 141–148, https://doi.org/10.1093/imanum/8.1.141

    - Burdakov, O., Dai, Y. and Huang, N., 2019. STABILIZED BARZILAI-BORWEIN METHOD. Journal of Computational Mathematics, 37(6). https://doi.org/10.4208/jcm.1911-m2019-0171

    - https://en.wikipedia.org/wiki/Barzilai-Borwein_method
    """

    def __init__(self, initial, mode='short', stabilisation_param="auto"):
        '''Initialises the step size rule
        '''

        self.mode=mode
        if self.mode == 'short':
            self.is_short = True
        elif self.mode == 'long' or self.mode == 'alternate':
            self.is_short = False
        else:
            raise ValueError('Mode should be chosen from "long", "short" or "alternate". ')

        self.store_grad=None
        self.store_x=None
        self.initial=initial
        if stabilisation_param == 'auto':
            self.adaptive = True
            stabilisation_param = numpy.inf
        elif stabilisation_param == "off":
            self.adaptive = False
            stabilisation_param = numpy.inf
        elif ( isinstance(stabilisation_param, Number) and stabilisation_param >=0):
            self.adaptive = False
        else:
            raise TypeError(" The stabilisation_param should be 'auto', a positive number or 'off'")
        self.stabilisation_param=stabilisation_param



    def get_step_size(self, algorithm):
        """
        Applies the B-B rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        #For the first iteration we use an initial step size because the BB step size requires a previous iterate.
        if self.store_x is None:
            self.store_x=algorithm.x.copy() # We store the last iterate in order to calculate the BB step size
            self.store_grad=algorithm.gradient_update.copy()# We store the last gradient in order to calculate the BB step size
            return self.initial

        gradient_norm = algorithm.gradient_update.norm()
        #If the gradient is zero, gradient based algorithms will not update and te step size calculation will divide by zero so we stop iterations.
        if gradient_norm < 1e-8:
            raise StopIteration

        algorithm.x.subtract(self.store_x, out=self.store_x)
        algorithm.gradient_update.subtract(self.store_grad, out=self.store_grad)
        if self.is_short:
                ret = (self.store_x.dot(self.store_grad))/ (self.store_grad.dot(self.store_grad))
        else:
            ret = (self.store_x.dot(self.store_x))/ (self.store_x.dot(self.store_grad))


        #This computes the default stabilisation parameter, using the first three iterations
        if (algorithm.iteration <=3 and self.adaptive):
            self.stabilisation_param = min(self.stabilisation_param, self.store_x.norm() )

        # Computes the step size as the minimum of the ret, above, and :math:`\Delta/\|g_k\|` ignoring any NaN values.
        ret = numpy.nanmin( numpy.array([ret, self.stabilisation_param/gradient_norm]))

        # We store the last iterate and gradient in order to calculate the BB step size
        self.store_x.fill(algorithm.x)
        self.store_grad.fill(algorithm.gradient_update)

        if self.mode == "alternate":
            self.is_short =  not self.is_short

        return ret


class LADMMConstantStepSize(StepSizeRule):
    r"""Step-size rule returning a constant ``(tau, sigma)`` pair for :class:`~cil.optimisation.algorithms.LADMM`.

    Defaults are :math:`\sigma = 1` and :math:`\tau = \sigma/\|K\|^{2}`.

    Parameters
    ----------
    step_size : list or tuple of length two, default=[None, None]
        The primal and dual step sizes :math:`(\tau, \sigma)`. Either entry may be ``None``, in which case the
        default above is used.
    """

    def __init__(self, step_size=[None, None]):
        if len(step_size) != 2:
            raise ValueError(
                "step_size should be a list or tuple of length two, step_size = {}".format(step_size))
        self.tau = step_size[0]
        self.sigma = step_size[1]

    def get_initial_step_size(self, algorithm):
        """Returns the initial ``(tau, sigma)``, filling in any defaults from the operator norm."""
        self.tau, self.sigma = _resolve_ladmm_step_sizes(
            self.tau, self.sigma, algorithm.operator)
        return self.tau, self.sigma

    def get_step_size(self, algorithm):
        """Returns the unchanged ``(tau, sigma)``."""
        return self.tau, self.sigma


class LADMMAdaptiveStepSizeSRA(StepSizeRule):
    r"""Adaptive penalty selection for :class:`~cil.optimisation.algorithms.LADMM` by spectral radius approximation.

    The LADMM penalty parameter :math:`\rho` rescales the step sizes, :math:`(\tau, \sigma) = (\tau_{0}/\rho, \sigma_{0}/\rho)`
    This rule re-estimates :math:`\rho` every ``update_interval``
    iterations as the ratio :math:`\sqrt{p/q}` based on the primal and dual residuals :math:`p` and :math:`q` defined in Lozenski et al. (2026). 
    If either :math:`p` or :math:`q` is non-positive, the penalty is multiplied or divided by a multiplicative safeguard.

    Parameters
    ----------
    initial_step_size : list or tuple of length two, default=[None, 0.9]
        The base step sizes :math:`(\tau_{0}, \sigma_{0})`. Either entry may be ``None``, in which case
        :math:`\sigma_{0} = 1` and :math:`\tau_{0} = \sigma_{0}/\|K\|^{2}`. :math:`\sigma_{0}` must not exceed 1
        and :math:`\tau_{0}\|K\|^{2}` must not exceed 1, else the residual quantities are not non-negative.
    initial_penalty : float, positive, default=1.0
        The initial penalty :math:`\rho`.
    update_interval : int, positive, default=5
        Number of iterations between penalty updates.
    penalty_incr, penalty_decr : float, greater than 1, default=10.
        Multiplicative safeguards used when one of the two residual quantities is non-positive.

    Reference
    ---------
    Lozenski, L., McCann, M. T., Wohlberg, B. An Adaptive Multiparameter Penalty Selection Method for
    Multiconstraint and Multiblock ADMM. IEEE Open Journal of Signal Processing, 7, 410-427 (2026).
    https://doi.org/10.1109/OJSP.2026.3664275
    """

    def __init__(self, initial_step_size=[None, 0.9], initial_penalty=1.0,
                 update_interval=5, penalty_incr=10., penalty_decr=10.):
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, initial_step_size = {}".format(
                    initial_step_size))
        if not (isinstance(initial_penalty, Number) and initial_penalty > 0):
            raise ValueError(
                "initial_penalty must be a positive number, got initial_penalty = {}".format(initial_penalty))
        if not (isinstance(update_interval, int) and update_interval > 0):
            raise ValueError(
                "update_interval must be a positive integer, got update_interval = {}".format(update_interval))
        for name, value in (("penalty_incr", penalty_incr), ("penalty_decr", penalty_decr)):
            if not (isinstance(value, Number) and value > 1):
                raise ValueError("{0} must be a number greater than 1, got {0} = {1}".format(name, value))

        self.tau0 = initial_step_size[0]
        self.sigma0 = initial_step_size[1]
        self.rho = initial_penalty
        self.update_interval = update_interval
        self.penalty_incr = penalty_incr
        self.penalty_decr = penalty_decr

        self._x_prev = None
        self._x_prevprev = None
        self._z_prev = None
        self._z_prevprev = None

    def get_initial_step_size(self, algorithm):
        """Resolves the base step sizes, checks the metric preconditions, and returns ``(tau0/rho, sigma0/rho)``."""
        self.tau0, self.sigma0 = _resolve_ladmm_step_sizes(
            self.tau0, self.sigma0, algorithm.operator)

        if not isinstance(self.sigma0, Number) or not isinstance(self.tau0, Number):
            raise ValueError(
                "LADMMAdaptiveStepSizeSRA requires scalar step sizes, got tau0 = {0!r}, sigma0 = {1!r}.".format(
                    self.tau0, self.sigma0))
        if self.sigma0 > 1:
            raise ValueError(
                "LADMMAdaptiveStepSizeSRA requires sigma <= 1, got sigma = {}.".format(self.sigma0))
        if self.tau0 * algorithm.operator.norm() ** 2 > 1:
            raise ValueError(
                "LADMMAdaptiveStepSizeSRA requires tau*||K||**2 <= 1, got tau = {0} and ||K|| = {1}.".format(
                    self.tau0, algorithm.operator.norm()))

        # seed the iterate history with the initial point, so the first update can use it
        self._x_prev = algorithm.x.copy()
        self._z_prev = algorithm.z.copy()

        return self.tau0 / self.rho, self.sigma0 / self.rho

    def get_step_size(self, algorithm):
        """Updates the penalty if this is an update iteration, and returns ``(tau0/rho, sigma0/rho)``."""
        x = algorithm.x
        z = algorithm.z

        if (self._x_prevprev is not None
                and algorithm.iteration % self.update_interval == 1):
            self._update_penalty(algorithm, x, z)

        self._x_prevprev = self._x_prev
        self._z_prevprev = self._z_prev
        self._x_prev = x.copy()
        self._z_prev = z.copy()

        return self.tau0 / self.rho, self.sigma0 / self.rho

    def _update_penalty(self, algorithm, x, z):
        """Applies one spectral radius approximation update to ``self.rho``."""
        operator = algorithm.operator
        # algorithm.tmp_dir holds K x on exit from LADMM.update()
        primal_residual = algorithm.tmp_dir - z

        dx = x - self._x_prev
        dz = z - self._z_prev
        d2x = x - 2 * self._x_prev + self._x_prevprev
        d2z = z - 2 * self._z_prev + self._z_prevprev

        p2 = 3 * primal_residual.norm() ** 2
        p2 += 2 * (1 / self.sigma0 - 1) * d2z.norm() ** 2
        p2 += (1 / self.tau0) * d2x.norm() ** 2
        p2 -= operator.direct(d2x).norm() ** 2

        q2 = (2 / self.sigma0 - 1) * dz.norm() ** 2
        q2 += (1 / self.tau0) * dx.norm() ** 2
        q2 -= operator.direct(dx).norm() ** 2

        if p2 > 0 and q2 > 0:
            self.rho *= (p2 / q2) ** 0.5
        elif p2 <= 0 < q2:
            self.rho /= self.penalty_decr
        elif q2 <= 0 < p2:
            self.rho *= self.penalty_incr
