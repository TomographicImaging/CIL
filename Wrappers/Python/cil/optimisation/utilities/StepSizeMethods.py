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
from copy import deepcopy
from numbers import Number
import logging
import numpy as np

log = logging.getLogger(__name__)


def _resolve_pdhg_step_sizes(tau, sigma, operator):
    """Validate user-supplied PDHG primal/dual step sizes and fill in defaults.

    Either, both or neither of ``tau``/``sigma`` may be provided. Scalars must be
    positive; array-like values must match the operator's domain (``tau``) or
    range (``sigma``) geometry. Any missing value is derived from the operator norm.

    Returns
    -------
    tuple
        ``(tau, sigma)`` with any ``None`` entries replaced by their defaults.
    """
    # Check acceptable values of the primal-dual step-sizes
    if tau is not None:
        if isinstance(tau, Number):
            if tau <= 0:
                raise ValueError(
                    "The step-sizes of PDHG must be positive, passed tau = {}".format(tau))
        elif tau.shape != operator.domain_geometry().shape:
            raise ValueError(" The shape of tau = {0} is not the same as the shape of the domain_geometry = {1}".format(
                tau.shape, operator.domain_geometry().shape))

    if sigma is not None:
        if isinstance(sigma, Number):
            if sigma <= 0:
                raise ValueError(
                    "The step-sizes of PDHG are positive, passed sigma = {}".format(sigma))
        elif sigma.shape != operator.range_geometry().shape:
            raise ValueError(" The shape of sigma = {0} is not the same as the shape of the range_geometry = {1}".format(
                sigma.shape, operator.range_geometry().shape))

    # Default sigma and tau step-sizes
    if tau is None and sigma is None:
        sigma = 1.0/operator.norm()
        tau = 1.0/operator.norm()
    elif tau is not None and sigma is not None:
        pass
    elif sigma is None and isinstance(tau, Number):
        sigma = 1./(tau*operator.norm()**2)
    elif tau is None and isinstance(sigma, Number):
        tau = 1./(sigma*operator.norm()**2)
    else:
        raise NotImplementedError(
            "If using arrays for sigma or tau both must arrays must be provided.")
    return tau, sigma


def _validate_pdhg_step_sizes(tau, sigma, operator):
    """Validate step sizes produced by a PDHG step-size rule.

    ``tau`` and ``sigma`` must each be a positive scalar or an array-like whose
    shape matches the operator's domain (``tau``) or range (``sigma``) geometry.
    Raises ``ValueError`` if the step sizes have the wrong type or shape, e.g.
    because an incompatible step-size rule was passed to PDHG.
    """
    for name, value, shape in (
            ("tau", tau, operator.domain_geometry().shape),
            ("sigma", sigma, operator.range_geometry().shape)):
        if value is None:
            raise ValueError(
                "The step-size rule returned {0} = None. This step-size rule is "
                "not compatible with PDHG.".format(name))
        if isinstance(value, Number):
            if value <= 0:
                raise ValueError(
                    "The step-sizes of PDHG must be positive, got {0} = {1}.".format(name, value))
        elif getattr(value, "shape", None) != shape:
            raise ValueError(
                "The shape of {0} = {1} is not the same as the expected shape = {2}. "
                "This step-size rule may not be compatible with PDHG.".format(
                    name, getattr(value, "shape", type(value)), shape))


def _validate_spdhg_step_sizes(tau, sigma, n_operators):
    """Validate step sizes produced by an SPDHG step-size rule.

    ``tau`` must be a positive scalar and ``sigma`` a list/array of positive
    numbers with one entry per operator. Raises ``ValueError`` otherwise, e.g.
    because an incompatible step-size rule was passed to SPDHG.
    """
    if not (isinstance(tau, Number) and tau > 0):
        raise ValueError(
            "The primal step-size tau of SPDHG must be a positive number, got tau = {0}. "
            "This step-size rule may not be compatible with SPDHG.".format(tau))
    if isinstance(sigma, Number) or len(sigma) != n_operators:
        raise ValueError(
            "The dual step-size sigma of SPDHG must be a list of {0} positive numbers "
            "(one per operator), got sigma = {1}. This step-size rule may not be "
            "compatible with SPDHG.".format(n_operators, sigma))
    if not all(isinstance(si, Number) and si > 0 for si in sigma):
        raise ValueError(
            "The dual step-sizes sigma of SPDHG must all be positive numbers, got sigma = {0}.".format(sigma))


def _spdhg_sigma_from_gamma(gamma, rho, norms):
    """Dual SPDHG step sizes from the ratio ``gamma`` and product parameter ``rho``."""
    return [gamma * rho / ni for ni in norms]


def _spdhg_tau_from_sigma(sigma, norms, prob_weights, rho):
    """Primal SPDHG step size consistent with the dual step sizes ``sigma``.

    Under the requirement that :math:`\\tau\\sigma_i\\|K_i\\|^2 \\leq \\rho p_i`, where :math:`p_i` is the probability weight for operator :math:`K_i`, 
    :math:`\\tau` is returned that satisfies this for all operators.
    """
    values = [rho * pi / (si * ni**2)
              for pi, ni, si in zip(prob_weights, norms, sigma)
              if ni > 0 and pi > 0]
    if not values:
        raise ValueError(
            "Could not compute an SPDHG primal step size from the given sigma. "
            "Every block had either a zero operator norm or a zero probability weight. "
            "Check the operator norms and probability weights.")
    return min(values)


class StepSizeRule(ABC):
    """
    Abstract base class for a step size rule. The abstract method, `get_step_size` takes in an algorithm and thus can access all parts of the algorithm (e.g. current iterate, current gradient, objective functions etc) and from this  should return a float as a step size.

    Notes
    -----
    There are two families of step-size rule in CIL:

    * **Gradient-based algorithms** (:class:`~cil.optimisation.algorithms.GD`,
      :class:`~cil.optimisation.algorithms.ISTA`, :class:`~cil.optimisation.algorithms.FISTA`)
      call :meth:`get_step_size` after each gradient calculation and expect a single scalar
      step size to be returned. These rules only need to implement :meth:`get_step_size`.
    * **Primal-dual algorithms** (:class:`~cil.optimisation.algorithms.PDHG`,
      :class:`~cil.optimisation.algorithms.SPDHG`) additionally require a
      ``get_initial_step_size(self, algorithm)`` method which is called once during set-up and
      returns the initial ``(tau, sigma)`` step-size pair. Their :meth:`get_step_size` is called
      at the end of every iteration and returns the updated ``(tau, sigma)`` pair for the next
      iteration. The algorithms check for the presence of ``get_initial_step_size`` (via
      ``hasattr``) and raise a clear error if an incompatible, gradient-only, rule is passed.
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
    max_iterations: integer, optional, default is np.ceil (2 * np.log10(alpha) / np.log10(2))
        The maximum number of iterations to find a suitable step size
    warmstart: Boolean, default is True
        If `warmstart = True` the initial step size at each Armijo iteration is the calculated step size from the last iteration. If `warmstart = False` at each  Armijo iteration, the initial step size is reset to the original, large `alpha`.
        In the case of *well-behaved* convex functions, `warmstart = True` is likely to be computationally less expensive. In the case of non-convex functions, or particularly tricky functions, setting `warmstart = False` may be beneficial.

    """

    def __init__(self, alpha=1e6, beta=0.5, max_iterations=None, warmstart=True):
        '''Initialises the step size rule
        '''

        self.alpha_orig = alpha
        if self.alpha_orig is None:  # Can be removed when alpha and beta are deprecated in GD
            self.alpha_orig = 1e6
        self.alpha = self.alpha_orig
        self.beta = beta
        if self.beta is None:  # Can be removed when alpha and beta are deprecated in GD
            self.beta = 0.5

        self.max_iterations = max_iterations
        if self.max_iterations is None:
            self.max_iterations = np.ceil(
                2 * np.log10(self.alpha_orig) / np.log10(2))

        self.warmstart = warmstart

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

        f_x = algorithm.calculate_objective_function_at_point(
            algorithm.solution)

        self.x_armijo = algorithm.solution.copy()

        log.debug(
            "Starting Armijo backtracking with initial step size: %f", self.alpha)

        while k < self.max_iterations:

            algorithm.gradient_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.solution.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.calculate_objective_function_at_point(
                self.x_armijo)
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

        self.mode = mode
        if self.mode == 'short':
            self.is_short = True
        elif self.mode == 'long' or self.mode == 'alternate':
            self.is_short = False
        else:
            raise ValueError(
                'Mode should be chosen from "long", "short" or "alternate". ')

        self.store_grad = None
        self.store_x = None
        self.initial = initial
        if stabilisation_param == 'auto':
            self.adaptive = True
            stabilisation_param = np.inf
        elif stabilisation_param == "off":
            self.adaptive = False
            stabilisation_param = np.inf
        elif (isinstance(stabilisation_param, Number) and stabilisation_param >= 0):
            self.adaptive = False
        else:
            raise TypeError(
                " The stabilisation_param should be 'auto', a positive number or 'off'")
        self.stabilisation_param = stabilisation_param

    def get_step_size(self, algorithm):
        """
        Applies the B-B rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        # For the first iteration we use an initial step size because the BB step size requires a previous iterate.
        if self.store_x is None:
            # We store the last iterate in order to calculate the BB step size
            self.store_x = algorithm.x.copy()
            # We store the last gradient in order to calculate the BB step size
            self.store_grad = algorithm.gradient_update.copy()
            return self.initial

        gradient_norm = algorithm.gradient_update.norm()
        # If the gradient is zero, gradient based algorithms will not update and te step size calculation will divide by zero so we stop iterations.
        if gradient_norm < 1e-8:
            raise StopIteration

        algorithm.x.subtract(self.store_x, out=self.store_x)
        algorithm.gradient_update.subtract(
            self.store_grad, out=self.store_grad)
        if self.is_short:
            ret = (self.store_x.dot(self.store_grad)) / \
                (self.store_grad.dot(self.store_grad))
        else:
            ret = (self.store_x.dot(self.store_x)) / \
                (self.store_x.dot(self.store_grad))

        # This computes the default stabilisation parameter, using the first three iterations
        if (algorithm.iteration <= 3 and self.adaptive):
            self.stabilisation_param = min(
                self.stabilisation_param, self.store_x.norm())

        # Computes the step size as the minimum of the ret, above, and :math:`\Delta/\|g_k\|` ignoring any NaN values.
        ret = np.nanmin(np.array(
            [ret, self.stabilisation_param/gradient_norm]))

        # We store the last iterate and gradient in order to calculate the BB step size
        self.store_x.fill(algorithm.x)
        self.store_grad.fill(algorithm.gradient_update)

        if self.mode == "alternate":
            self.is_short = not self.is_short

        return ret


class PDHGStronglyConvexUpdate(StepSizeRule):
    r'''Updates step sizes (theta, sigma, tau) in the PDHG algorithm in the cases of primal or dual acceleration using the strongly convexity property.
            The case where both functions are strongly convex is not available at the moment.


            The PDHG algorithm can be accelerated if the functions :math:`f^{*}` and/or :math:`g` are strongly convex. In these cases, the step-sizes :math:`\sigma` and :math:`\tau` are updated using the :meth:`get_step_size` method. A function :math:`f` is strongly convex with constant :math:`\gamma>0` if

        .. math::

            f(x) - \frac{\gamma}{2}\|x\|^{2} \quad\mbox{ is convex. }


        * For instance the function :math:`\frac{1}{2}\|x\|^{2}_{2}` is :math:`\gamma` strongly convex for :math:`\gamma\in(-\infty,1]`. We say it is 1-strongly convex because it is the largest constant for which :math:`f - \frac{1}{2}\|\cdot\|^{2}` is convex.


        * The :math:`\|\cdot\|_{1}` norm is not strongly convex. For more information, see `Strongly Convex <https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions>`_.


        * If :math:`g` is strongly convex with constant :math:`\gamma` then the step-sizes :math:`\sigma`, :math:`\tau` and :math:`\theta` are updated as:


        .. math::
            :nowrap:

                \begin{aligned}

                    \theta_{n} & = \frac{1}{\sqrt{1 + 2\gamma\tau_{n}}}\\
                    \tau_{n+1} & = \theta_{n}\tau_{n}\\
                    \sigma_{n+1} & = \frac{\sigma_{n}}{\theta_{n}}

                \end{aligned}

        * If :math:`f^{*}` is strongly convex, we swap :math:`\sigma` with :math:`\tau`.



            Parameters
            -------------
            initial_step_size : list or tuple of length two, optional, default=(None, None)
                Initial values of the primal and dual step sizes ``(tau, sigma)``. Either, both or neither may be
                ``None``: any missing value is filled in from the operator norm exactly as in
                :class:`PDHGConstantStepSize`.
            gamma_g : positive :obj:`float`, optional, default=None
                Strongly convex constant if the function g is strongly convex. Allows primal acceleration of the PDHG algorithm.
            gamma_fconj : positive :obj:`float`, optional, default=None
                Strongly convex constant if the convex conjugate of f is strongly convex. Allows dual acceleration of the PDHG algorithm.

            Note
            ----
            This rule is also selected automatically when the (deprecated) ``gamma_g`` or
            ``gamma_fconj`` keyword arguments are passed directly to
            :class:`~cil.optimisation.algorithms.PDHG`. New code should construct and pass this
            rule explicitly via the ``step_size`` argument instead.
            '''

    def __init__(self, initial_step_size=(None, None), gamma_g=None, gamma_fconj=None):
        '''Initialises the step size rule'''

        if gamma_g is not None and gamma_fconj is not None:
            raise NotImplementedError(
                "PDHG strongly convex step size update not implemented for both primal and dual acceleration. Please choose only one of gamma_g or gamma_fconj.")

        if not (gamma_g is None or (isinstance(gamma_g, Number) and gamma_g > 0)):
            raise ValueError(
                "A positive float is expected for the strongly convex constant of the function g, {} is passed".format(gamma_g))
        self.gamma_g = gamma_g

        if not (gamma_fconj is None or (isinstance(gamma_fconj, Number) and gamma_fconj > 0)):
            raise ValueError(
                "A positive float is expected for the strongly convex constant of the convex conjugate of function f, {} is passed".format(gamma_fconj))
        self.gamma_fconj = gamma_fconj

        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, step_size = {}".format(initial_step_size))

    def get_initial_step_size(self, algorithm):
        """Sets the initial sigma and tau step-sizes for the PDHG algorithm. The step sizes can be either scalar or array-objects.

        The values come from ``initial_step_size``, where the user can set either, both or none. Values passed by the
        user are accepted as long as they are positive numbers, or array-like objects of the correct shape; any value
        left as ``None`` is derived from the operator norm.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to, used here for its operator.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        self.tau = self.initial_step_size[0]
        self.sigma = self.initial_step_size[1]
        self.tau, self.sigma = _resolve_pdhg_step_sizes(
            self.tau, self.sigma, algorithm.operator)
        return self.tau, self.sigma

    def get_step_size(self, algorithm):
        """
        Applies the PDHG strongly convex step size update to calculate the new primal and dual step sizes.

        The relaxation parameter :math:`\\theta` of ``algorithm`` is updated in place as part of the acceleration.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        # Update sigma and tau based on the strong convexity of G
        if self.gamma_g is not None:
            algorithm._theta = 1.0 / \
                np.sqrt(1 + 2 * self.gamma_g * algorithm._tau)
            self.tau *= algorithm._theta
            self.sigma /= algorithm._theta

        # Update sigma and tau based on the strong convexity of F
        # Following operations are reversed due to symmetry, sigma --> tau, tau -->sigma
        if self.gamma_fconj is not None:
            algorithm._theta = 1.0 / \
                np.sqrt(1 + 2 * self.gamma_fconj * algorithm._sigma)
            self.sigma *= algorithm._theta
            self.tau /= algorithm._theta

        return self.tau, self.sigma


class PDHGAdaptiveStepSize2013(StepSizeRule):
    r"""Adaptively updates the PDHG primal and dual step sizes using the backtracking and residual-balancing method of :cite:`goldstein2013adaptive`.

    This is a PDHG-compatible step-size rule: it provides an initial pair of step sizes via
    :meth:`get_initial_step_size` and then, at the end of every PDHG iteration,
    :meth:`get_step_size` updates :math:`\tau` and :math:`\sigma` in place based on the
    observed behaviour of the algorithm. It combines two mechanisms applied each iteration.

    **1. Backtracking.** Using the primal and dual increments
    :math:`\Delta x = x^{n+1} - x^{n}` and :math:`\Delta y = y^{n+1} - y^{n}`, a
    backtracking quantity is formed

    .. math::

        b = \frac{2\,\sigma\tau\,|\langle \Delta y, K\Delta x\rangle|}
                 {\gamma\,\sigma\,\|\Delta x\|^{2} + \gamma\,\tau\,\|\Delta y\|^{2}}.

    While :math:`b > 1` the step is unstable, so both step sizes are shrunk,
    :math:`\tau \leftarrow \tfrac{\beta}{b}\tau`, :math:`\sigma \leftarrow \tfrac{\beta}{b}\sigma`,
    the PDHG update is recomputed (via :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update`)
    and :math:`b` is re-evaluated. The loop repeats until :math:`b \le 1` or
    ``inner_iterations`` is reached.

    **2. Residual balancing.** The primal and dual residuals

    .. math::

        p = \left\| \tfrac{\Delta x}{\tau} - K^{*}\Delta y \right\|, \qquad
        d = \left\| \tfrac{\Delta y}{\sigma} - K\Delta x \right\|

    are compared using the balancing scale :math:`s` and band parameter :math:`\delta > 1`.
    If :math:`p < \tfrac{s}{\delta}\, d` the ratio is tilted towards the dual variable
    (:math:`\tau \leftarrow (1-\alpha)\tau`, :math:`\sigma \leftarrow \sigma/(1-\alpha)`);
    if :math:`s\,\delta\, d < p` it is tilted towards the primal variable
    (:math:`\tau \leftarrow \tau/(1-\alpha)`, :math:`\sigma \leftarrow (1-\alpha)\sigma`).
    Each time a tilt is applied the adaptation strength decays, :math:`\alpha \leftarrow \eta\,\alpha`,
    so the updates become progressively smaller. The product :math:`\sigma\tau` is left
    unchanged by the balancing step, preserving the stability guaranteed by the backtracking step.

    Parameters
    -------------
    initial_step_size : list of two positive :obj:`float`, optional, default=[10/algorithm.operator.norm(), 10/algorithm.operator.norm()]
        Initial values of the primal and dual step sizes used in the adaptive step size method.
    initial_alpha : :obj:`float` in :math:`(0,1)`, optional, default=0.95
        Initial value of the adaptation strength :math:`\alpha` controlling the size of the residual-balancing update.
        Each rebalance multiplies or divides :math:`\tau` by :math:`1-\alpha`, so :math:`\alpha` close to one gives a
        *large* rescaling (:math:`\alpha = 0.95` changes :math:`\tau` by a factor of 20) and :math:`\alpha` close to
        zero a gentle one.
    beta : positive :obj:`float`, optional, default=0.95
        The factor :math:`\beta` by which the step sizes are shrunk during backtracking (via :math:`\beta/b`).
    gamma : positive :obj:`float`, optional, default=0.9
        The convergence constant :math:`\gamma` appearing in the denominator of the backtracking quantity :math:`b`.
    delta : positive :obj:`float`, greater than one,  optional, default=1.5
        The band parameter :math:`\delta` setting how far apart the primal and dual residuals are allowed to drift before rebalancing.
    s : positive :obj:`float`, optional, default= Norm of the operator A
        The balancing scale :math:`s` used to compare the primal and dual residuals. Defaults to the operator norm :math:`\|K\|`.
    eta : positive :obj:`float`, optional, default=0.95
        The decay factor :math:`\eta \in (0,1)` applied to :math:`\alpha` each time the step sizes are rebalanced.
    inner_iterations : :obj:`int`, optional, default=50
        The maximum number of inner iterations for the backtracking loop.
    auto_stop : :obj:`boolean`, optional, default=True
        If True, the adaptive step size method automatically stops updating the step sizes once they have not changed for more than ``auto_stop_patience`` consecutive iterations.
    auto_stop_patience : :obj:`int`, optional, default=10
        Number of consecutive iterations with no change to the step sizes that are tolerated before the adaptive updates are stopped (only used when ``auto_stop=True``).


    Notes
    -----
    This method is memory expensive, requiring the storage of 1 extra image copy and 2 extra data copies. When ``auto_stop=True`` the adaptive updates are switched off once the step sizes have been unchanged for more than ``auto_stop_patience`` consecutive iterations; the extra image and data are then released, reducing ongoing memory use and computational cost. For a more time expensive, but less memory expensive method, see :class:`PDHGBayesOptimisationStepSize` which does not require the storage of extra images and data.

    Both mechanisms are skipped for an iteration in which the primal and dual residuals have both fallen below the
    attribute ``tolerance`` (default ``1e-6``), on the grounds that the iterates have stopped moving and the residuals
    no longer carry usable information. ``tolerance`` is not a constructor argument but can be set on the rule after
    construction.

    See Also
    --------
    PDHGAdaptiveStepSize2015 : A related backtracking/residual-balancing rule from :cite:`Goldstein2015`.

    Reference
    ---------
    Goldstein, T., Li, M., Yuan, X., Esser, E. and Baraniuk, R., 2013. Adaptive primal-dual hybrid gradient methods for saddle-point problems. arXiv preprint arXiv:1305.0546. :cite:`goldstein2013adaptive`
    """

    def __init__(self, initial_step_size=[None, None], initial_alpha=0.95, beta=0.95, gamma=0.9, delta=1.5, s=None, eta=0.95, inner_iterations=50, auto_stop=True, auto_stop_patience=10):
        '''Initialises the step size rule'''
        self.alpha = initial_alpha
        self.eta = eta
        self.beta = beta
        self.delta = delta
        self.s = s
        self.gamma = gamma
        self.tolerance = 1e-6
        self.p_norm = 100
        self.d_norm = 100
        self.inner_iterations = inner_iterations
        self.auto_stop = auto_stop
        self.auto_stop_patience = auto_stop_patience
        self.count = 0

        self.y_old = None
        self.x_resid = None
        self.y_resid = None
        self.x_prev = None

        self.adaptive = True
        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, step_size = {}".format(initial_step_size))

    def get_initial_step_size(self, algorithm):
        r"""Returns the initial primal and dual step sizes for the PDHG algorithm.

        Values are taken from ``initial_step_size``; each entry left as ``None`` defaults to
        :math:`10/\|K\|`.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to, used here for its operator.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        tau = self.initial_step_size[0]
        sigma = self.initial_step_size[1]
        if tau is None:
            tau = 10/algorithm.operator.norm()
        if sigma is None:
            sigma = 10/algorithm.operator.norm()
        return tau, sigma

    def get_step_size(self, algorithm):
        """Applies the backtracking and residual-balancing updates for one PDHG iteration.

        Both steps modify ``algorithm._tau`` and ``algorithm._sigma`` in place, and backtracking
        additionally recomputes the PDHG update. Once the adaptive updates have been stopped by
        ``auto_stop`` this returns the current step sizes unchanged.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        if self.adaptive:
            if self.s is None:
                self.s = algorithm.operator.norm()  # default balancing scale, ||A||
            if self.y_old is None:
                self.y_old = algorithm.operator.range_geometry().allocate(0)  # Extra range data 1
                self.x_resid = algorithm.operator.domain_geometry().allocate(0)  # Extra image 1
                self.y_resid = algorithm.operator.range_geometry().allocate(0)  # Extra range data 2
                # x^{k-1}, the extrapolation partner of this iteration. On the first call
                # x^{-1} = x^{0} = algorithm.x_old, which is what set_up copied x from.
                self.x_prev = algorithm.x_old.copy()  # Extra image 2
            # adaptive step sizes only when above tolerance
            if self.p_norm > self.tolerance and self.d_norm > self.tolerance:
                log.debug('Before adaptive step-size step, tau = {}, sigma = {}'.format(
                    algorithm._tau, algorithm._sigma))
                b = self._calculate_backtracking(algorithm)
                converged = False
                for k in range(self.inner_iterations):
                    if b <= 1:
                        log.debug('Finished backtracking step, backtracking value b = {}, step sizes are tau = {}, sigma = {}'.format(
                            b, algorithm._tau, algorithm._sigma))
                        converged = True
                        break

                    algorithm._tau *= self.beta/b
                    algorithm._sigma *= self.beta/b
                    log.debug(' Backtracking step - multiplying primal and dual step sizes by beta/b = {}, new step sizes are tau = {}, sigma ={}'.format(
                        self.beta / b, algorithm._tau, algorithm._sigma))

                    self._restore_iterate(algorithm)
                    algorithm._pdhg_update()
                    b = self._calculate_backtracking(algorithm)
                    self.count = 0

                if not converged:
                    log.warning('Backtracking step did not converge after {} iterations, backtracking value b = {}, step sizes are tau = {}, sigma = {}'.format(
                        self.inner_iterations, b, algorithm._tau, algorithm._sigma))

                self._calculate_pnorm_dnorm(algorithm)
                log.debug('Started the rebalancing step with p_norm = {}, d_norm = {}'.format(
                    self.p_norm, self.d_norm))
                if self.p_norm < (self.s/self.delta)*self.d_norm:
                    algorithm._tau *= (1 - self.alpha)
                    algorithm._sigma /= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                    log.debug('p_norm < (s/delta)*d_norm so rebalancing step sizes, new step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))
                elif (self.s*self.delta)*self.d_norm < self.p_norm:
                    algorithm._tau /= (1 - self.alpha)
                    algorithm._sigma *= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                    log.debug('(s*delta)*d_norm < p_norm so rebalancing step sizes, new step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))
                else:
                    log.debug('No change from the rebalancing step, step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))
                    self.count += 1
            else:
                log.debug('No change from the rebalancing step as pnorm and dnorm are below threshold, step sizes are tau = {}, sigma ={}'.format(
                    algorithm._tau, algorithm._sigma))

            self.y_old.fill(algorithm.y)
            # algorithm.x_old is x^{k}, the extrapolation partner of iteration k+1
            self.x_prev.fill(algorithm.x_old)

            if self.count > self.auto_stop_patience and self.auto_stop:
                self.adaptive = False
                log.debug('Automatic stopping of adaptive step size updates, step sizes have not changed for {} iterations, step sizes are tau = {}, sigma ={}'.format(
                    self.auto_stop_patience, algorithm._tau, algorithm._sigma))
                del self.x_resid
                del self.y_resid
                del self.y_old
                del self.x_prev

        return algorithm._tau, algorithm._sigma

    def _restore_iterate(self, algorithm):
        """Restores the iterate to the state it had at the start of the iteration.

        :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update` overwrites ``x`` and ``y``
        in place, so before the step can be retaken with smaller step sizes both must be
        put back. Otherwise the retry extrapolates from the rejected trial iterate rather
        than from :math:`x^{k-1}`, and the dual variable accumulates over successive
        trials, which can make the iterate overflow before the backtracking loop has had
        a chance to bring the step sizes down.
        """
        algorithm.x.fill(self.x_prev)
        algorithm.y.fill(self.y_old)

    def _calculate_pnorm_dnorm(self, algorithm):
        algorithm.operator.adjoint(self.y_resid, out=algorithm.x_tmp)
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        self.x_resid.sapyb((1/algorithm._tau),
                           algorithm.x_tmp, -1.0, out=algorithm.x_tmp)
        self.y_resid.sapyb((1/algorithm._sigma),
                           algorithm.y_tmp, -1.0, out=algorithm.y_tmp)
        self.p_norm = algorithm.x_tmp.norm()
        self.d_norm = algorithm.y_tmp.norm()

    def _calculate_backtracking(self, algorithm):
        """ Calculates the backtracking parameter b used to update step sizes in the adaptive PDHG algorithm.
            Returns
            -------
            b : :obj:`float`
                Backtracking parameter used to update step sizes in the adaptive PDHG algorithm.
        """

        algorithm.x.sapyb(1.0, algorithm.x_old, -1.0, out=self.x_resid)
        x_change_norm = self.x_resid.norm()
        algorithm.y.sapyb(1.0, self.y_old, -1.0, out=self.y_resid)
        y_change_norm = self.y_resid.norm()
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        cross_term = np.abs(2*algorithm._sigma *
                            algorithm._tau*self.y_resid.dot(algorithm.y_tmp))

        denominator = (self.gamma*algorithm._sigma)*x_change_norm**2 + \
            (self.gamma*algorithm._tau)*y_change_norm**2
        if denominator == 0:
            # No change in the iterate (x == x_old and y == y_old): nothing to
            # backtrack. Return 0 so the caller's `b <= 1` test accepts the step
            # (dividing here would give 0/0 = nan and poison the step sizes).
            b = 0.0
        else:
            b = cross_term/denominator
        log.debug('Backtracking value = {}'.format(b))
        return b


class PDHGAdaptiveStepSize2015(StepSizeRule):
    r"""Adaptively updates the PDHG primal and dual step sizes using the backtracking and residual-balancing method of :cite:`Goldstein2015`.

    This is a PDHG-compatible step-size rule and a variant of :class:`PDHGAdaptiveStepSize2013`.
    It provides an initial pair of step sizes via :meth:`get_initial_step_size` and then, at the
    end of every PDHG iteration, :meth:`get_step_size` updates :math:`\tau` and :math:`\sigma`
    in place. Two mechanisms are applied each iteration.

    **1. Backtracking.** With the increments :math:`\Delta x = x^{n+1} - x^{n}` and
    :math:`\Delta y = y^{n+1} - y^{n}`, a backtracking quantity is formed

    .. math::

        b = c\,\sigma\,\|\Delta x\|^{2} + c\,\tau\,\|\Delta y\|^{2}
            - 4\,\sigma\tau\,|\langle \Delta y, K\Delta x\rangle|.

    The step is accepted when :math:`b \ge 0`. Otherwise both step sizes are halved,
    :math:`\tau \leftarrow \tfrac{1}{2}\tau`, :math:`\sigma \leftarrow \tfrac{1}{2}\sigma`, the
    PDHG update is recomputed (via :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update`) and
    :math:`b` is re-evaluated, up to ``inner_iterations`` times.

    **2. Residual balancing.** The primal and dual residuals

    .. math::

        p = \left\| \tfrac{\Delta x}{\tau} - K^{*}\Delta y \right\|, \qquad
        d = \|K\| \left\| \tfrac{\Delta y}{\sigma} - K\Delta x \right\|

    are compared. If :math:`2p < d` the ratio is tilted towards the dual variable
    (:math:`\tau \leftarrow (1-\alpha)\tau`, :math:`\sigma \leftarrow \sigma/(1-\alpha)`);
    if :math:`2d < p` it is tilted towards the primal variable
    (:math:`\tau \leftarrow \tau/(1-\alpha)`, :math:`\sigma \leftarrow (1-\alpha)\sigma`).
    Whenever a tilt is applied the adaptation strength decays, :math:`\alpha \leftarrow \eta\,\alpha`.
    The balancing step preserves the product :math:`\sigma\tau`.

    Parameters
    -------------
    initial_step_size : list of two positive :obj:`float`, optional, default= [10/algorithm.operator.norm(), 10/algorithm.operator.norm()]
        Initial values of the primal and dual step sizes used in the adaptive step size method.
    initial_alpha : :obj:`float` in :math:`(0,1)`, optional, default=0.95
        Initial value of the adaptation strength :math:`\alpha` controlling the size of the residual-balancing update.
        Each rebalance multiplies or divides :math:`\tau` by :math:`1-\alpha`, so :math:`\alpha` close to one gives a
        *large* rescaling (:math:`\alpha = 0.95` changes :math:`\tau` by a factor of 20) and :math:`\alpha` close to
        zero a gentle one.
    eta : positive :obj:`float`, optional, default=0.95
        The decay factor :math:`\eta \in (0,1)` applied to :math:`\alpha` each time the step sizes are rebalanced.
    c : positive :obj:`float`, optional, default=0.9
        The convergence constant :math:`c` appearing in the backtracking quantity :math:`b`.
    inner_iterations : :obj:`int`, optional, default=50
        The maximum number of inner iterations for the backtracking loop.
    auto_stop : :obj:`boolean`, optional, default=True
        If True, the adaptive step size method automatically stops updating the step sizes once they have not changed for more than ``auto_stop_patience`` consecutive iterations.
    auto_stop_patience : :obj:`int`, optional, default=10
        Number of consecutive iterations with no change to the step sizes that are tolerated before the adaptive updates are stopped (only used when ``auto_stop=True``).


    Notes
    -----
    This method is memory expensive, requiring the storage of 1 extra image copy and 2 extra data copies. When ``auto_stop=True`` the adaptive updates are switched off once the step sizes have been unchanged for more than ``auto_stop_patience`` consecutive iterations; the extra image and data are then released, reducing ongoing memory use and computational cost. For a more time expensive, but less memory expensive method, see :class:`PDHGBayesOptimisationStepSize` which does not require the storage of extra images and data.

    Both mechanisms are skipped for an iteration in which the primal and dual residuals have both fallen below the
    attribute ``tolerance`` (default ``1e-6``), on the grounds that the iterates have stopped moving and the residuals
    no longer carry usable information. ``tolerance`` is not a constructor argument but can be set on the rule after
    construction.

    See Also
    --------
    PDHGAdaptiveStepSize2013 : The related backtracking/residual-balancing rule from :cite:`goldstein2013adaptive`.

    Reference
    ---------
    Goldstein, T., Li, M. and Yuan, X., 2015. Adaptive primal-dual splitting methods for statistical learning and image processing. Advances in Neural Information Processing Systems, 28. :cite:`Goldstein2015`
    """

    def __init__(self, initial_step_size=[None, None],  initial_alpha=0.95, eta=0.95, c=0.9, inner_iterations=50, auto_stop=True, auto_stop_patience=10):
        '''Initialises the step size rule'''

        self.adaptive = True
        self.alpha = initial_alpha
        self.eta = eta
        self.c = c
        self.tolerance = 1e-6
        self.p_norm = 100
        self.d_norm = 100
        self.auto_stop = auto_stop
        self.auto_stop_patience = auto_stop_patience
        self.count = 0
        self.inner_iterations = inner_iterations

        self.y_old = None
        self.x_resid = None
        self.y_resid = None
        self.x_prev = None
        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, step_size = {}".format(initial_step_size))

    def get_initial_step_size(self, algorithm):
        r"""Returns the initial primal and dual step sizes for the PDHG algorithm.

        Values are taken from ``initial_step_size``; each entry left as ``None`` defaults to
        :math:`10/\|K\|`.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to, used here for its operator.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        tau = self.initial_step_size[0]
        sigma = self.initial_step_size[1]
        if tau is None:
            tau = 10/algorithm.operator.norm()
        if sigma is None:
            sigma = 10/algorithm.operator.norm()
        return tau, sigma

    def get_step_size(self, algorithm):
        """Applies the backtracking and residual-balancing updates for one PDHG iteration.

        Both steps modify ``algorithm._tau`` and ``algorithm._sigma`` in place, and backtracking
        additionally recomputes the PDHG update. Once the adaptive updates have been stopped by
        ``auto_stop`` this returns the current step sizes unchanged.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        if self.adaptive:
            if self.y_old is None:
                self.y_old = algorithm.operator.range_geometry().allocate(0)  # Extra range data 1
                self.x_resid = algorithm.operator.domain_geometry().allocate(0)  # Extra image 1
                self.y_resid = algorithm.operator.range_geometry().allocate(0)  # Extra range data 2
                # x^{k-1}, the extrapolation partner of this iteration. On the first call
                # x^{-1} = x^{0} = algorithm.x_old, which is what set_up copied x from.
                self.x_prev = algorithm.x_old.copy()  # Extra image 2
            if self.p_norm > self.tolerance and self.d_norm > self.tolerance:
                log.debug('Before adaptive step-size step, tau = {}, sigma = {}'.format(
                    algorithm._tau, algorithm._sigma))

                b = self._calculate_backtracking(algorithm)
                converged = False
                for k in range(self.inner_iterations):
                    if b >= 0:
                        log.debug('Finished backtracking step, backtracking value b = {}, step sizes are tau = {}, sigma = {}'.format(
                            b, algorithm._tau, algorithm._sigma))
                        converged = True
                        break
                    algorithm._tau *= 0.5
                    algorithm._sigma *= 0.5
                    log.debug(' Backtracking step - multiplying primal and dual step sizes by 1/2, new step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))

                    self._restore_iterate(algorithm)
                    algorithm._pdhg_update()
                    b = self._calculate_backtracking(algorithm)
                    self.count = 0
                if not converged:
                    log.warning('Backtracking step did not converge after {} iterations, backtracking value b = {}, step sizes are tau = {}, sigma = {}'.format(
                        self.inner_iterations, b, algorithm._tau, algorithm._sigma))

                self._calculate_pnorm_dnorm(algorithm)
                log.debug('Started the rebalancing step with p_norm = {}, d_norm = {}'.format(
                    self.p_norm, self.d_norm))
                if 2*self.p_norm < self.d_norm:
                    algorithm._tau *= (1 - self.alpha)
                    algorithm._sigma /= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                    log.debug('2*p_norm < d_norm so rebalancing step sizes, new step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))
                elif 2*self.d_norm < self.p_norm:
                    algorithm._tau /= (1 - self.alpha)
                    algorithm._sigma *= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                    log.debug('2*d_norm < p_norm so rebalancing step sizes, new step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))
                else:
                    log.debug('No change from the rebalancing step, step sizes are tau = {}, sigma ={}'.format(
                        algorithm._tau, algorithm._sigma))
                    self.count += 1
            else:
                log.debug('No change from the rebalancing step as pnorm and dnorm are below threshold, step sizes are tau = {}, sigma ={}'.format(
                    algorithm._tau, algorithm._sigma))
            self.y_old.fill(algorithm.y)
            # algorithm.x_old is x^{k}, the extrapolation partner of iteration k+1
            self.x_prev.fill(algorithm.x_old)

            if self.count > self.auto_stop_patience and self.auto_stop:
                self.adaptive = False
                log.debug('Automatic stopping of adaptive step size updates, step sizes have not changed for {} iterations, step sizes are tau = {}, sigma ={}'.format(
                    self.auto_stop_patience, algorithm._tau, algorithm._sigma))
                del self.y_resid
                del self.y_old
                del self.x_resid
                del self.x_prev

        return algorithm._tau, algorithm._sigma

    def _restore_iterate(self, algorithm):
        """Restores the iterate to the state it had at the start of the iteration.

        :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update` overwrites ``x`` and ``y``
        in place, so before the step can be retaken with smaller step sizes both must be
        put back. Otherwise the retry extrapolates from the rejected trial iterate rather
        than from :math:`x^{k-1}`, and the dual variable accumulates over successive
        trials, which can make the iterate overflow before the backtracking loop has had
        a chance to bring the step sizes down.
        """
        algorithm.x.fill(self.x_prev)
        algorithm.y.fill(self.y_old)

    def _calculate_backtracking(self, algorithm):
        """ Calculates the backtracking parameter b used to update step sizes in the adaptive PDHG algorithm.
            Returns
            -------
            b : :obj:`float`
                Backtracking parameter used to update step sizes in the adaptive PDHG algorithm.
        """

        algorithm.x.sapyb(1.0, algorithm.x_old, -1.0, out=self.x_resid)
        x_change_norm = self.x_resid.norm()
        algorithm.y.sapyb(1.0, self.y_old, -1.0, out=self.y_resid)
        y_change_norm = self.y_resid.norm()
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        cross_term = np.abs(4*algorithm._sigma*algorithm._tau *
                            self.y_resid.dot(algorithm.y_tmp))
        b = self.c*algorithm._sigma*x_change_norm**2 + \
            self.c*algorithm._tau*y_change_norm**2 - cross_term
        log.debug('Backtracking value = {}'.format(b))
        return b

    def _calculate_pnorm_dnorm(self, algorithm):
        """Calculates the primal and dual norms used in the rebalancing step of the adaptive PDHG algorithm.
        """
        algorithm.operator.adjoint(self.y_resid, out=algorithm.x_tmp)
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        self.x_resid.sapyb((1/algorithm._tau),
                           algorithm.x_tmp, -1.0, out=algorithm.x_tmp)
        self.y_resid.sapyb((1/algorithm._sigma),
                           algorithm.y_tmp, -1.0, out=algorithm.y_tmp)
        self.p_norm = algorithm.x_tmp.norm()
        self.d_norm = algorithm.operator.norm()*algorithm.y_tmp.norm()


class _BayesOptimisationStepSizeBase(StepSizeRule):
    """Shared implementation for the Bayesian-optimisation step-size rules.

    A Gaussian-process Bayesian optimisation (via ``skopt.gp_minimize``) chooses the
    scalar ratio ``gamma`` between the primal and dual step sizes that gives the best
    objective after a small number of iterations. The step sizes are then fixed for the
    remainder of the algorithm.

    Subclasses implement :meth:`_step_sizes_from_gamma` (the mapping from ``gamma`` to
    the ``(tau, sigma)`` step sizes) and may override :meth:`_default_n_iterations`.

    Once the search has run, the chosen ratio is available as ``gamma`` and the step
    sizes it produced as ``tau`` and ``sigma``.
    """

    def __init__(self, gamma_bounds=None, n_initial_points=5, n_calls=20, n_iterations=None, seed=None, plot=False):
        '''Initialises the step size rule'''
        self.gamma_bounds = gamma_bounds
        if gamma_bounds is not None:
            if len(gamma_bounds) != 2:
                raise ValueError(
                    "gamma_bounds should be a list or tuple of length two, gamma_bounds = {}".format(gamma_bounds))
            if gamma_bounds[0] <= 0 or gamma_bounds[1] <= 0:
                raise ValueError(
                    "gamma_bounds should be positive and strictly greater than zero, gamma_bounds = {}".format(gamma_bounds))
            if gamma_bounds[0] >= gamma_bounds[1]:
                raise ValueError(
                    "gamma_bounds should be increasing, gamma_bounds = {}".format(gamma_bounds))

        if not (isinstance(n_initial_points, (int, np.integer)) and n_initial_points > 0):
            raise ValueError(
                "n_initial_points should be a positive integer, n_initial_points = {}".format(n_initial_points))
        if not (isinstance(n_calls, (int, np.integer)) and n_calls >= n_initial_points):
            raise ValueError(
                "n_calls should be an integer greater than or equal to n_initial_points = {}, n_calls = {}".format(n_initial_points, n_calls))
        # the objective of each trial is read off after n_iterations-1 updates, so a
        # single iteration would leave nothing to read
        if n_iterations is not None and not (isinstance(n_iterations, (int, np.integer)) and n_iterations > 1):
            raise ValueError(
                "n_iterations should be an integer greater than one, or None, n_iterations = {}".format(n_iterations))

        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.n_iterations = n_iterations
        self.seed = seed
        self.plot = plot

    def _default_n_iterations(self, algorithm):
        """Number of iterations per objective evaluation when ``n_iterations`` is None."""
        return 10

    def _step_sizes_from_gamma(self, algorithm, gamma):
        """Map the scalar ratio ``gamma`` to ``(tau, sigma)``. Implemented by subclasses."""
        raise NotImplementedError

    def _algorithm_set_up(self, algorithm, step_size):
        """Set up the algorithm with the given step size."""
        raise NotImplementedError

    def get_initial_step_size(self, algorithm):
        try:
            from skopt import gp_minimize
        except ImportError:
            raise ImportError(
                "skopt is required for the Bayesian optimisation step size rule. Please install scikit-optimize to use this step size rule.")

        if self.n_iterations is None:
            self.n_iterations = self._default_n_iterations(algorithm)
            log.debug(
                "n_iterations not provided, set to {}".format(self.n_iterations))

        if self.gamma_bounds is None:
            ratio = np.sqrt(algorithm.f(algorithm.operator.direct(
                0*algorithm.x)))/algorithm.operator.norm()
            log.debug("ratio: {}".format(ratio))
            self.gamma_bounds = (1e-5/ratio, 1e5/ratio)
            log.debug(
                "gamma_bounds not provided, set to (1e-5/ratio, 1e5/ratio) = {}".format(self.gamma_bounds))

        log_gamma_bounds = (
            np.log(self.gamma_bounds[0]), np.log(self.gamma_bounds[1]))
        update_objective_interval = algorithm.update_objective_interval

        def objective_function(log_gamma):
            """ Objective function for the Bayesian optimisation. Evaluates the algorithm for a given log(gamma) and returns the objective function value after n_iterations."""
            gamma = np.exp(log_gamma[0])
            log.debug(
                "Evaluating objective function for gamma = {}".format(gamma))
            # Set the step sizes based on the current gamma
            tau, sigma = self._step_sizes_from_gamma(algorithm, gamma)

            self._algorithm_set_up(algorithm, [tau, sigma])

            try:
                with np.errstate(over='raise'):
                    algorithm.run(self.n_iterations, callbacks=[])
            except FloatingPointError:
                log.debug(
                    "Objective function diverged for gamma = {}".format(gamma))
                return 1e10  # large penalty

            log.debug("Objective function value for gamma = {}: {}".format(
                gamma, algorithm.objective[-1]))
            if not np.isfinite(algorithm.objective[-1]):
                return 1e10  # large penalty
            return algorithm.objective[-1]

        algorithm.update_objective_interval = self.n_iterations-1
        gp_result = gp_minimize(objective_function, [
            log_gamma_bounds], n_random_starts=self.n_initial_points, n_calls=self.n_calls, initial_point_generator="lhs", random_state=self.seed)

        algorithm.update_objective_interval = update_objective_interval

        # gp_result.x[0] is log(gamma) as the optimisation was over log_gamma_bounds
        self.gamma = np.exp(gp_result.x[0])
        self.tau, self.sigma = self._step_sizes_from_gamma(algorithm, self.gamma)

        log.debug("Best gamma found: {}, with objective function value: {}".format(
            self.gamma, gp_result.fun))
        log.debug('Initial step sizes are tau = {}, sigma = {}'.format(
            self.tau, self.sigma))

        self._algorithm_set_up(algorithm, [self.tau, self.sigma])

        if self.plot:
            from matplotlib import pyplot as plt
            from skopt.plots import plot_convergence, plot_gaussian_process
            plot_convergence(gp_result)
            plt.show()
            plot_gaussian_process(gp_result)
            plt.yscale('log')
            plt.xlabel('log gamma')
            plt.title("Best gamma found: {}, with objective function value: {}".format(
                self.gamma, gp_result.fun))
            plt.show()

        return self.tau, self.sigma

    def get_step_size(self, algorithm):
        log.debug('Returning step sizes tau = {}, sigma = {}'.format(
            self.tau, self.sigma))
        return self.tau, self.sigma


class PDHGBayesOptimisationStepSize(_BayesOptimisationStepSizeBase):
    r"""The ratio between the primal and dual step sizes (gamma) in the PDHG algorithm is chosen using a Gaussian process Bayesian optimisation, choosing the gamma that gives the best performance after a small number of iterations. The step sizes are chosen at the beginning of the algorithm and then kept constant throughout the iterations.
        Parameters
        -------------
        gamma_bounds : list or tuple of length two, optional, default=``(1e-5/ratio, 1e5/ratio)``
            Bounds for the ratio between the primal and dual step sizes (gamma) in the Bayesian optimisation. The gamma that gives the best performance after a small number of iterations is chosen as the ratio between the primal and dual step sizes for the PDHG algorithm. If not provided, the bounds span five orders of magnitude either side of ``ratio``, an estimate of the natural scale of gamma given by :math:`\sqrt{f(K0)}/\|K\|`. This is approximately :math:`\|b\|/\|K\|` for a data-fitting term with data :math:`b`, so the default bounds are approximately :math:`[10^{-5}, 10^{5}]\,\|K\|/\|b\|`.
        n_initial_points : int, optional, default=5
            Number of initial random evaluations of the objective function in the Bayesian optimisation.
        n_calls : int, optional, default=20
            Total number of evaluations of the objective function in the Bayesian optimisation, including the initial random evaluations.
        n_iterations : int, optional, default=10
            Number of iterations to run the PDHG algorithm for each evaluation of the objective function in the Bayesian optimisation. The gamma that gives the best performance after this number of iterations is chosen as the ratio between the primal and dual step sizes for the PDHG algorithm.
        seed : int, optional, default= None
            Random seed for the Bayesian optimisation. This is used to ensure reproducibility of the results.
        plot : bool, optional, default=False
            If True, plots the convergence of the Bayesian optimisation and the fitted Gaussian process after the step sizes have been chosen. Requires matplotlib and blocks on ``plt.show()``, so it should be left False for headless/non-interactive runs.

        Notes
        -----
        The step sizes are parametrised by a single scalar ratio :math:`\gamma` and the operator
        norm :math:`\|K\|`,

        .. math::

            \tau = \frac{1}{\gamma \|K\|}, \qquad \sigma = \frac{\gamma}{\|K\|},

        so that the product :math:`\sigma\tau = 1/\|K\|^{2}` is fixed and only the primal/dual
        balance is optimised. The search is performed over :math:`\log\gamma` (which spans
        several orders of magnitude) using ``skopt.gp_minimize``, so `scikit-optimize` must be
        installed to use this rule.

        This is a computationally expensive step size rule, as it requires running the PDHG algorithm for a number of iterations for each evaluation of the objective function in the Bayesian optimisation. It is recommended to use this step size rule where you are memory constrained but less time constrained. For the opposite case, where you are more time constrained, not memory constrained, we recommend using the :class:`PDHGAdaptiveStepSize2013` or :class:`PDHGAdaptiveStepSize2015` step size rules, which are adaptive step size rules that update the step sizes at each iteration based on the observed behaviour of the algorithm.
        """

    def __init__(self, gamma_bounds=None, n_initial_points=5, n_calls=20, n_iterations=10, seed=None, plot=False):
        '''Initialises the step size rule'''
        super().__init__(gamma_bounds=gamma_bounds, n_initial_points=n_initial_points,
                         n_calls=n_calls, n_iterations=n_iterations, seed=seed, plot=plot)

    def _step_sizes_from_gamma(self, algorithm, gamma):
        norm = algorithm.operator.norm()
        tau = 1.0 / (gamma * norm)
        sigma = 1.0 * gamma / norm
        return tau, sigma

    def _algorithm_set_up(self, algorithm, step_size):
        """Set up the algorithm with the given step size."""
        algorithm.set_up(initial=algorithm.initial, f=algorithm.f,
                         g=algorithm.g, operator=algorithm.operator, step_size=step_size)
        algorithm._reset_iteration_state()


class PDHGConstantStepSize(StepSizeRule):
    r"""
    Step-size rule that always returns a constant step-size.

    The user can set either the primal or dual step size, both or none.

    By default, the step sizes :math:`\sigma` and :math:`\tau` are positive scalars and defined as below:

      * If ``sigma`` is ``None`` and ``tau`` is ``None``:

      .. math::

        \sigma = \frac{1}{\|K\|},  \tau = \frac{1}{\|K\|}

      * If ``tau`` is ``None``:

      .. math::

        \tau = \frac{1}{\sigma\|K\|^{2}}

      * If ``sigma`` is ``None``:

      .. math::

        \sigma = \frac{1}{\tau\|K\|^{2}}


    Parameters
    ----------
    step_size : list or tuple of length two,  default=[None, None]
        Initial values of the primal and dual step sizes. If both are ``None`` they are set to the default values defined above. If one is ``None`` it is calculated based on the other and the norm of the operator. If both are provided, they are used as they are, as long as they are positive numbers.
    """

    def __init__(self,  step_size=[None, None]):
        '''Initialises the constant step size rule'''

        if len(step_size) != 2:
            raise ValueError(
                "step_size should be a list or tuple of length two, step_size = {}".format(step_size))
        self.tau = step_size[0]
        self.sigma = step_size[1]

    def get_initial_step_size(self, algorithm):
        """Sets sigma and tau step-sizes for the PDHG algorithm."""
        self.tau, self.sigma = _resolve_pdhg_step_sizes(
            self.tau, self.sigma, algorithm.operator)
        return self.tau, self.sigma

    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        return self.tau, self.sigma


class SPDHGConstantStepSize(StepSizeRule):
    r"""Step-size rule that always returns a constant step-size for the SPDHG algorithm.
    The user can set either the primal or dual step size, both or none. 

    When setting `sigma` and `tau`, there are 4 possible cases considered by setup function: 

        - Case 1: If neither `sigma` or `tau` are provided then `sigma` is set using the formula:

        .. math:: \sigma_i= \frac{0.99}{\|K_i\|}

        and `tau` is set as per case 2

        - Case 2: If `sigma` is provided but not `tau` then `tau` is calculated using the formula 

        .. math:: \tau = 0.99\min_i( \frac{p_i}{ (\sigma_i  \|K_i\|^2) })

        - Case 3: If `tau` is provided but not `sigma` then `sigma` is calculated using the formula

        .. math:: \sigma_i= \frac{0.99 p_i}{\tau\|K_i\|^2}

        - Case 4: Both `sigma` and `tau` are provided.

    Parameters
        ----------
        step_size : list or tuple of length two,  default=[None, None]
                Initial values of the primal and dual step sizes. If both are ``None`` they are set to the default values defined above. If one is ``None`` it is calculated based on the other and the norms of the operators. If both are provided, they are used as they are, as long as sigma is a list or array of positive numbers of length equal to the number of operators and tau is a positive number.
    """

    def __init__(self,  step_size=[None, None]):
        '''Initialises the constant step size rule
       '''

        if len(step_size) != 2:
            raise ValueError(
                "step_size should be a list or tuple of length two, step_size = {}".format(step_size))
        self.tau = step_size[0]
        self.sigma = step_size[1]

    def get_initial_step_size(self, algorithm):
        r""" Sets sigma and tau step-sizes for the SPDHG algorithm after the initial set-up. The step sizes can be either scalar or array-objects.
        """
        gamma = 1.
        rho = .99

        if self.tau is not None and not (isinstance(self.tau, Number) and self.tau > 0):
            raise ValueError(
                "The step-sizes of SPDHG must be positive, passed tau = {}".format(self.tau))

        if self.sigma is not None:

            if not isinstance(self.sigma, Number) and len(self.sigma) == algorithm._ndual_subsets:
                if all(isinstance(x, Number) and x > 0 for x in self.sigma):
                    pass
                else:
                    raise ValueError(
                        "Sigma expected to be a positive number.")

            else:
                raise ValueError(
                    "Please pass a list of floats to sigma with the same number of entries as number of operators")

        elif self.tau is None:
            self.sigma = _spdhg_sigma_from_gamma(gamma, rho, algorithm._norms)
        else:
            self.sigma = [
                rho*pi / (self.tau*ni**2) for ni, pi in zip(algorithm._norms, algorithm._prob_weights)]

        if self.tau is None:
            self.tau = _spdhg_tau_from_sigma(
                self.sigma, algorithm._norms, algorithm._prob_weights, rho)

        return self.tau, self.sigma

    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the primal step size and the list of dual step sizes as a tuple ``(tau, sigma)``
        """
        return self.tau, self.sigma


class SPDHGStepSizesFromRatio(StepSizeRule):
    r""" Sets gamma, the step-size ratio for the SPDHG algorithm. Currently gamma takes a scalar value.

    The step sizes `sigma` and `tau` are set using the equations:

    .. math:: \sigma_i= \frac{\gamma\rho }{\|K_i\|}

    .. math::  \tau = \rho\min_i\left( \frac{p_i }{\sigma_i  \|K_i\|^2}\right)

    where :math:`p_i` is the sampling probability of the :math:`i`-th operator. The dual step
    size :math:`\sigma` is therefore a list with one entry per operator, while the primal step
    size :math:`\tau` is a scalar.

    Parameters
    ----------
        gamma : Positive float
            parameter controlling the trade-off between the primal and dual step sizes
        rho : Positive float
            parameter controlling the size of the product :math:`\sigma\tau`



    """

    def __init__(self, gamma, rho):
        """Initialises the step size rule"""
        self.gamma = gamma
        self.rho = rho

    def get_initial_step_size(self, algorithm):
        if isinstance(self.gamma, Number):
            if self.gamma <= 0:
                raise ValueError(
                    "The step-sizes of SPDHG are positive, gamma should also be positive")

        else:
            raise ValueError(
                "We currently only support scalar values of gamma")
        if isinstance(self.rho, Number):
            if self.rho <= 0:
                raise ValueError(
                    "The step-sizes of SPDHG are positive, rho should also be positive")

        else:
            raise ValueError(
                "We currently only support scalar values of gamma")

        self.sigma = _spdhg_sigma_from_gamma(
            self.gamma, self.rho, algorithm._norms)
        self.tau = _spdhg_tau_from_sigma(
            self.sigma, algorithm._norms, algorithm._prob_weights, self.rho)

        return self.tau, self.sigma

    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the primal step size and the list of dual step sizes as a tuple ``(tau, sigma)``
        """
        return self.tau, self.sigma


class SPDHGBayesOptimisationStepSize(_BayesOptimisationStepSizeBase):
    r"""The ratio between the primal and dual step sizes (gamma) in the SPDHG algorithm is chosen using a Gaussian process Bayesian optimisation, choosing the gamma that gives the best performance after a small number of iterations. The step sizes are chosen at the beginning of the algorithm and then kept constant throughout the iterations.
        Parameters
        -------------
        gamma_bounds : list or tuple of length two, optional, default=``(1e-5/ratio, 1e5/ratio)``
            Bounds for the ratio between the primal and dual step sizes (gamma) in the Bayesian optimisation. The gamma that gives the best performance after a small number of iterations is chosen as the ratio between the primal and dual step sizes for the SPDHG algorithm. If not provided, the bounds span five orders of magnitude either side of ``ratio``, an estimate of the natural scale of gamma given by :math:`\sqrt{f(K0)}/\|K\|`. This is approximately :math:`\|b\|/\|K\|` for a data-fitting term with data :math:`b`, so the default bounds are approximately :math:`[10^{-5}, 10^{5}]\,\|K\|/\|b\|`.
        n_initial_points : int, optional, default=5
            Number of initial random evaluations of the objective function in the Bayesian optimisation.
        n_calls : int, optional, default=20
            Total number of evaluations of the objective function in the Bayesian optimisation, including the initial random evaluations.
        n_iterations : int, optional, default=None
            Number of iterations to run the SPDHG algorithm for each evaluation of the objective function in the Bayesian optimisation. The gamma that gives the best performance after this number of iterations is chosen as the ratio between the primal and dual step sizes for the SPDHG algorithm. If None, set to be 10*number of operators in the SPDHG algorithm.
        seed : int, optional, default= None
            Random seed for the Bayesian optimisation. This is used to ensure reproducibility of the results.
        plot : bool, optional, default=False
            If True, plots the convergence of the Bayesian optimisation and the fitted Gaussian process after the step sizes have been chosen. Requires matplotlib and uses ``plt.show()``, so it should be left False for non-interactive runs.

        Notes
        -----
        The step sizes are parametrised by a single scalar ratio :math:`\gamma` together with
        the fixed product parameter :math:`\rho = 0.99`, the operator norms :math:`\|K_i\|` and
        the sampling probabilities :math:`p_i`,

        .. math::

            \sigma_i = \frac{\gamma \rho}{\|K_i\|}, \qquad
            \tau = \rho \min_i \frac{p_i}{\sigma_i \|K_i\|^{2}},

        so only the primal/dual balance is optimised. The search is performed over
        :math:`\log\gamma` using ``skopt.gp_minimize``, so `scikit-optimize` must be installed
        to use this rule.

        This is a computationally expensive step size rule, as it requires running the SPDHG algorithm for a number of iterations for each evaluation of the objective function in the Bayesian optimisation. It is recommended to use this step size rule where you are memory constrained but less time constrained.
        """

    def __init__(self, gamma_bounds=None, n_initial_points=5, n_calls=20, n_iterations=None, seed=None, plot=False):
        '''Initialises the step size rule'''
        super().__init__(gamma_bounds=gamma_bounds, n_initial_points=n_initial_points,
                         n_calls=n_calls, n_iterations=n_iterations, seed=seed, plot=plot)
        self.rho = 0.99
        self._pristine_sampler = None

    def _default_n_iterations(self, algorithm):
        '''Default number of iterations per objective evaluation when ``n_iterations`` is None. Overwrites the base class method.'''
        return 10 * len(algorithm._norms)

    def _step_sizes_from_gamma(self, algorithm, gamma):
        '''Map the scalar ratio ``gamma`` to ``(tau, sigma)``.'''
        sigma = _spdhg_sigma_from_gamma(gamma, self.rho, algorithm._norms)
        tau = _spdhg_tau_from_sigma(
            sigma, algorithm._norms, algorithm._prob_weights, self.rho)
        return tau, sigma

    def _algorithm_set_up(self, algorithm, step_size):
        """Set up the algorithm with the given step size.

        The sampler and probability weights are passed back explicitly because
        ``SPDHG.set_up`` would otherwise reset them to the uniform default. Each trial
        is given an identical copy of the caller's sampler, so that every gamma is
        scored on the same sequence of subsets rather than on wherever the previous
        trial happened to leave the generator.
        """
        if self._pristine_sampler is None:
            self._pristine_sampler = deepcopy(algorithm._sampler)

        algorithm.set_up(initial=algorithm.initial, f=algorithm.f, g=algorithm.g,
                         operator=algorithm.operator, step_size=step_size,
                         sampler=deepcopy(self._pristine_sampler),
                         prob_weights=algorithm._prob_weights)
        algorithm._reset_iteration_state()


class _SPDHGAdaptiveStepSizeBase(StepSizeRule):
    r"""Shared machinery for the adaptive SPDHG step-size rules of :cite:`chambolle2023stochastic`.

    The two concrete rules (:class:`SPDHGAdaptiveStepSizeBalancing` and
    :class:`SPDHGAdaptiveStepSizeAngle`) only *rescale* the step sizes between iterations,
    keeping the standard extrapolated SPDHG update. Following equation (2.6) of
    :cite:`chambolle2023stochastic`, a single balancing scalar :math:`\gamma` is used at
    each iteration,

    .. math::

        \tau \leftarrow \tau/\gamma, \qquad \sigma_i \leftarrow \gamma\,\sigma_i
        \quad (i = 1,\dots,n),

    so the same factor is applied to every dual step size and the products
    :math:`\tau\sigma_i` are preserved. Both rules are *balancing only* — there is no
    backtracking inner loop, so the
    step sizes are adjusted once per iteration from quantities measured on the subset that
    was sampled that iteration.

    This base class is not meant to be used directly. It owns the buffers required to form
    the per-subset primal/dual increments and defers the actual decision to
    :meth:`_rebalance`, implemented by each subclass. 

    Notes
    -----
    This rule stores three extra domain copies, and two range copies. When
    ``auto_stop=True`` the adaptive updates are switched off, and the extra storage
    released, once either the step sizes have been unchanged for more than
    ``auto_stop_patience`` consecutive iterations or the adaptation strength
    :math:`\alpha` has decayed below ``alpha_tolerance``.

    This method is memory intensive. It is recommended to use this step size rule where you are more time constrained but less memory constrained. For the opposite case, where you are more memory constrained but less time constrained, we recommend using the :class:`SPDHGBayesOptimisationStepSize` step size rule, which is a Bayesian optimisation step size rule that chooses the best step sizes at the beginning of the algorithm and then keeps them constant throughout the iterations.
    
    
    """

    def __init__(self, initial_step_size=[None, None], initial_alpha=0.95, eta=0.995,
                 auto_stop=True, auto_stop_patience=None, alpha_tolerance=1e-3):
        '''Initialises the shared adaptive SPDHG step size rule'''
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, "
                "initial_step_size = {}".format(initial_step_size))
        self.initial_step_size = initial_step_size
        self.alpha = initial_alpha
        self.eta = eta
        self.auto_stop = auto_stop
        self.auto_stop_patience = auto_stop_patience
        self.alpha_tolerance = alpha_tolerance
        self.tolerance = 1e-6
        self.count = 0
        self.adaptive = True
        # buffers allocated in get_initial_step_size
        self.x_prev = None
        self.y_prev = None
        self.adj_tmp = None
        self.v_tmp = None
        self.forward_tmp = None

    def _resolve_initial(self, algorithm):
        '''Resolve the initial (tau, sigma) pair, filling in any missing value.'''
        gamma = 1.
        rho = 0.99
        tau = self.initial_step_size[0]
        sigma = self.initial_step_size[1]
        if sigma is not None and isinstance(sigma, Number):
            # a single dual step size is broadcast to every operator
            sigma = [sigma] * algorithm._ndual_subsets
        if sigma is None and tau is None:
            sigma = _spdhg_sigma_from_gamma(gamma, rho, algorithm._norms)
            tau = _spdhg_tau_from_sigma(
                sigma, algorithm._norms, algorithm._prob_weights, rho)
        elif sigma is None:
            sigma = [rho * pi / (tau * ni**2)
                     for ni, pi in zip(algorithm._norms, algorithm._prob_weights)]
        elif tau is None:
            tau = _spdhg_tau_from_sigma(
                sigma, algorithm._norms, algorithm._prob_weights, rho)
        if self.auto_stop and self.auto_stop_patience is None:
            self.auto_stop_patience = 10 * len(algorithm._norms)
        return tau, sigma

    def get_initial_step_size(self, algorithm):
        ''' Allocate the buffers required to form the primal and dual increments, and return the initial step sizes.'''
        tau, sigma = self._resolve_initial(algorithm)
        self.x_prev = algorithm.x.copy()
        self.y_prev = algorithm._y_old.copy()
        self.adj_tmp = algorithm.operator.domain_geometry().allocate(0)
        self.v_tmp = algorithm.operator.domain_geometry().allocate(0)
        self.forward_tmp = algorithm.operator.range_geometry().allocate(0)
        return tau, sigma

    def get_step_size(self, algorithm):
        '''Return the current step sizes, and update them if the adaptive rule is still active.'''
        if self.adaptive:
            i = algorithm._index
            p_i = algorithm._prob_weights[i]
            tau = algorithm._tau
            sigma = algorithm._sigma

            # primal increment  Delta x = x^k - x^{k+1}
            algorithm.x.sapyb(-1.0, self.x_prev, 1.0, out=self.x_prev)
            # dual increment on the sampled subset  Delta y_i = y^k_i - y^{k+1}_i
            self.y_prev[i].subtract(algorithm._y_old[i], out=self.y_prev[i])
            # back-projected dual increment  A_i^* Delta y_i
            algorithm.operator[i].adjoint(self.y_prev[i], out=self.adj_tmp)
            # primal residual direction  q = Delta x / tau - (1/p_i) A_i^* Delta y_i
            self.x_prev.sapyb(1.0 / tau, self.adj_tmp, -
                              1.0 / p_i, out=self.v_tmp)

            # Relative test: the increments scale with the problem, so an absolute
            # threshold would silence the rule entirely on rescaled data.
            if (self.x_prev.norm() > self.tolerance * algorithm.x.norm()
                    and self.y_prev[i].norm()
                    > self.tolerance * algorithm._y_old[i].norm()):
                tau_new, sigma_new, changed = self._rebalance(
                    algorithm, i, p_i, tau, sigma)
                if not changed:
                    self.count += 1
            else:
                log.debug('SPDHG adaptive step size: increments below relative '
                          'tolerance, no change to step sizes.')
                tau_new, sigma_new = tau, sigma
                self.count += 1

            # refresh the stored iterates for the next call (only subset i changed)
            self.x_prev.fill(algorithm.x)
            self.y_prev[i].fill(algorithm._y_old[i])

            if self.auto_stop:
                # Two ways the rule can be finished: the step sizes have settled inside
                # the band, or alpha has decayed so far that a rebalance would no longer
                # move them (both rules rescale by 1 -/+ alpha, which tends to 1).
                if self.count > self.auto_stop_patience:
                    self._stop_adapting(
                        'step sizes unchanged for {} iterations'.format(
                            self.auto_stop_patience))
                elif self.alpha < self.alpha_tolerance:
                    self._stop_adapting(
                        'adaptation strength alpha = {} has decayed below '
                        'alpha_tolerance = {}'.format(
                            self.alpha, self.alpha_tolerance))

            return tau_new, sigma_new

        return algorithm._tau, algorithm._sigma

    def _stop_adapting(self, reason):
        '''Switch off the adaptive updates and release the working buffers.'''
        self.adaptive = False
        log.debug('SPDHG adaptive step size: automatically stopping updates, '
                  '{}.'.format(reason))
        del self.x_prev, self.y_prev, self.adj_tmp, self.v_tmp, self.forward_tmp

    def _rebalance(self, algorithm, i, p_i, tau, sigma):
        """Decide and apply the step-size rescaling for one iteration.

        Returns
        -------
        tuple
            ``(tau, sigma, changed)`` where ``sigma`` is the list of dual step sizes and
            ``changed`` is ``True`` if the step sizes were rescaled this iteration.
        """
        raise NotImplementedError


class SPDHGAdaptiveStepSizeBalancing(_SPDHGAdaptiveStepSizeBase):
    r"""Adaptively balances the SPDHG primal and dual step sizes by tracking the primal and dual progress, following rule (a) of :cite:`chambolle2023stochastic`.

    This is an SPDHG-compatible step-size rule (:class:`SPDHGAdaptiveStepSizeBalancing`
    implements A-SPDHG rule (a), Algorithm 3.1 of :cite:`chambolle2023stochastic`). It is
    the stochastic, minibatch generalisation of the deterministic adaptive PDHG rule of
    :cite:`goldstein2013adaptive` (:class:`PDHGAdaptiveStepSize2013`), to which it reduces
    when there is a single operator.

    At the end of each iteration, with sampled subset :math:`i`, sampling probability
    :math:`p_i`, primal increment :math:`\Delta x = x^{k} - x^{k+1}` and dual increment
    :math:`\Delta y_i = y^{k}_i - y^{k+1}_i`, two :math:`\ell_1` "progress" residuals are
    formed (equation (3.6) of :cite:`chambolle2023stochastic`),

    .. math::

        v = \left\| \frac{\Delta x}{\tau}
                    - \frac{1}{p_i} A_i^{*}\Delta y_i \right\|_1, \qquad
        d = \frac{1}{p_i}\left\| \frac{\Delta y_i}{\sigma_i}
                    - A_i \Delta x \right\|_1 ,

    which track the primal and dual subgradients respectively. In both cases the
    :math:`1/p_i` weight makes the quantity measured on the sampled subset an unbiased
    estimate of the corresponding full (deterministic) residual: for :math:`v` it
    reweights the only nonzero block of :math:`A^{*}\Delta y`, and for :math:`d` it
    reweights the sampled block's contribution to :math:`\sum_j \|d_j\|_1`. 
    They are compared using the balancing scale :math:`s`
    and the band parameter :math:`\delta > 1`:

    - if :math:`v > s\,\delta\, d` the primal is making less progress, so the primal step
      is boosted and the dual shrunk,
      :math:`\tau \leftarrow \tau/(1-\alpha)`, :math:`\sigma_i \leftarrow (1-\alpha)\sigma_i`;
    - if :math:`v < s\, d/\delta` the opposite tilt is applied,
      :math:`\tau \leftarrow (1-\alpha)\tau`, :math:`\sigma_i \leftarrow \sigma_i/(1-\alpha)`;
    - otherwise the step sizes are left unchanged.

    Every dual step size :math:`\sigma_i` is rescaled by the same factor, and each time a
    tilt is applied the adaptation strength decays, :math:`\alpha \leftarrow \eta\,\alpha`,
    so the changes become progressively smaller (guaranteeing the summable-deviation
    condition of :cite:`chambolle2023stochastic`). The products :math:`\tau\sigma_i` are
    preserved.

    Parameters
    ----------
    initial_step_size : list of length two, optional, default=[None, None]
        Initial primal and dual step sizes ``[tau, sigma]``. ``tau`` is a positive scalar
        and ``sigma`` is either a positive scalar (broadcast to every operator) or a list
        of one positive number per operator. Any entry left as ``None`` is filled in from
        the operator norms and sampling probabilities using the standard SPDHG relations
        (as in :class:`SPDHGConstantStepSize`).
    initial_alpha : positive :obj:`float`, optional, default=0.95
        Initial value of the adaptation strength :math:`\alpha \in (0,1)`. Each rebalance
        multiplies or divides :math:`\tau` by :math:`1-\alpha`, so :math:`\alpha` close to
        one gives a *large* rescaling (:math:`\alpha = 0.95` changes :math:`\tau` by a
        factor of 20) and :math:`\alpha` close to zero a gentle one.
    eta : positive :obj:`float`, optional, default=0.95
        The decay factor :math:`\eta \in (0,1)` applied to :math:`\alpha` each time the
        step sizes are rebalanced. 
    delta : positive :obj:`float`, greater than one, optional, default=1.5
        The band parameter :math:`\delta` setting how far apart the primal and dual residuals are allowed to drift before rebalancing.
    s : positive :obj:`float`, optional, default=norm of the operator
        The balancing scale :math:`s` used to compare the primal and dual residuals. Defaults to the operator norm :math:`\|A\|`, as recommended in :cite:`chambolle2023stochastic`.
    auto_stop : :obj:`bool`, optional, default=True
        If True, the adaptive updates stop and the extra storage is released once either the step sizes have been unchanged for more than ``auto_stop_patience`` consecutive iterations or :math:`\alpha` has decayed below ``alpha_tolerance``.
    auto_stop_patience : :obj:`int`, optional, default=``10 * n_operators``
        Number of consecutive iterations with no change to the step sizes that are tolerated before the adaptive updates are stopped (only used when ``auto_stop=True``). If left as ``None`` it is set to ten times the number of operators, so that the patience scales with the number of iterations needed to sample every subset.
    alpha_tolerance : positive :obj:`float`, optional, default=1e-3
        Threshold on the adaptation strength :math:`\alpha` below which the adaptive updates are stopped (only used when ``auto_stop=True``). Once :math:`\alpha` has decayed this far a rebalance changes the step sizes by a relative amount of roughly :math:`\alpha`, so the rule keeps paying for the extra operator applications and storage while making no practical difference. Set to ``0`` to disable this criterion and rely on ``auto_stop_patience`` alone.

    Notes
    -----
    The dual step size ``sigma`` is a list with one entry per operator, while ``tau`` is a
    scalar; :meth:`get_initial_step_size` and :meth:`get_step_size` return the tuple
    ``(tau, sigma)``. Computing the residuals costs one extra forward and one extra adjoint
    application per iteration (the overhead discussed in :cite:`chambolle2023stochastic`).

    This is a memory intensive step size rule, as it requires storing three extra domain copies and two extra range copies. It is recommended to use this step size rule where you are more time constrained but less memory constrained. For the opposite case, where you are more memory constrained but less time constrained, we recommend using the :class:`SPDHGBayesOptimisationStepSize` step size rule, which is a Bayesian optimisation step size rule that chooses the best step sizes at the beginning of the algorithm and then keeps them constant throughout the iterations.

    See Also
    --------
    SPDHGAdaptiveStepSizeAngle : The companion angle-alignment rule (rule (b)) from the same paper.
    PDHGAdaptiveStepSize2013 : The deterministic PDHG rule this generalises.

    Reference
    ---------
    Chambolle, A., Delplancke, C., Ehrhardt, M.J., Schönlieb, C.-B. and Tang, J., 2023. Stochastic Primal-Dual Hybrid Gradient Algorithm with Adaptive Step-Sizes. arXiv preprint arXiv:2301.02511. :cite:`chambolle2023stochastic`
    """

    def __init__(self, initial_step_size=[None, None], initial_alpha=0.95, eta=0.95,
                 delta=1.5, s=None, auto_stop=True, auto_stop_patience=None,
                 alpha_tolerance=1e-3):
        '''Initialises the step size rule'''
        super().__init__(initial_step_size=initial_step_size, initial_alpha=initial_alpha,
                         eta=eta, auto_stop=auto_stop, auto_stop_patience=auto_stop_patience,
                         alpha_tolerance=alpha_tolerance)
        self.delta = delta
        self.s = s

    def _rebalance(self, algorithm, i, p_i, tau, sigma):
        '''Decide and apply the step-size rescaling for one iteration, using the primal/dual residuals to balance the step sizes. Returns the new step sizes and a boolean indicating whether they were changed.'''
        if self.s is None:
            self.s = algorithm.operator.norm()  # default balancing scale, ||A||

        v = self.v_tmp.abs().sum()
        algorithm.operator[i].direct(self.x_prev, out=self.forward_tmp[i])
        self.y_prev[i].sapyb(1.0 / (p_i * sigma[i]),
                             self.forward_tmp[i], -1.0/p_i, out=self.y_prev[i])
        d = self.y_prev[i].abs().sum()

        log.debug(
            'SPDHG adaptive balancing: v = {}, d = {}, s = {}'.format(v, d, self.s))
        if v > self.s * self.delta * d:
            factor = 1 - self.alpha
            tau /= factor
            sigma = [s * factor for s in sigma]
            self.alpha *= self.eta
            self.count = 0
            log.debug(
                'SPDHG adaptive balancing: v > s*delta*d, increasing tau and decreasing sigma, tau = {}, sigma = {}'.format(tau, sigma))
            return tau, sigma, True
        elif v < self.s * d / self.delta:
            factor = 1 - self.alpha
            tau *= factor
            sigma = [s / factor for s in sigma]
            self.alpha *= self.eta
            self.count = 0
            log.debug(
                'SPDHG adaptive balancing: v < s*d/delta, decreasing tau and increasing sigma, tau = {}, sigma = {}'.format(tau, sigma))
            return tau, sigma, True

        log.debug(
            'SPDHG adaptive balancing: v and d within band, no change to step sizes.')
        return tau, sigma, False


class SPDHGAdaptiveStepSizeAngle(_SPDHGAdaptiveStepSizeBase):
    r"""Adaptively adjusts the SPDHG primal and dual step sizes using the alignment between successive primal directions, following rule (b) of :cite:`chambolle2023stochastic`.

    This is an SPDHG-compatible step-size rule (A-SPDHG rule (b), Algorithm 3.2 of
    :cite:`chambolle2023stochastic`). It is the stochastic extension of the angle-based
    adaptive PDHG scheme of :cite:`yokota2017efficient`, and an alternative to the
    residual-balancing :class:`SPDHGAdaptiveStepSizeBalancing`.

    At the end of each iteration, with sampled subset :math:`i`, sampling probability
    :math:`p_i`, primal increment :math:`\Delta x = x^{k} - x^{k+1}` and dual increment
    :math:`\Delta y_i = y^{k}_i - y^{k+1}_i`, the primal-residual direction is formed
    (equation (3.8) of :cite:`chambolle2023stochastic`),

    .. math::

        q = \frac{\Delta x}{\tau} - \frac{1}{p_i} A_i^{*}\Delta y_i ,

    and the cosine of the angle between :math:`\Delta x` and :math:`q` is measured
    (equation (3.9)),

    .. math::

        w = \frac{\langle \Delta x, q\rangle}{\|\Delta x\|_2\,\|q\|_2} .

    The step sizes are then adjusted against the threshold :math:`c` (close to one):

    - if :math:`w \ge c` the directions are well aligned, so the primal step is increased,
      :math:`\tau \leftarrow (1+\alpha)\tau`, :math:`\sigma_i \leftarrow \sigma_i/(1+\alpha)`;
    - if :math:`w < 0` the directions oppose, so the primal step is decreased,
      :math:`\tau \leftarrow \tau/(1+\alpha)`, :math:`\sigma_i \leftarrow (1+\alpha)\sigma_i`;
    - otherwise the step sizes are left unchanged.

    Every dual step size :math:`\sigma_i` is rescaled by the same factor, and each time the
    step sizes change the adaptation strength decays, :math:`\alpha \leftarrow \eta\,\alpha`.
    The products :math:`\tau\sigma_i` are preserved.

    Parameters
    ----------
    initial_step_size : list of length two, optional, default=[None, None]
        Initial primal and dual step sizes ``[tau, sigma]``. ``tau`` is a positive scalar
        and ``sigma`` is either a positive scalar (broadcast to every operator) or a list
        of one positive number per operator. Any entry left as ``None`` is filled in from
        the operator norms and sampling probabilities using the standard SPDHG relations.
    initial_alpha : positive :obj:`float`, optional, default=1.0
        Initial value of the adaptation strength :math:`\alpha`. Each change multiplies or divides :math:`\tau` by :math:`1+\alpha`, so :math:`\alpha_0 = 1` (the value used in the paper) rescales :math:`\tau` by a factor of two.
    eta : positive :obj:`float`, optional, default=0.995
        The decay factor :math:`\eta \in (0,1)` applied to :math:`\alpha` each time the step sizes change. Since :math:`\alpha` decays only when the step sizes change, this bounds the total drift of :math:`\tau` through :math:`\sum_k \alpha_k = \alpha_0/(1-\eta)`.
    c : :obj:`float`, optional, default=0.999
        The cosine threshold :math:`c` above which the primal step size is increased. Should be close to one in high-dimensional problems, as recommended in :cite:`chambolle2023stochastic`.
    auto_stop : :obj:`bool`, optional, default=True
        If True, the adaptive updates stop and the extra storage is released once either the step sizes have been unchanged for more than ``auto_stop_patience`` consecutive iterations or :math:`\alpha` has decayed below ``alpha_tolerance``.
    auto_stop_patience : :obj:`int`, optional, default=``10 * n_operators``
        Number of consecutive iterations with no change to the step sizes that are tolerated before the adaptive updates are stopped (only used when ``auto_stop=True``). If left as ``None`` it is set to ten times the number of operators, so that the patience scales with the number of iterations needed to sample every subset.
    alpha_tolerance : positive :obj:`float`, optional, default=1e-3
        Threshold on the adaptation strength :math:`\alpha` below which the adaptive updates are stopped (only used when ``auto_stop=True``). Once :math:`\alpha` has decayed this far a rebalance changes the step sizes by a relative amount of roughly :math:`\alpha`, so the rule keeps paying for the extra operator applications and storage while making no practical difference. Set to ``0`` to disable this criterion and rely on ``auto_stop_patience`` alone.

    Notes
    -----
    The dual step size ``sigma`` is a list with one entry per operator, while ``tau`` is a
    scalar; :meth:`get_initial_step_size` and :meth:`get_step_size` return the tuple
    ``(tau, sigma)``. Computing the direction costs one extra adjoint application per
    iteration.

    This is a memory intensive step size rule, as it requires storing three extra domain copies and two extra range copies. It is recommended to use this step size rule where you are more time constrained but less memory constrained. For the opposite case, where you are more memory constrained but less time constrained, we recommend using the :class:`SPDHGBayesOptimisationStepSize` step size rule, which is a Bayesian optimisation step size rule that chooses the best step sizes at the beginning of the algorithm and then keeps them constant throughout the iterations.

    See Also
    --------
    SPDHGAdaptiveStepSizeBalancing : The companion residual-balancing rule (rule (a)) from the same paper.

    Reference
    ---------
    Chambolle, A., Delplancke, C., Ehrhardt, M.J., Schönlieb, C.-B. and Tang, J., 2023. Stochastic Primal-Dual Hybrid Gradient Algorithm with Adaptive Step-Sizes. arXiv preprint arXiv:2301.02511. :cite:`chambolle2023stochastic`

    Yokota, T. and Hontani, H., 2017. An efficient method for adapting step-size parameters of primal-dual hybrid gradient method in application to total variation regularization. 2017 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 973-979. :cite:`yokota2017efficient`
    """

    def __init__(self, initial_step_size=[None, None], initial_alpha=1.0, eta=0.995,
                 c=0.999, auto_stop=True, auto_stop_patience=None, alpha_tolerance=1e-3):
        '''Initialises the step size rule'''
        super().__init__(initial_step_size=initial_step_size, initial_alpha=initial_alpha,
                         eta=eta, auto_stop=auto_stop, auto_stop_patience=auto_stop_patience,
                         alpha_tolerance=alpha_tolerance)
        self.c = c

    def _rebalance(self, algorithm, i, p_i, tau, sigma):
        '''Decide and apply the step-size rescaling for one iteration, using the cosine of the angle between successive primal directions. Returns the new step sizes and a boolean indicating whether they were changed.'''
        # cosine of the angle between Delta x (held in x_prev) and the primal
        # residual direction q (held in v_tmp), both set by the base class.
        w = self.x_prev.dot(self.v_tmp) / \
            (self.x_prev.norm() * self.v_tmp.norm())

        log.debug(
            'SPDHG adaptive angle: cos(angle) w = {}, index = {}'.format(w, i))
        if w >= self.c:
            factor = 1 + self.alpha
            tau_new = tau * factor
            sigma_new = [si / factor for si in sigma]
            self.alpha *= self.eta
            self.count = 0
            log.debug(
                'SPDHG adaptive angle: increasing step sizes, tau -> {}, sigma -> {}'.format(tau_new, sigma_new))
            return tau_new, sigma_new, True
        elif w < 0:
            factor = 1 + self.alpha
            tau_new = tau / factor
            sigma_new = [si * factor for si in sigma]
            self.alpha *= self.eta
            self.count = 0
            log.debug(
                'SPDHG adaptive angle: decreasing step sizes, tau -> {}, sigma -> {}'.format(tau_new, sigma_new))
            return tau_new, sigma_new, True
        return tau, sigma, False
