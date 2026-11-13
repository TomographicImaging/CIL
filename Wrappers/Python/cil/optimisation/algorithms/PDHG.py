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
# Hok Shing Wong (University of Bath)
# Joshua DM Hellier (University of Manchester) [documentation]

from cil.framework import DataContainer, BlockDataContainer
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import (
    StepSizeRule,
    PDHGStepSizeRule,
    PDHGStronglyConvexUpdate,
    PDHGConstantStepSize,
)
import warnings
import numpy as np
from numbers import Number
import logging


log = logging.getLogger(__name__)


def _check_no_positional_step_sizes(step_size, initial, name):
    """Raise a clear error for calls that pass ``tau``/``sigma`` positionally.

    Before ``step_size`` was introduced, the 4th and 5th positional arguments
    of PDHG and SPDHG were ``tau`` and ``sigma``. They are now ``step_size``
    and ``initial``, so such calls would otherwise fail with a confusing
    error. A number is never a valid ``step_size``, and ``initial`` must
    contain DataContainers, so neither check can misread a valid call.
    """
    def is_number(x):
        return isinstance(x, Number) and not isinstance(x, bool)

    old_sigma = initial is not None and (
        is_number(initial) or (isinstance(initial, (list, tuple))
                               and any(is_number(v) for v in initial)))
    if is_number(step_size) or old_sigma:
        raise ValueError(
            "{0} no longer accepts `tau` and `sigma` as positional arguments: "
            "the 4th and 5th positional arguments are now `step_size` and "
            "`initial`. Please pass the step sizes as `step_size=(tau, "
            "sigma)`, e.g. `{0}(f, g, operator, step_size=(tau, sigma))`, or "
            "use a step size rule.".format(name))


class PDHG(Algorithm):

    r"""Primal Dual Hybrid Gradient (PDHG) algorithm, see :cite:`CP2011`, :cite:`EZXC2010`.

    PDHG minimises objectives of the form:

    .. math::
        \min_{x\in X} f(Kx) + g(x),

    where :math:`f` and the regulariser :math:`g` need to be proper, convex and
    lower semi-continuous. The function :math:`f` and the convex conjugate of
    :math:`g` must also have calculable proximal methods.

    Parameters
    ----------
    f : Function
        A convex function with a "simple" proximal method of its conjugate. This function must map from the operator range to the Reals, as :math: `f(Kx)` will be the contribution to the total objective. See below for details.
    g : Function
        A convex function with a "simple" proximal. This function must map from the operator domain to the Reals. See below for details.
    operator : LinearOperator
        A Linear Operator.
    step_size : :class:`~cil.optimisation.utilities.PDHGStepSizeRule`, or `list` or `tuple` of length two, optional, default=None
        Either a :class:`~cil.optimisation.utilities.PDHGStepSizeRule` or a
        `list` or `tuple`  of (tau, sigma) where sigma is the step size for the
        dual problem and tau is the step size for the primal problem. The step
        sizes can be either None,  scalar or array-objects. If not provided,
        default values will be set based on the operator norm as described
        below. See the PDHG step-size rules in
        :mod:`cil.optimisation.utilities.StepSizeMethods` for the adaptive and
        Bayesian-optimisation alternatives to a fixed step size.
    initial : `DataContainer`, or `list` or `tuple` of `DataContainer`s, optional, default is a DataContainer of zeros for both primal and dual variables
        Initial point for the PDHG algorithm. If just one data container is provided, it is used for the primal and the dual variable is initialised as zeros.  If a list or tuple is passed,  the first element is used for the primal variable and the second one for the dual variable. If either of the two is not provided, it is initialised as a DataContainer of zeros.

    **kwargs:
        update_objective_interval : :obj:`int`, optional, default=1
            Evaluates objectives, e.g., primal/dual/primal-dual gap every ``update_objective_interval``.
        check_convergence : :obj:`boolean`, default=True
            Checks scalar sigma and tau values satisfy convergence criterion and warns if not satisfied. Can be computationally expensive for custom sigma or tau values.
        theta :  Float between 0 and 1, default 1.0
            Relaxation parameter for the over-relaxation of the primal variable.

    Example
    -------

    In our CIL-Demos repository (https://github.com/TomographicImaging/CIL-Demos) you can find examples using the PDHG algorithm for different imaging problems, such as Total Variation denoising, Total Generalised Variation inpainting
    and Total Variation Tomography reconstruction. More examples can also be found in :cite:`Jorgensen_et_al_2021`, :cite:`Papoutsellis_et_al_2021`.


    Notes
    -----

    A first-order primal-dual algorithm for convex optimization problems with known saddle-point structure with applications in imaging.

    The general problem considered in the PDHG algorithm is the generic saddle-point problem

    .. math:: \min_{x\in X}\max_{y\in Y} \langle Kx, y \rangle + g(x) - f^{*}(y)

    where :math:`f` and :math:`g` are convex functions with "simple" proximal operators.

    :math:`X` and :math:`Y` are two two finite-dimensional vector spaces with an inner product and representing the domain of :math:`g` and :math:`f^{*}`, the convex conjugate of :math:`f`, respectively.

    The operator :math:`K` is a continuous linear operator with operator norm defined as

    .. math:: \|K\| = \max\{ \|Kx\| : x\in X, \|x\|\leq1\}


    The saddle point problem is decomposed into the primal problem:

    .. math:: \min_{x\in X} f(Kx) + g(x),

    and its corresponding dual problem

    .. math:: \max_{y\in Y} - g^{*}(-K^{*}y) - f^{*}(y).

    The PDHG algorithm consists of three steps:

    * gradient ascent step for the dual problem,
    * gradient descent step for the primal problem and
    * an over-relaxation of the primal variable.

    .. math::

        y^{n+1} = \mathrm{prox}_{\sigma f^{*}}(y^{n} + \sigma K \bar{x}^{n})

    .. math::

        x^{n+1} = \mathrm{prox}_{\tau g}(x^{n} - \tau K^{*}y^{n+1})

    .. math::

        \bar{x}^{n+1} = x^{n+1} + \theta (x^{n+1} - x^{n})

    Notes
    -----

    - Convergence is guaranteed if :math:`\theta` = 1.0,  the operator norm :math:`\|K\|`, \the dual step size :math:`\sigma` and the primal step size :math:`\tau`, satisfy the following inequality:

    .. math::

        \tau \sigma \|K\|^2 < 4/3

    For reference, see Li, Y. and Yan, M., 2022. On the improved conditions for some primal-dual algorithms. arXiv preprint arXiv:2201.00139.

    - By default, the step sizes :math:`\sigma` and :math:`\tau` are positive
      scalars. If ``step_size`` is ``None``, or a tuple ``(tau, sigma)`` with
      one or both entries ``None``, they are defined as below:

      * If ``sigma`` is ``None`` and ``tau`` is ``None``:

      .. math::

        \sigma = \frac{1}{\|K\|},  \tau = \frac{1}{\|K\|}

      * If ``tau`` is ``None``:

      .. math::

        \tau = \frac{1}{\sigma\|K\|^{2}}

      * If ``sigma`` is ``None``:

      .. math::

        \sigma = \frac{1}{\tau\|K\|^{2}}


    - To monitor the convergence of the algorithm, we compute the primal/dual
      objectives and the primal-dual gap in :meth:`objective`.\

      The primal objective is

      .. math::

        f(Kx) + g(x)

      and the dual objective is

      .. math::

        - g^{*}(-K^{*}y) - f^{*}(y)

      The primal-dual gap (or duality gap) is

      .. math::

        f(Kx) + g(x) + g^{*}(-K^{*}y) + f^{*}(y)

      and measures how close is the primal-dual pair (x,y) to the primal-dual solution. It is always non-negative and is used to monitor convergence of the PDHG algorithm. \
      For more information, see `Duality Gap <https://en.wikipedia.org/wiki/Duality_gap>`_.


    Note
    ----

        - The primal objective is printed if `verbose=1`, ``pdhg.run(verbose=1)``.
        - All the objectives are printed if `verbose=2`, ``pdhg.run(verbose=2)``.

        Computing these objectives can be costly, so it is better to compute every some iterations. To do this, use ``update_objective_interval = #number``.



    """

    def __init__(self, f, g, operator, step_size=None,  initial=None,
                 **kwargs):
        """Initialisation of the PDHG algorithm"""

        _check_no_positional_step_sizes(step_size, initial, 'PDHG')
        self.initial = initial
        self._theta = kwargs.pop('theta', 1.0)
        if self._theta > 1 or self._theta < 0:
            raise ValueError(
                "The relaxation parameter theta must be in the range [0,1], "
                "passed theta = {}".format(self._theta))

        sigma = kwargs.pop('sigma', None)  # To be deprecated
        tau = kwargs.pop('tau', None)  # To be deprecated
        if sigma is not None or tau is not None:  # To be deprecated
            if step_size is not None:
                warnings.warn(
                    "The parameters `sigma` and `tau` are being deprecated "
                    "and, because `step_size` was also passed, they are "
                    "being ignored. Please pass these as part of the "
                    "`step_size` argument, either as a tuple of (tau, sigma) "
                    "or using a compatible step size rule.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "The parameters `sigma` and `tau` are being deprecated. "
                    "In the future, please pass these as part of the "
                    "`step_size` argument, either as a tuple of (tau, sigma) "
                    "or using a compatible step size rule.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                step_size = (tau, sigma)

        gamma_g = kwargs.pop('gamma_g', None)  # To be deprecated
        gamma_fconj = kwargs.pop('gamma_fconj', None)  # To be deprecated
        if gamma_g is not None or gamma_fconj is not None:  # To be deprecated
            warnings.warn(
                "The parameters `gamma_g` and `gamma_fconj` are being "
                "deprecated. In the future, if you would like to utilise "
                "strong convexity you should use the step size method "
                "cil.optimisation.utilities.StepSizeMethods."
                "PDHGStronglyConvexUpdate.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(step_size, (tuple, list)):
                initial_step_size = step_size
            else:
                initial_step_size = (None, None)
            step_size = PDHGStronglyConvexUpdate(
                initial_step_size=initial_step_size,
                gamma_g=gamma_g,
                gamma_fconj=gamma_fconj,
            )

        self._check_convergence = kwargs.pop('check_convergence', True)

        super().__init__(**kwargs)

        self.set_up(f=f, g=g, operator=operator,
                    step_size=step_size,  initial=initial)

    @property
    def tau(self):
        """The primal step-size - Returns the currently being used step size
        for the primal problem. Note that this can be updated at each iteration
        if a step size rule is used.
        """
        return self._tau

    @property
    def sigma(self):
        """The dual step-size - Returns the currently being used step size for
        the dual problem. Note that this can be updated at each iteration if a
        step size rule is used.
        """
        return self._sigma

    @property
    def theta(self):
        """The relaxation parameter for the over-relaxation of the primal variable """
        return self._theta

    def set_up(self, f, g, operator, step_size=None,  initial=None):
        """Initialisation of the algorithm

        Parameters
        ----------
        f : Function
            A convex function with a "simple" proximal method of its conjugate.
        g : Function
            A convex function with a "simple" proximal.
        operator : LinearOperator
            A Linear Operator.
        initial : `DataContainer`, or `list` or `tuple` of `DataContainer`s, optional, default is a DataContainer of zeros for both primal and dual variables
            Initial point for the PDHG algorithm. If just one data container is provided, it is used for the primal and the dual variable is initialised as zeros.  If a list or tuple is passed,  the first element is used for the primal variable and the second one for the dual variable. If either of the two is not provided, it is initialised as a DataContainer of zeros.
        step_size : :class:`~cil.optimisation.utilities.PDHGStepSizeRule`, or `list` or `tuple` of length two, optional, default=None
            Either a :class:`~cil.optimisation.utilities.PDHGStepSizeRule` or a
            `list` or `tuple`  of (tau, sigma) where sigma is the step size for
            the dual problem and tau is the step size for the primal problem.
            The step sizes can be either scalar or array-objects. If not
            provided, default values will be set based on the operator norm, as
            described in the class documentation.


        """

        log.info("%s setting up", self.__class__.__name__)

        # Triplet (f, g, K)
        self.f = f
        self.g = g
        self.operator = operator

        if step_size is None:
            step_size = (None, None)
        if isinstance(step_size, PDHGStepSizeRule):
            self.step_size_rule = step_size
        elif isinstance(step_size, StepSizeRule):
            raise ValueError(
                "The step-size rule {} is not compatible with PDHG. PDHG "
                "requires a step-size rule "
                "derived from cil.optimisation.utilities."
                "PDHGStepSizeRule.".format(
                    type(step_size).__name__
                )
            )
        elif isinstance(step_size, (tuple, list)):
            self.step_size_rule = PDHGConstantStepSize(step_size=step_size)
        else:
            raise ValueError(
                "The `step_size` argument must be either None, a "
                "PDHGStepSizeRule or a tuple of (tau, sigma) where sigma is "
                "the step size for the dual problem and tau is the step size "
                "for the primal problem.")

        if isinstance(initial, (tuple, list)):
            if initial[0] is not None:
                self.x_old = initial[0].copy()
            else:
                self.x_old = self.operator.domain_geometry().allocate(0)

            if len(initial) > 1 and initial[1] is not None:
                self.y = initial[1].copy()
            else:
                self.y = self.operator.range_geometry().allocate(0)

        else:
            self.y = self.operator.range_geometry().allocate(0)
            if initial is None:
                self.x_old = self.operator.domain_geometry().allocate(0)
            else:
                self.x_old = initial.copy()

        self.x = self.x_old.copy()
        self.x_tmp = self.operator.domain_geometry().allocate(0)
        self.y_tmp = self.operator.range_geometry().allocate(0)

        self._tau, self._sigma = self.step_size_rule.get_initial_step_size(
            self)
        PDHGStepSizeRule.validate_step_sizes(
            self._tau, self._sigma, self.operator)

        if self._check_convergence:
            self.check_convergence()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def _update_previous_solution(self):
        """Swaps the references to the current and previous solution."""
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def get_output(self):
        " Returns the current solution. "
        return self.x_old

    def _pdhg_update(self):
        """Applies the primal-dual updates for one PDHG step (see the Notes in
        the
        class docstring for the update equations)."""

        # calculate x-bar and store in self.x_tmp
        self.x_old.sapyb((self.theta + 1.0), self.x, -
                         self.theta, out=self.x_tmp)

        # Gradient ascent for the dual variable
        self.operator.direct(self.x_tmp, out=self.y_tmp)

        self.y_tmp.sapyb(self.sigma, self.y, 1.0, out=self.y_tmp)

        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)

        # Gradient descent for the primal variable
        self.operator.adjoint(self.y, out=self.x_tmp)

        self.x_tmp.sapyb(-self.tau, self.x_old, 1.0, self.x_tmp)

        self.g.proximal(self.x_tmp, self.tau, out=self.x)

    def update(self):
        """Performs a single iteration of the PDHG algorithm"""
        self._pdhg_update()
        self.update_step_sizes()

    def update_step_sizes(self):
        """Updates the primal and dual step sizes at the end of an iteration,
        using the step-size rule ``step_size_rule``.
        """
        self._tau, self._sigma = self.step_size_rule.get_step_size(self)

    def set_step_sizes(self, sigma=None, tau=None):
        """Sets sigma and tau step-sizes for the PDHG algorithm. The step sizes
        can be either scalar or array-objects.

        .. deprecated::
            Pass the step sizes to PDHG with the ``step_size`` argument
            instead, either as a tuple ``(tau, sigma)`` or using a
            :class:`~cil.optimisation.utilities.PDHGConstantStepSize`.

        The user can set either, both or none. Values passed by the user will
        be accepted as long as they are positive numbers, or correct shape
        array like objects. Missing values are set from the operator norm, as
        described in the class documentation. If the current step-size rule is
        a :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`, the
        strong convexity acceleration is kept and restarted from the new step
        sizes.

        Parameters
        ----------
            sigma : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default=None
                Step size for the dual problem.
            tau : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default=None
                Step size for the primal problem.
        """
        warnings.warn(
            "`set_step_sizes` is being deprecated. In the future, please "
            "pass the step sizes to PDHG with the `step_size` argument, "
            "either as a tuple of (tau, sigma) or using a compatible step "
            "size rule.",
            category=DeprecationWarning, stacklevel=2)
        if isinstance(self.step_size_rule, PDHGStronglyConvexUpdate):
            rule = PDHGStronglyConvexUpdate(
                initial_step_size=(tau, sigma),
                gamma_g=self.step_size_rule.gamma_g,
                gamma_fconj=self.step_size_rule.gamma_fconj,
            )
        else:
            rule = PDHGConstantStepSize(step_size=(tau, sigma))
        tau, sigma = rule.get_initial_step_size(self)
        PDHGStepSizeRule.validate_step_sizes(tau, sigma, self.operator)
        self.step_size_rule = rule
        self._tau, self._sigma = tau, sigma

    @property
    def gamma_g(self):
        """The strongly convex constant for the function g, or ``None`` if the
        step-size rule is not a
        :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`.

        .. deprecated::
            Use the ``gamma_g`` attribute of a
            :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`
            step-size rule instead.
        """
        warnings.warn(
            "The PDHG property `gamma_g` is being deprecated. In the future, "
            "please use the step size rule "
            "cil.optimisation.utilities.PDHGStronglyConvexUpdate.",
            category=DeprecationWarning, stacklevel=2)
        return getattr(self.step_size_rule, 'gamma_g', None)

    @property
    def gamma_fconj(self):
        """The strongly convex constant for the convex conjugate of the
        function f, or ``None`` if the step-size rule is not a
        :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`.

        .. deprecated::
            Use the ``gamma_fconj`` attribute of a
            :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`
            step-size rule instead.
        """
        warnings.warn(
            "The PDHG property `gamma_fconj` is being deprecated. In the "
            "future, please use the step size rule "
            "cil.optimisation.utilities.PDHGStronglyConvexUpdate.",
            category=DeprecationWarning, stacklevel=2)
        return getattr(self.step_size_rule, 'gamma_fconj', None)

    def set_gamma_g(self, value):
        '''Set the value of the strongly convex constant for function `g`.

        .. deprecated::
            Pass a
            :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`
            step-size rule to PDHG with the ``step_size`` argument instead.

        The step-size rule is replaced by a
        :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate` starting
        from the current step sizes.

        Parameters
        ----------
            value : a positive number or None
                If None, nothing is changed.
        '''
        warnings.warn(
            "`set_gamma_g` is being deprecated. In the future, please pass "
            "the step size rule "
            "cil.optimisation.utilities.PDHGStronglyConvexUpdate to PDHG "
            "with the `step_size` argument.",
            category=DeprecationWarning, stacklevel=2)
        if value is None:
            return
        self._set_strongly_convex_rule(gamma_g=value, gamma_fconj=getattr(
            self.step_size_rule, 'gamma_fconj', None))

    def set_gamma_fconj(self, value):
        '''Set the value of the strongly convex constant for the convex
        conjugate of function `f`.

        .. deprecated::
            Pass a
            :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate`
            step-size rule to PDHG with the ``step_size`` argument instead.

        The step-size rule is replaced by a
        :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate` starting
        from the current step sizes.

        Parameters
        ----------
            value : a positive number or None
                If None, nothing is changed.
        '''
        warnings.warn(
            "`set_gamma_fconj` is being deprecated. In the future, please "
            "pass the step size rule "
            "cil.optimisation.utilities.PDHGStronglyConvexUpdate to PDHG "
            "with the `step_size` argument.",
            category=DeprecationWarning, stacklevel=2)
        if value is None:
            return
        self._set_strongly_convex_rule(gamma_g=getattr(
            self.step_size_rule, 'gamma_g', None), gamma_fconj=value)

    def _set_strongly_convex_rule(self, gamma_g, gamma_fconj):
        """Replaces the step-size rule by a
        :class:`~cil.optimisation.utilities.PDHGStronglyConvexUpdate` starting
        from the current step sizes. Used by the deprecated setters.
        """
        rule = PDHGStronglyConvexUpdate(
            initial_step_size=(self._tau, self._sigma),
            gamma_g=gamma_g,
            gamma_fconj=gamma_fconj,
        )
        rule.get_initial_step_size(self)
        self.step_size_rule = rule

    def check_convergence(self):
        """Check whether convergence criterion for PDHG is satisfied with scalar values of tau and sigma

        The criterion :math:`\\tau\\sigma\\|K\\|^{2} < 4/3` is only evaluated
        for :class:`~cil.optimisation.utilities.PDHGConstantStepSize`, and only
        when both step sizes are scalars. Rules with ``enforces_convergence =
        True``, such as
        :class:`~cil.optimisation.utilities.PDHGAdaptiveStepSize2013`, return
        ``True`` without a check, because their backtracking step enforces the
        corresponding condition at every iteration. For any other step-size
        rule the check is not implemented, so a warning is raised and ``False``
        returned.

        Returns
        -------
        Boolean
            True if convergence criterion is satisfied. False if not satisfied or convergence is unknown.

        Reference
        ----------
        Li, Y. and Yan, M., 2022. On the improved conditions for some primal-dual algorithms. arXiv preprint arXiv:2201.00139.

        """
        if isinstance(self.step_size_rule, PDHGConstantStepSize):
            if isinstance(self.tau, Number) and isinstance(self.sigma, Number):
                if self.sigma * self.tau * self.operator.norm()**2 > 4/3:
                    warnings.warn(
                        "Convergence criterion of PDHG for scalar step-sizes "
                        "is not satisfied.")
                    return False
                return True
            else:
                warnings.warn(
                    "Convergence criterion can only be checked for scalar "
                    "values of tau and sigma, tau={0}, sigma={1}".format(
                        self.tau, self.sigma
                    )
                )
        elif getattr(self.step_size_rule, 'enforces_convergence', False):
            return True
        else:
            warnings.warn(
                "Convergence checks not currently implemented for this type "
                "of step size rule.")
        return False

    def update_objective(self):
        """Evaluates the primal objective, the dual objective and the primal-dual gap."""
        self.operator.direct(self.x_old, out=self.y_tmp)
        f_eval_p = self.f(self.y_tmp)
        g_eval_p = self.g(self.x_old)
        p1 = f_eval_p + g_eval_p

        self.operator.adjoint(self.y, out=self.x_tmp)
        self.x_tmp.multiply(-1.0, out=self.x_tmp)

        f_eval_d = self.f.convex_conjugate(self.y)
        g_eval_d = self.g.convex_conjugate(self.x_tmp)
        d1 = f_eval_d + g_eval_d

        self.loss.append([p1, -d1, p1+d1])

    @property
    def objective(self):
        return [x[0] for x in self.loss]

    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]

    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]
