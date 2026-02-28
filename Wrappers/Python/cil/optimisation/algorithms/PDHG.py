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

from cil.framework import DataContainer, BlockDataContainer
from cil.optimisation.algorithms import Algorithm
import warnings
import numpy as np
from numbers import Number
import logging


log = logging.getLogger(__name__)


class PDHG(Algorithm):

    r"""Primal Dual Hybrid Gradient (PDHG) algorithm, see :cite:`CP2011`, :cite:`EZXC2010`.

    Parameters
    ----------
    f : Function
        A convex function with a "simple" proximal method of its conjugate.
    g : Function
        A convex function with a "simple" proximal.
    operator : LinearOperator
        A Linear Operator.
    sigma :  positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (tau*norm(K)**2) if tau is provided
        Step size for the dual problem.
    tau :  positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (sigma*norm(K)**2) if sigma is provided
        Step size for the primal problem.
    initial : `DataContainer`, or `list` or `tuple` of `DataContainer`s, optional, default is a DataContainer of zeros for both primal and dual variables
        Initial point for the PDHG algorithm. If just one data container is provided, it is used for the primal and the dual variable is initialised as zeros.  If a list or tuple is passed,  the first element is used for the primal variable and the second one for the dual variable. If either of the two is not provided, it is initialised as a DataContainer of zeros.
    update_step_sizes_rules : list of `StepSizeRule`s, optional, default is []

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

    Note
    ----

    Currently, the strongly convex constants are passed as parameters of PDHG.
    In the future, these parameters will be properties of the corresponding functions.


    Notes
    -----

    A first-order primal-dual algorithm for convex optimization problems with known saddle-point structure with applications in imaging.

    The general problem considered in the PDHG algorithm is the generic saddle-point problem

    .. math:: \min_{x\in X}\max_{y\in Y} \langle Kx, y \rangle + g(x) - f^{*}(x)

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

    - By default, the step sizes :math:`\sigma` and :math:`\tau` are positive scalars and defined as below:

      * If ``sigma`` is ``None`` and ``tau`` is ``None``:

      .. math::

        \sigma = \frac{1}{\|K\|},  \tau = \frac{1}{\|K\|}

      * If ``tau`` is ``None``:

      .. math::

        \tau = \frac{1}{\sigma\|K\|^{2}}

      * If ``sigma`` is ``None``:

      .. math::

        \sigma = \frac{1}{\tau\|K\|^{2}}


    - To monitor the convergence of the algorithm, we compute the primal/dual objectives and the primal-dual gap in :meth:`update_objective`.\

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

    def __init__(self, f, g, operator, tau=None, sigma=None, initial=None,  update_step_sizes_rules=[],
                 **kwargs):
        """Initialisation of the PDHG algorithm"""

        self._theta = kwargs.pop('theta', 1.0)
        if self._theta > 1 or self._theta < 0:
            raise ValueError(
                "The relaxation parameter theta must be in the range [0,1], passed theta = {}".format(self.theta))

        self._check_convergence = kwargs.pop('check_convergence', True)

        super().__init__(**kwargs)

        self._tau = None
        self._sigma = None

        self.update_step_sizes_rules = update_step_sizes_rules

        self.set_up(f=f, g=g, operator=operator, tau=tau,
                    sigma=sigma, initial=initial)

    @property
    def tau(self):
        """The primal step-size """
        return self._tau

    @property
    def sigma(self):
        """The dual step-size """
        return self._sigma

    @property
    def theta(self):
        """The relaxation parameter for the over-relaxation of the primal variable """
        return self._theta

    def set_up(self, f, g, operator, tau=None, sigma=None, initial=None):
        """Initialisation of the algorithm

        Parameters
        ----------
        f : Function
            A convex function with a "simple" proximal method of its conjugate.
        g : Function
            A convex function with a "simple" proximal.
        operator : LinearOperator
            A Linear Operator.
        sigma : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (tau*norm(K)**2) if tau is provided
            Step size for the dual problem.
        tau : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (sigma*norm(K)**2) if sigma is provided
            Step size for the primal problem.
        initial : `DataContainer`, or `list` or `tuple` of `DataContainer`s, optional, default is a DataContainer of zeros for both primal and dual variables
            Initial point for the PDHG algorithm. If just one data container is provided, it is used for the primal and the dual variable is initialised as zeros.  If a list or tuple is passed,  the first element is used for the primal variable and the second one for the dual variable. If either of the two is not provided, it is initialised as a DataContainer of zeros.

        """
        log.info("%s setting up", self.__class__.__name__)

        # Triplet (f, g, K)
        self.f = f
        self.g = g
        self.operator = operator

        self.set_step_sizes(sigma=sigma, tau=tau)
        print('Initial step sizes: tau = {}, sigma = {}'.format(
            self.tau, self.sigma))

        if self._check_convergence:
            self.check_convergence()

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

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def _update_previous_solution(self):
        """
        Swaps the references to current and previous solution based on the
        :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def get_output(self):
        " Returns the current solution. "
        return self.x_old

    def _pdhg_update(self):

        # calculate x-bar and store in self.x_tmp
        self.x_old.sapyb((self.theta + 1.0), self.x, -
                         self.theta, out=self.x_tmp)  # somewhere in line 4

        # Gradient ascent for the dual variable
        self.operator.direct(self.x_tmp, out=self.y_tmp)  # line 4

        self.y_tmp.sapyb(self.sigma, self.y, 1.0, out=self.y_tmp)  # line 4

        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)  # line 5

        # Gradient descent for the primal variable
        self.operator.adjoint(self.y, out=self.x_tmp)  # line 2

        self.x_tmp.sapyb(-self.tau, self.x_old, 1.0, self.x_tmp)  # line 2
        print('x_tmp norm = ', self.x_tmp.norm())

        self.g.proximal(self.x_tmp, self.tau, out=self.x)  # line 3
        print('x norm = ', (self.x - self.x_old).norm())
        # update_previous_solution() called after update by base class
        # i.e current solution is now in x_old, previous solution is now in x

    def update(self):
        """Performs a single iteration of the PDHG algorithm"""
        print('Iteration number: {}'.format(self.iteration))
        self._pdhg_update()

        # update the step sizes for special cases
        self.update_step_sizes()

    def check_convergence(self):
        """Check whether convergence criterion for PDHG is satisfied with scalar values of tau and sigma

        Returns
        -------
        Boolean
            True if convergence criterion is satisfied. False if not satisfied or convergence is unknown.

        Reference
        ----------
        Li, Y. and Yan, M., 2022. On the improved conditions for some primal-dual algorithms. arXiv preprint arXiv:2201.00139.

        """
        if isinstance(self.tau, Number) and isinstance(self.sigma, Number):
            if self.sigma * self.tau * self.operator.norm()**2 > 4/3:
                warnings.warn(
                    "Convergence criterion of PDHG for scalar step-sizes is not satisfied.")
                return False
            return True
        warnings.warn(
            "Convergence criterion can only be checked for scalar values of tau and sigma.")
        return False

    def set_step_sizes(self, sigma=None, tau=None):
        """Sets sigma and tau step-sizes for the PDHG algorithm. The step sizes can be either scalar or array-objects.

        Parameters
        ----------
            sigma : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default=None
                Step size for the dual problem.
            tau : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default=None
                Step size for the primal problem.

        The user can set either, both or none. Values passed by the user will be accepted as long as they are positive numbers,
        or correct shape array like objects.
        """
        # Check acceptable values of the primal-dual step-sizes
        if tau is not None:
            if isinstance(tau, Number):
                if tau <= 0:
                    raise ValueError(
                        "The step-sizes of PDHG must be positive, passed tau = {}".format(tau))
            elif tau.shape != self.operator.domain_geometry().shape:
                raise ValueError(" The shape of tau = {0} is not the same as the shape of the domain_geometry = {1}".format(
                    tau.shape, self.operator.domain_geometry().shape))

        if sigma is not None:
            if isinstance(sigma, Number):
                if sigma <= 0:
                    raise ValueError(
                        "The step-sizes of PDHG are positive, passed sigma = {}".format(sigma))
            elif sigma.shape != self.operator.range_geometry().shape:
                raise ValueError(" The shape of sigma = {0} is not the same as the shape of the range_geometry = {1}".format(
                    sigma.shape, self.operator.range_geometry().shape))

        # Default sigma and tau step-sizes
        if tau is None and sigma is None:
            if self.adaptive:
                self._sigma = 1e5
                self._tau = 1e5
            else:
                self._sigma = 1.0/self.operator.norm()
                self._tau = 1.0/self.operator.norm()
        elif tau is not None and sigma is not None:
            self._sigma = sigma
            self._tau = tau
        elif sigma is None and isinstance(tau, Number):
            if self.adaptive:
                self._sigma = 1e5
            else:
                self._sigma = 1./(tau*self.operator.norm()**2)
            self._tau = tau
        elif tau is None and isinstance(sigma, Number):
            self._sigma = sigma
            if self.adaptive:
                self._tau = 1e5
            else:
                self._tau = 1./(self.sigma*self.operator.norm()**2)
        else:
            raise NotImplementedError(
                "If using arrays for sigma or tau both must arrays must be provided.")

    def update_step_sizes(self):
        """
        Updates the step sizes, based on the rules passed
        """

        for rule in self.update_step_sizes_rules:
            self._theta, self._tau, self._sigma = rule.get_step_size(self)

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
