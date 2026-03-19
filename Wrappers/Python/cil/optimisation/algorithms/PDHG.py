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
from cil.optimisation.utilities import StepSizeRule, PDHGStronglyConvexUpdate, PDHGConstantStepSize
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
    step_size:
        Either a PDHG compatible step size rule or a `list` or `tuple`  of (tau, sigma) where sigma is the step size for the dual problem and tau is the step size for the primal problem. The step sizes can be either None,  scalar or array-objects. If not provided, default values will be set based on the operator norm as described below.
    initial : `DataContainer`, or `list` or `tuple` of `DataContainer`s, optional, default is a DataContainer of zeros for both primal and dual variables
        Initial point for the PDHG algorithm. If just one data container is provided, it is used for the primal and the dual variable is initialised as zeros.  If a list or tuple is passed,  the first element is used for the primal variable and the second one for the dual variable. If either of the two is not provided, it is initialised as a DataContainer of zeros.

    **kwargs:
        objective_interval : :obj:`int`, optional, default=1
            Evaluates objectives, e.g., primal/dual/primal-dual gap every ``objective_interval``.
        check_convergence : :obj:`boolean`, default=True
            Checks scalar sigma and tau values satisfy convergence criterion and warns if not satisfied. Can be computationally expensive for custom sigma or tau values. 
        theta :  Float between 0 and 1, default 1.0
            Relaxation parameter for the over-relaxation of the primal variable.
        gamma_g : positive :obj:`float`, optional, default=None
            Note: this is being deprecated. Strongly convex constant if the function g is strongly convex. Allows primal acceleration of the PDHG algorithm.
        gamma_fconj : positive :obj:`float`, optional, default=None
            Note: this is being deprecated. Strongly convex constant if the convex conjugate of f is strongly convex. Allows dual acceleration of the PDHG algorithm.
        sigma :  positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (tau*norm(K)**2) if tau is provided
           Step size for the dual problem. Note: this is being deprecated. In the future, please pass this as part of the `step_size` argument, either as a tuple of (sigma, tau) or using a compatible step size rule.
        tau :  positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (sigma*norm(K)**2) if sigma is provided
            Step size for the primal problem. In the future, please pass this as part of the `step_size` argument, either as a tuple of (sigma, tau) or using a compatible step size rule.
    Example
    -------

    In our CIL-Demos repository (https://github.com/TomographicImaging/CIL-Demos) you can find examples using the PDHG algorithm for different imaging problems, such as Total Variation denoising, Total Generalised Variation inpainting 
    and Total Variation Tomography reconstruction. More examples can also be found in :cite:`Jorgensen_et_al_2021`, :cite:`Papoutsellis_et_al_2021`.

   
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


    - To monitor the convergence of the algorithm, we compute the primal/dual objectives and the primal-dual gap in :meth:`objective`.\

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

        Computing these objectives can be costly, so it is better to compute every some iterations. To do this, use ``objective_interval = #number``.



    """

    def __init__(self, f, g, operator, step_size=None,  initial=None,
                 **kwargs):
        """Initialisation of the PDHG algorithm"""


        self._sigma = kwargs.pop('sigma', None) #To be deprecated
        self._tau = kwargs.pop('tau', None) #To be deprecated
        self._theta = kwargs.pop('theta', 1.0)
        if self._theta > 1 or self._theta < 0:
            raise ValueError(
                "The relaxation parameter theta must be in the range [0,1], passed theta = {}".format(self.theta))
            
        if step_size is not None: #To be deprecated
            if self._sigma is not None or self._tau is not None: #To be deprecated
                raise ValueError( "The parameters `sigma` and `tau` are being deprecated in favour of `step_size`. You have passed both. Instead please pass these as part of the `step_size` argument, either as a tuple of (sigma, tau) or using a compatible step size rule.", DeprecationWarning)
        
        if self._sigma is not None or self._tau is not None: #To be deprecated
            warnings.warn("The parameters `sigma` and `tau` are being deprecated. In the future, please pass these as part of the `step_size` argument, either as a tuple of (sigma, tau) or using a compatible step size rule.", DeprecationWarning)
            step_size = (self._tau, self._sigma)


        self._gamma_g = kwargs.pop('gamma_g', None) #To be deprecated
        self._gamma_fconj = kwargs.pop('gamma_fconj', None) #To be deprecated
        if self._gamma_g is not None or self._gamma_fconj is not None: #To be deprecated
            warnings.warn("The parameter `gamma_g` is being deprecated. In the future, if you would like to utilise strong convexity you should use the step size method cil.optimisation.utilities.StepSizeMethods.PDHGStronglyConvexUpdate.", DeprecationWarning)
            step_size = PDHGStronglyConvexUpdate( initial_step_size =(self._tau, self._sigma), gamma_g=self._gamma_g, gamma_fconj=self._gamma_fconj)
        
        
        
        self._check_convergence = kwargs.pop('check_convergence', True)


        super().__init__(**kwargs)

               
        self.set_up(f=f, g=g, operator=operator, step_size= step_size,  initial=initial)

    @property
    def tau(self):
        """The primal step-size - Returns the currently being used step size for the primal problem. Note that this can be updated at each iteration if a step size rule is used."""
        return self._tau

    @property
    def sigma(self):
        """The dual step-size  - Returns the currently being used step size for the dual problem. Note that this can be updated at each iteration if a step size rule is used."""
        return self._sigma

    @property
    def theta(self):
        """The relaxation parameter for the over-relaxation of the primal variable """
        return self._theta

    def set_up(self, f, g, operator, step_size=[None, None],  initial=None):
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
        step_size:
            Either a PDHG compatible step size rule or a `list` or `tuple`  of (tau, sigma) where sigma is the step size for the dual problem and tau is the step size for the primal problem. The step sizes can be either scalar or array-objects. If not provided, default values will be set based on the operator norm as described below.
    

        """
    
        log.info("%s setting up", self.__class__.__name__)

        # Triplet (f, g, K)
        self.f = f
        self.g = g
        self.operator = operator

        if step_size is None: #This line can be removed when sigma and tau deprecated
            step_size = (None, None) 
        if isinstance(step_size, StepSizeRule):
            self.step_size_rule = step_size
        elif isinstance(step_size, (tuple, list)):
            self.step_size_rule = PDHGConstantStepSize(step_size=step_size)
        else:
            raise ValueError("The `step_size` argument must be either None, a PDHG compatible step size rule or a tuple of (sigma, tau) where sigma is the step size for the dual problem and tau is the step size for the primal problem.")


        self._tau, self._sigma = self.step_size_rule.get_initial_step_size(self)
        
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
        :func:`~Algorithm.previous_solution` of the base class :class:`Algorithm`.
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
        # previous_solution() called after update by base class
        # i.e current solution is now in x_old, previous solution is now in x

    def update(self):
        """Performs a single iteration of the PDHG algorithm"""
        print('Iteration number: {}'.format(self.iteration))
        self._pdhg_update()

        # update the step sizes for special cases
        self._tau, self._sigma = self.step_size_rule.get_step_size(self)

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


        self.loss.append([p1, -d1, p1+d1])


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
