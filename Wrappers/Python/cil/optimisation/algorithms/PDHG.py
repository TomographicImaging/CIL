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
    sigma : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (tau*norm(K)**2) if tau is provided
        Step size for the dual problem.
    tau : positive :obj:`float`, or `np.ndarray`, `DataContainer`, `BlockDataContainer`, optional, default is 1.0/norm(K) or 1.0/ (sigma*norm(K)**2) if sigma is provided
        Step size for the primal problem.
    initial : DataContainer, optional, default is a DataContainer of zeros
        Initial point for the PDHG algorithm.
    gamma_g : positive :obj:`float`, optional, default=None
        Strongly convex constant if the function g is strongly convex. Allows primal acceleration of the PDHG algorithm.
    gamma_fconj : positive :obj:`float`, optional, default=None
        Strongly convex constant if the convex conjugate of f is strongly convex. Allows dual acceleration of the PDHG algorithm.

    **kwargs:
        update_objective_interval : :obj:`int`, optional, default=1
            Evaluates objectives, e.g., primal/dual/primal-dual gap every ``update_objective_interval``.
        check_convergence : :obj:`boolean`, default=True
            Checks scalar sigma and tau values satisfy convergence criterion and warns if not satisfied. Can be computationally expensive for custom sigma or tau values. 
        theta :  Float between 0 and 1, default 1.0
            Relaxation parameter for the over-relaxation of the primal variable.


    Example
    -------

    In our `CIL-Demos <https://github.com/TomographicImaging/CIL-Demos/blob/main/binder/TomographyReconstruction.ipynb>`_ repository\
    you can find examples using the PDHG algorithm for different imaging problems, such as Total Variation denoising, Total Generalised Variation inpainting\
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

        \tau \sigma \|K\|^2 < 1


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


    - PDHG algorithm can be accelerated if the functions :math:`f^{*}` and/or :math:`g` are strongly convex. In these cases, the step-sizes :math:`\sigma` and :math:`\tau` are updated using the :meth:`update_step_sizes` method. A function :math:`f` is strongly convex with constant :math:`\gamma>0` if

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

    Note
    ----

    The case where both functions are strongly convex is not available at the moment.


    .. todo:: Implement acceleration of PDHG when both functions are strongly convex.


    """

    def __init__(self, f, g, operator, tau=None, sigma=None, initial=None, gamma_g=None, gamma_fconj=None, **kwargs):
        """Initialisation of the PDHG algorithm"""
        
        self._theta = kwargs.pop('theta', 1.0)
        if self.theta>1 or self.theta<0:
            raise ValueError("The relaxation parameter theta must be in the range [0,1], passed theta = {}".format(theta))  

        self._check_convergence = kwargs.pop('check_convergence', True)
        
        super().__init__(**kwargs)
        
        self._tau = None
        self._sigma = None

        # check for gamma_g, gamma_fconj, strongly convex constants
        self._gamma_g = None
        self._gamma_fconj = None
        self.set_gamma_g(gamma_g)
        self.set_gamma_fconj(gamma_fconj)

        self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, initial=initial)

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

    @property
    def gamma_g(self):
        """The strongly convex constant for the function g """
        return self._gamma_g

    @property
    def gamma_fconj(self):
        """The strongly convex constant for the convex conjugate of the function f """
        return self._gamma_fconj

    def set_gamma_g(self, value):
        '''Set the value of the strongly convex constant for function `g`

        Parameters
        ----------
            value : a positive number or None
        '''
        if self.gamma_fconj is not None and value is not None:
            raise ValueError("The adaptive update of the PDHG stepsizes in the case where both functions are strongly convex is not implemented at the moment." +\
                "Currently the strongly convex constant of the convex conjugate of the function f has been specified as ", self.gamma_fconj)

        if isinstance (value, Number):
            if value <= 0:
                raise ValueError("Strongly convex constant is a positive number, {} is passed for the strongly convex function g.".format(value))
            self._gamma_g = value
        elif value is None:
            pass
        else:
            raise ValueError("Positive float is expected for the strongly convex constant of function g, {} is passed".format(value))

    def set_gamma_fconj(self, value):
        '''Set the value of the strongly convex constant for the convex conjugate of function `f`

        Parameters
        ----------
            value : a positive number or None
        '''
        if self.gamma_g is not None and value is not None:
            raise ValueError("The adaptive update of the PDHG stepsizes in the case where both functions are strongly convex is not implemented at the moment." +\
                "Currently the strongly convex constant of the function g has been specified as ", self.gamma_g)

        if isinstance (value, Number):
            if value <= 0:
                raise ValueError("Strongly convex constant is positive, {} is passed for the strongly convex conjugate function of f.".format(value))
            self._gamma_fconj = value
        elif value is None:
            pass
        else:
            raise ValueError("Positive float is expected for the strongly convex constant of the convex conjugate of function f, {} is passed".format(value))

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
        initial : DataContainer, optional, default is a DataContainer of zeros
            Initial point for the PDHG algorithm.       """
        log.info("%s setting up", self.__class__.__name__)
        # Triplet (f, g, K)
        self.f = f
        self.g = g
        self.operator = operator

        self.set_step_sizes(sigma=sigma, tau=tau)

        if self._check_convergence:
            self.check_convergence()

        if initial is None:
            self.x_old = self.operator.domain_geometry().allocate(0)
        else:
            self.x_old = initial.copy()

        self.x = self.x_old.copy()
        self.x_tmp = self.operator.domain_geometry().allocate(0)
        self.y = self.operator.range_geometry().allocate(0)
        self.y_tmp = self.operator.range_geometry().allocate(0)


        if self.gamma_g is not None:
            warnings.warn("Primal Acceleration of PDHG: The function g is assumed to be strongly convex with positive parameter `gamma_g`. You need to be sure that gamma_g = {} is the correct strongly convex constant for g. ".format(self.gamma_g))

        if self.gamma_fconj is not None:
            warnings.warn("Dual Acceleration of PDHG: The convex conjugate of function f is assumed to be strongly convex with positive parameter `gamma_fconj`. You need to be sure that gamma_fconj = {} is the correct strongly convex constant".format(self.gamma_fconj))

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

    def update(self):
        """Performs a single iteration of the PDHG algorithm"""
        #calculate x-bar and store in self.x_tmp
        self.x_old.sapyb((self.theta + 1.0), self.x, -self.theta, out=self.x_tmp)

        # Gradient ascent for the dual variable
        self.operator.direct(self.x_tmp, out=self.y_tmp)

        self.y_tmp.sapyb(self.sigma, self.y, 1.0 , out=self.y_tmp)

        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)

        # Gradient descent for the primal variable
        self.operator.adjoint(self.y, out=self.x_tmp)

        self.x_tmp.sapyb(-self.tau, self.x_old, 1.0 , self.x_tmp)

        self.g.proximal(self.x_tmp, self.tau, out=self.x)

        # update_previous_solution() called after update by base class
        #i.e current solution is now in x_old, previous solution is now in x

        # update the step sizes for special cases
        self.update_step_sizes()

    def check_convergence(self):
        """Check whether convergence criterion for PDHG is satisfied with scalar values of tau and sigma

        Returns
        -------
        Boolean
            True if convergence criterion is satisfied. False if not satisfied or convergence is unknown.
        """
        if isinstance(self.tau, Number) and isinstance(self.sigma, Number):
            if self.sigma * self.tau * self.operator.norm()**2 > 1:
                warnings.warn("Convergence criterion of PDHG for scalar step-sizes is not satisfied.")
                return False
            return True
        warnings.warn("Convergence criterion can only be checked for scalar values of tau and sigma.")
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
                    raise ValueError("The step-sizes of PDHG must be positive, passed tau = {}".format(tau))
            elif tau.shape != self.operator.domain_geometry().shape:
                raise ValueError(" The shape of tau = {0} is not the same as the shape of the domain_geometry = {1}".format(tau.shape, self.operator.domain_geometry().shape))

        if sigma is not None:
            if isinstance(sigma, Number):
                if sigma <= 0:
                    raise ValueError("The step-sizes of PDHG are positive, passed sigma = {}".format(sigma))
            elif sigma.shape != self.operator.range_geometry().shape:
                raise ValueError(" The shape of sigma = {0} is not the same as the shape of the range_geometry = {1}".format(sigma.shape, self.operator.range_geometry().shape))

        # Default sigma and tau step-sizes
        if tau is None and sigma is None:
            self._sigma = 1.0/self.operator.norm()
            self._tau = 1.0/self.operator.norm()
        elif tau is not None and sigma is not None:
            self._sigma = sigma
            self._tau = tau
        elif sigma is None and isinstance(tau, Number):
            self._sigma = 1./(tau*self.operator.norm()**2)
            self._tau = tau
        elif tau is None and isinstance(sigma, Number):
            self._sigma = sigma
            self._tau = 1./(self.sigma*self.operator.norm()**2)
        else:
            raise NotImplementedError("If using arrays for sigma or tau both must arrays must be provided.")

    def update_step_sizes(self):
        """
        Updates step sizes in the cases of primal or dual acceleration using the strongly convexity property.
        The case where both functions are strongly convex is not available at the moment.
        """
        # Update sigma and tau based on the strong convexity of G
        if self.gamma_g is not None:
            self._theta = 1.0/ np.sqrt(1 + 2 * self.gamma_g * self.tau)
            self._tau *= self.theta
            self._sigma /= self.theta

        # Update sigma and tau based on the strong convexity of F
        # Following operations are reversed due to symmetry, sigma --> tau, tau -->sigma
        if self.gamma_fconj is not None:
            self._theta = 1.0 / np.sqrt(1 + 2 * self.gamma_fconj * self.sigma)
            self._sigma *= self.theta
            self._tau /= self.theta

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
