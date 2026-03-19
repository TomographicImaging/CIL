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
import numpy as np

log = logging.getLogger(__name__)


class StepSizeRule(ABC):
    """
    Abstract base class for a step size rule. The abstract method, `get_step_size` takes in an algorithm and thus can access all parts of the algorithm (e.g. current iterate, current gradient, objective functions etc) and from this  should return a float as a step size. 
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
        if self.alpha_orig is None:  # Can be removed when alpha and beta are deprecated in GD
            self.alpha_orig = 1e6
        self.alpha = self.alpha_orig
        self.beta = beta
        if self.beta is None:  # Can be removed when alpha and beta are deprecated in GD
            self.beta = 0.5

        self.max_iterations = max_iterations
        if self.max_iterations is None:
            self.max_iterations = numpy.ceil(
                2 * numpy.log10(self.alpha_orig) / numpy.log10(2))

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
            if f_x_a - f_x <= - (self.alpha/2.) * sqnorm:
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
            stabilisation_param = numpy.inf
        elif stabilisation_param == "off":
            self.adaptive = False
            stabilisation_param = numpy.inf
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
        ret = numpy.nanmin(numpy.array(
            [ret, self.stabilisation_param/gradient_norm]))

        # We store the last iterate and gradient in order to calculate the BB step size
        self.store_x.fill(algorithm.x)
        self.store_grad.fill(algorithm.gradient_update)

        if self.mode == "alternate":
            self.is_short = not self.is_short

        return ret


class PDHGStronglyConvexUpdate(StepSizeRule):
    '''Updates step sizes (theta, sigma, tau) in the PDHG algorithm in the cases of primal or dual acceleration using the strongly convexity property.
            The case where both functions are strongly convex is not available at the moment.


            The PDHG algorithm can be accelerated if the functions :math:`f^{*}` and/or :math:`g` are strongly convex. In these cases, the step-sizes :math:`\sigma` and :math:`\tau` are updated using the :meth:`update_step_sizes` method. A function :math:`f` is strongly convex with constant :math:`\gamma>0` if

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
            gamma_g : positive :obj:`float`, optional, default=None
                Strongly convex constant if the function g is strongly convex. Allows primal acceleration of the PDHG algorithm.
            gamma_fconj : positive :obj:`float`, optional, default=None
                Strongly convex constant if the convex conjugate of f is strongly convex. Allows dual acceleration of the PDHG algorithm.
            '''

    def __init__(self, initial_step_size =(None, None), gamma_g=None, gamma_fconj=None):#TODO: tuple of list 
        '''Initialises the step size rule'''
        
        self.gamma_g = gamma_g
        self.gamma_fconj = gamma_fconj
        if self.gamma_g is not None and self.gamma_fconj is not None:
            raise NotImplementedError(
                "PDHG strongly convex step size update not implemented for both primal and dual acceleration. Please choose only one of gamma_g or gamma_fconj.")
        if isinstance(gamma_g, Number):
            if gamma_g <= 0:
                raise ValueError(
                    "Strongly convex constant is positive, {} is passed for the strongly convex conjugate function of f.".format(gamma_g))
            self.gamma_g = gamma_g
        elif gamma_g is None:
            pass
        else:
            raise ValueError(
                "Positive float is expected for the strongly convex constant of the function g, {} is passed".format(gamma_g))
            pass

        if isinstance(gamma_fconj, Number):
            if gamma_fconj <= 0:
                raise ValueError(
                    "Strongly convex constant is positive, {} is passed for the strongly convex conjugate function of f.".format(gamma_fconj))
            self.gamma_fconj = gamma_fconj
        elif gamma_fconj is None:
            pass
        else:
            raise ValueError(
                "Positive float is expected for the strongly convex constant of the convex conjugate of function f, {} is passed".format(gamma_fconj))

        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, step_size = {}".format(step_size))
    
        
    def get_initial_step_size(self, algorithm):
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
        self.tau = self.initial_step_size[0]
        self.sigma = self.initial_step_size[1]
        
        # Check acceptable values of the primal-dual step-sizes
        if self.tau is not None:
            if isinstance(self.tau, Number):
                if self.tau <= 0:
                    raise ValueError(
                        "The step-sizes of PDHG must be positive, passed tau = {}".format(self.tau))
            elif self.tau.shape != algorithm.operator.domain_geometry().shape:
                raise ValueError(" The shape of tau = {0} is not the same as the shape of the domain_geometry = {1}".format(
                    self.tau.shape, algorithm.operator.domain_geometry().shape))

        if self.sigma is not None:
            if isinstance(self.sigma, Number):
                if self.sigma <= 0:
                    raise ValueError(
                        "The step-sizes of PDHG are positive, passed sigma = {}".format(self.sigma))
            elif self.sigma.shape != algorithm.operator.range_geometry().shape:
                raise ValueError(" The shape of sigma = {0} is not the same as the shape of the range_geometry = {1}".format(
                    self.sigma.shape, algorithm.operator.range_geometry().shape))

        # Default sigma and tau step-sizes
        if self.tau is None and self.sigma is None:
            self.sigma = 1.0/algorithm.operator.norm()
            self.tau = 1.0/algorithm.operator.norm()
        elif self.tau is not None and self.sigma is not None:
            pass
        elif self.sigma is None and isinstance(self.tau, Number):
            self.sigma = 1./(self.tau*algorithm.operator.norm()**2)
        elif self.tau is None and isinstance(self.sigma, Number):
            self.tau = 1./(self.sigma*algorithm.operator.norm()**2)
        else:
            raise NotImplementedError(
                "If using arrays for sigma or tau both must arrays must be provided.")
        return self.tau, self.sigma
    
    
    def get_step_size(self, algorithm):
        """
        Applies the PDHG strongly convex step size update to calculate the new primal and dual step sizes

        Returns
        --------

        """
        # Update sigma and tau based on the strong convexity of G
        if self.gamma_g is not None:
            algorithm._theta = 1.0 / np.sqrt(1 + 2 * self.gamma_g * algorithm.tau)
            self.tau *= algorithm._theta
            self.sigma /= algorithm._theta

        # Update sigma and tau based on the strong convexity of F
        # Following operations are reversed due to symmetry, sigma --> tau, tau -->sigma
        if self.gamma_fconj is not None:
            algorithm._theta = 1.0 / np.sqrt(1 + 2 * self.gamma_fconj * algorithm.sigma)
            self.sigma *= algorithm._theta
            self.tau /= algorithm._theta

        return self.tau, self.sigma 


class PDHGAdaptiveStepSize2013(StepSizeRule):
    ''''The PDHG step sizes are updated adaptively based on the method proposed in :cite:`goldstein2013adaptive`.
        Parameters
        -------------
        initial_step_size : list of two positive :obj:`float`, optional, default=[10e5, 10e5]
            Initial values of the primal and dual step sizes used in the adaptive step size method.
        initial_alpha : positive :obj:`float`, optional, default=0.95
            Initial value of the parameter alpha used in the adaptive step size method.
        beta : positive :obj:`float`, optional, default=0.95
            Value of the parameter eta used in the adaptive step size method.
        gamma : positive :obj:`float`, optional, default=0.75
            Value of the parameter c used in the adaptive step size method.
        delta : positive :obj:`float`,greater than one,  optional, default=1.5
            Value of the parameter delta used in the adaptive step size method.
        s : positive :obj:`float`, optional, default= Norm of the operator A 
            Value of the parameter s used in the adaptive step size method.
        eta : positive :obj:`float`, optional, default=0.95
            Value of the parameter eta used in the adaptive step size method.
        auto_stop : :obj:`boolean`, optional, default=True
            If True, the adaptive step size method automatically stops updating the step sizes when they have not changed over five consecutive iterations.

        '''

    def __init__(self, initial_step_size=[10e5, 10e5], initial_alpha=0.95, beta=0.95, gamma=0.9, delta=1.5, s=None, eta=0.95, auto_stop=True):
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

        self.auto_stop = auto_stop
        self.count = 0

        self.y_old = None
        self.x_resid = None
        self.y_resid = None
        self.x_store = None

        self.adaptive = True
        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, step_size = {}".format(initial_step_size))
 
    def get_initial_step_size(self, algorithm):
        tau = self.initial_step_size[0]
        sigma = self.initial_step_size[1]
        if tau is None:
            tau = 1e5
        if sigma is None:
            sigma = 1e5
        return tau, sigma

    def get_step_size(self, algorithm):
        if self.adaptive:
            if self.s is None:
                self.s = algorithm.operator.norm()  # Is this the right initial?
            if self.y_old is None:
                self.y_old = algorithm.operator.range_geometry().allocate(0)  # Extra range data 1
                self.x_resid = algorithm.operator.domain_geometry().allocate(0)  # Extra image 1
                self.y_resid = algorithm.operator.range_geometry().allocate(0)  # Extra range data 2
                self.x_store = algorithm.operator.domain_geometry().allocate(0)  # Extra image 2
            # adaptive step sizes only when above tolerance
            if self.p_norm > self.tolerance and self.d_norm > self.tolerance:
                # print('Before adaptive', self.tau, self.sigma)
                b = self._calculate_backtracking(algorithm)
                while b > 1:

                    print(
                        'Multiplying step sizes by beta/b, beta = {}, b = {}'.format(self.beta, b))
                    algorithm._tau *= self.beta/b
                    algorithm._sigma *= self.beta/b

                    # Swap x and x_store
                    tmp = algorithm.x
                    algorithm.x = self.x_store
                    self.x_store = tmp

                    print(
                        'Multiplying step sizes by beta/b, beta = {}, b = {}'.format(self.beta, b))
                    print('tau = {}, sigma = {}'.format(
                        algorithm._tau, algorithm._sigma))
                    algorithm._pdhg_update()
                    b = self._calculate_backtracking(algorithm)

                print('After possible reduction',
                      algorithm._tau, algorithm._sigma)
                
                self._calculate_pnorm_dnorm(algorithm)
                print('p_norm = {}, d_norm = {}'.format(
                    self.p_norm, self.d_norm))
                print('self.s, self.delta = {}, {}'.format(self.s, self.delta))
                if self.p_norm < (self.s/self.delta)*self.d_norm:
                    print('2*self.p_norm < self.d_norm')
                    algorithm._tau *= (1 - self.alpha)
                    algorithm._sigma /= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                elif (self.s*self.delta)*self.d_norm < self.p_norm:
                    print('2*self.d_norm < self.p_norm')
                    algorithm._tau /= (1 - self.alpha)
                    algorithm._sigma *= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                else:
                    print('No change')
                    self.count += 1
                    pass
                print('After adaptive', algorithm._tau,
                      algorithm._sigma, self.alpha)
            else:
                print('No adaptive step size update, below tolerance')
            # Can i do something other than copying every iteration?
            self.y_old = algorithm.y.copy()

            if self.count > 10 and self.auto_stop:
                self.adaptive = False
                print(
                    'Automatic stopping of adaptive step size updates, step sizes have not changed for 5 iterations')
                del self.x_resid
                del self.y_resid
                del self.x_store
                del self.y_old

        return  algorithm._tau, algorithm._sigma

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
        print('self.x, self.x_old = ', algorithm.x.norm(), algorithm.x_old.norm())
        x_change_norm = self.x_resid.norm()
        algorithm.y.sapyb(1.0, self.y_old, -1.0, out=self.y_resid)
        y_change_norm = self.y_resid.norm()
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        cross_term = np.real(2*algorithm._sigma *
                             algorithm._tau*self.y_resid.dot(algorithm.y_tmp))
        print('cross_term = ', cross_term, 'x_change_norm = ',
              x_change_norm, 'y_change_norm = ', y_change_norm)
        b = cross_term/((self.gamma*algorithm._sigma)*x_change_norm **
                        2 + (self.gamma*algorithm._tau)*y_change_norm**2)
        print(b)
        return b


class PDHGAdaptiveStepSize2015(StepSizeRule):
    '''The PDHG step sizes are updated adaptively based on the method proposed in :cite:`Goldstein2015`.

        Parameters
        -------------
        initial_step_size : list of two positive :obj:`float`, optional, default=[10e5, 10e5]
            Initial values of the primal and dual step sizes used in the adaptive step size method.
        initial_alpha : positive :obj:`float`, optional, default=0.95
        Initial value of the parameter alpha used in the adaptive step size method.
        eta : positive :obj:`float`, optional, default=0.95
            Value of the parameter eta used in the adaptive step size method.
        c : positive :obj:`float`, optional, default=0.9
            Value of the parameter c used in the adaptive step size method.

        '''

    def __init__(self, initial_step_size=[10e5, 10e5],  initial_alpha=0.95, eta=0.95, c=0.9, auto_stop=True):
        '''Initialises the step size rule'''

        self.adaptive = True
        self.alpha = initial_alpha
        self.eta = eta
        self.c = c
        self.tolerance = 1e-6
        self.p_norm = 100
        self.d_norm = 100
        self.auto_stop = auto_stop
        self.count = 0

        self.y_old = None
        self.x_resid = None
        self.y_resid = None
        self.x_store = None
        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, step_size = {}".format(initial_step_size))

        
    def get_initial_step_size(self, algorithm): #TODO: this needs some proper testing 
        tau = self.initial_step_size[0]
        sigma = self.initial_step_size[1]
        if tau is None:
            tau = 1e5
        if sigma is None:
            sigma = 1e5
        return tau, sigma

    def get_step_size(self, algorithm):
        if self.adaptive:
            if self.y_old is None:
                self.y_old = algorithm.operator.range_geometry().allocate(0)  # Extra range data 1
                self.x_resid = algorithm.operator.domain_geometry().allocate(0)  # Extra image 1
                self.y_resid = algorithm.operator.range_geometry().allocate(0)  # Extra range data 2
                self.x_store = algorithm.operator.domain_geometry().allocate(0)  # Extra image 2
            # adaptive step sizes only when above tolerance
            if self.p_norm > self.tolerance and self.d_norm > self.tolerance:
                # print('Before adaptive', self.tau, self.sigma)
                b = self._calculate_backtracking(algorithm)
                while b > 1:

                    print(
                        'Multiplying step sizes by beta/b, beta = {}, b = {}'.format(self.beta, b))
                    algorithm._tau *= self.beta/b
                    algorithm._sigma *= self.beta/b

                    # Swap x and x_store
                    tmp = algorithm.x
                    algorithm.x = self.x_store
                    self.x_store = tmp

                    print(
                        'Multiplying step sizes by beta/b, beta = {}, b = {}'.format(self.beta, b))
                    print('tau = {}, sigma = {}'.format(
                        algorithm._tau, algorithm._sigma))
                    algorithm._pdhg_update()
                    b = self._calculate_backtracking(algorithm)

                print('After possible reduction',
                      algorithm._tau, algorithm._sigma)
                algorithm.operator.adjoint(self.y_resid, out=algorithm.x_tmp)
                algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
                self.x_resid.sapyb((1/algorithm._tau),
                                   algorithm.x_tmp, -1.0, out=algorithm.x_tmp)
                self.y_resid.sapyb((1/algorithm._sigma),
                                   algorithm.y_tmp, -1.0, out=algorithm.y_tmp)
                self.p_norm = algorithm.x_tmp.norm()
                self.d_norm = algorithm.operator.norm()*algorithm.y_tmp.norm()
                print('p_norm = {}, d_norm = {}'.format(
                    self.p_norm, self.d_norm))
                if 2*self.p_norm < self.d_norm:
                    print('2*self.p_norm < self.d_norm')
                    algorithm._tau *= (1 - self.alpha)
                    algorithm._sigma /= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                elif 2*self.d_norm < self.p_norm:
                    print('2*self.d_norm < self.p_norm')
                    algorithm._tau /= (1 - self.alpha)
                    algorithm._sigma *= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                else:
                    print('No change')
                    self.count += 1
                    pass
                print('After adaptive', algorithm._tau,
                      algorithm._sigma, self.alpha)
            else:
                print('No adaptive step size update, below tolerance')
            # Can i do something other than copying every iteration?
            self.y_old = algorithm.y.copy()

            if self.count > 10 and self.auto_stop:
                self.adaptive = False
                print(
                    'Automatic stopping of adaptive step size updates, step sizes have not changed for 5 iterations')
                del self.x_resid
                del self.y_resid
                del self.x_store
                del self.y_old

        return  algorithm._tau, algorithm._sigma

    def _calculate_backtracking(self, algorithm):
        """ Calculates the backtracking parameter b used to update step sizes in the adaptive PDHG algorithm.
            Returns
            -------
            b : :obj:`float`
                Backtracking parameter used to update step sizes in the adaptive PDHG algorithm.
        """

        algorithm.x.sapyb(1.0, algorithm.x_old, -1.0, out=self.x_resid)
        print('self.x, self.x_old = ', algorithm.x.norm(), algorithm.x_old.norm())
        x_change_norm = self.x_resid.norm()
        algorithm.y.sapyb(1.0, self.y_old, -1.0, out=self.y_resid)
        y_change_norm = self.y_resid.norm()
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        cross_term = np.abs(4*algorithm._sigma*algorithm._tau *
                            self.y_resid.dot(algorithm.y_tmp))
        print('cross_term = ', cross_term, 'x_change_norm = ',
              x_change_norm, 'y_change_norm = ', y_change_norm)
        b = self.c*algorithm._sigma*x_change_norm**2 + \
            self.c*algorithm._tau*y_change_norm**2 - cross_term
        print(b)
        return b

class PDHGConstantStepSize(StepSizeRule):
    """
    Step-size rule that always returns a constant step-size.
     
    The user can set either the primal or dual step size, both or none. Values passed by the user will be accepted as long as they are positive numbers,
        or correct shape array like objects.
        
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

    def __init__(self,  step_size=[None,None]):
        '''Initialises the constant step size rule'''

        if len(step_size) != 2:
            raise ValueError(
                "step_size should be a list or tuple of length two, step_size = {}".format(step_size))
        self.tau = step_size[0]
        self.sigma = step_size[1]

    def get_initial_step_size(self, algorithm):
        """Sets sigma and tau step-sizes for the PDHG algorithm."""
        
        # Check acceptable values of the primal-dual step-sizes
        if self.tau is not None:
            if isinstance(self.tau, Number):
                if self.tau <= 0:
                    raise ValueError(
                        "The step-sizes of PDHG must be positive, passed tau = {}".format(self.tau))
            elif self.tau.shape != algorithm.operator.domain_geometry().shape:
                raise ValueError(" The shape of tau = {0} is not the same as the shape of the domain_geometry = {1}".format(
                    self.tau.shape, algorithm.operator.domain_geometry().shape))

        if self.sigma is not None:
            if isinstance(self.sigma, Number):
                if self.sigma <= 0:
                    raise ValueError(
                        "The step-sizes of PDHG are positive, passed sigma = {}".format(self.sigma))
            elif self.sigma.shape != algorithm.operator.range_geometry().shape:
                raise ValueError(" The shape of sigma = {0} is not the same as the shape of the range_geometry = {1}".format(
                    self.sigma.shape, algorithm.operator.range_geometry().shape))

        # Default sigma and tau step-sizes
        if self.tau is None and self.sigma is None:
            self.sigma = 1.0/algorithm.operator.norm()
            self.tau = 1.0/algorithm.operator.norm()
        elif self.tau is not None and self.sigma is not None:
            pass
        elif self.sigma is None and isinstance(self.tau, Number):
            self.sigma = 1./(self.tau*algorithm.operator.norm()**2)
        elif self.tau is None and isinstance(self.sigma, Number):
            self.tau = 1./(self.sigma*algorithm.operator.norm()**2)
        else:
            raise NotImplementedError(
                "If using arrays for sigma or tau both must arrays must be provided.")
        return self.tau, self.sigma
            
    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the calculated step size:float
        """
        return  self.tau, self.sigma