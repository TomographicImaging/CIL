#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
# Claire Delplancke (University of Bath)

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators import BlockOperator
import numpy as np
import logging
from cil.optimisation.utilities import Sampler
from numbers import Number
import warnings
from cil.framework import BlockDataContainer

log = logging.getLogger(__name__)


class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient (SPDHG) solves separable optimisation problems of the type: 
    .. math::

      \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)

    where :math:`f_i` and the regulariser :math:`g` need to be proper, convex and lower semi-continuous.

    Parameters
    ----------
    f : BlockFunction
        Each must be a convex function with a "simple" proximal method of its conjugate
    g : Function
        A convex function with a "simple" proximal
    operator : BlockOperator
        BlockOperator must contain Linear Operators
    tau : positive float, optional, default=None
        Step size parameter for primal problem. If `None` see note.
    sigma : list of positive float, optional, default=None
        List of Step size parameters for dual problem. If `None` see note.
    initial : DataContainer, optional, the default value is a zero DataContainer in the range of the `operator`.
        Initial point for the SPDHG algorithm
    gamma : float, optional
            Parameter controlling the trade-off between the primal and dual step sizes
    sampler: optional, an instance of a `cil.optimisation.utilities.Sampler` class or another class with the function __next__(self) implemented outputting an integer from {1,...,len(operator)}. 
            Method of selecting the next index for the SPDHG update. If None, a sampler will be created for random sampling with replacement and each index will have `probability = 1/len(operator)`
    prob_weights: optional, list of floats of length `num_indices` that sum to 1. Defaults to `[1/len(operator)]*len(operator)`
            Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. Note that this should not be passed if the provided sampler has it as an attribute: if the sampler has a `prob_weight` attribute it will take precedence on this parameter. 



    Example
    -------
    >>> data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
    >>> data.reorder('astra')
    >>> data = data.get_slice(vertical='centre')
    >>> ig = ag.get_ImageGeometry()

    >>> data_partitioned = data.partition(num_batches=10, mode='staggered')
    >>> A_partitioned = ProjectionOperator(ig, data_partitioned.geometry, device = "cpu")


    >>> F = BlockFunction(*[L2NormSquared(b=data_partitioned[i])
                            for i in range(10)])

    >>> alpha = 0.025
    >>> G = alpha * TotalVariation()
    >>> spdhg = SPDHG(f=F, g=G, operator=A_partitioned, sampler=Sampler.sequential(len(A)),
                      initial=A.domain_geometry().allocate(1), update_objective_interval=10)
    >>> spdhg.run(100)


    Example
    -------
    Further examples of usage see the [CIL demos.](https://github.com/vais-ral/CIL-Demos/blob/master/Tomography/Simulated/Single%20Channel/PDHG_vs_SPDHG.py)

    Note
    -----
    When setting `sigma` and `tau`, there are 4 possible cases considered by setup function. In all cases the probabilities :math:`p_i` are set by a default or user defined sampler:

    - Case 1: If neither `sigma` or `tau` are provided then `sigma` is set using the formula:

    .. math:: \sigma_i= \frac{0.99}{\|K_i\|^2}

    and `tau` is set as per case 2

    - Case 2: If `sigma` is provided but not `tau` then `tau` is calculated using the formula 

    .. math:: \tau = 0.99\min_i( \frac{p_i}{ (\sigma_i  \|K_i\|^2) })

    - Case 3: If `tau` is provided but not `sigma` then `sigma` is calculated using the formula

    .. math:: \sigma_i= \frac{0.99 p_i}{\tau\|K_i\|^2}

    - Case 4: Both `sigma` and `tau` are provided.


    Note
    ----

    Convergence is guaranteed provided that [2, eq. (12)]:

    .. math:: \|\sigma[i]^{1/2}  K[i]  \tau^{1/2} \|^2  < p_i \text{ for all } i

    References
    ----------

    [1]"Stochastic primal-dual hybrid gradient algorithm with arbitrary 
    sampling and imaging applications",
    Chambolle, Antonin, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schonlieb,
    SIAM Journal on Optimization 28, no. 4 (2018): 2783-2808.   https://doi.org/10.1137/17M1134834 

    [2]"Faster PET reconstruction with non-smooth priors by randomization and preconditioning",
    Matthias J Ehrhardt, Pawel Markiewicz and Carola-Bibiane Schönlieb,
    Physics in Medicine & Biology, Volume 64, Number 22, 2019. https://doi.org/10.1088/1361-6560/ab3d07
    '''

    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=None,
                 initial=None, sampler=None, prob_weights=None,   **kwargs):

        update_objective_interval = kwargs.pop('update_objective_interval', 1)
        super(SPDHG, self).__init__(
            update_objective_interval=update_objective_interval)

        self.set_up(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                    initial=initial,  sampler=sampler, prob_weights=prob_weights,  **kwargs)

    def set_up(self, f, g, operator, sigma=None, tau=None,
               initial=None,   sampler=None, prob_weights=None, **deprecated_kwargs):
        '''set-up of the algorithm
        '''
        log.info("%s setting up", self.__class__.__name__)

        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator

        if not isinstance(operator, BlockOperator):
            raise TypeError("operator should be a BlockOperator")

        self._ndual_subsets = len(self.operator)
        

        # Set up sampler and prob weights from deprecated "prob" argument
        self._deprecated_set_prob(deprecated_kwargs, prob_weights, sampler) 
        
        
        self._prob_weights = getattr(sampler, 'prob_weights', prob_weights) 
        if self._prob_weights is None: 
            self._prob_weights = [1/self._ndual_subsets]*self._ndual_subsets
        
        if  prob_weights is not None and self._prob_weights != prob_weights:
                    raise ValueError(' You passed a `prob_weights` argument and a sampler with a different attribute `prob_weights`, please remove the `prob_weights` argument.')

        if sampler is None:
            self._sampler = Sampler.random_with_replacement(
                len(operator), prob=self._prob_weights)
        else:
            self._sampler = sampler

        #Set the norms of the operators
        self._deprecated_set_norms(deprecated_kwargs) 
        self._norms = operator.get_norms_as_list()
        #Check for other kwargs
        self._deprecated_else(deprecated_kwargs)

        self.set_step_sizes(sigma=sigma, tau=tau)

        # initialize primal variable
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self._x_tmp = self.operator.domain_geometry().allocate(0)

        # initialize dual variable to 0
        self._y_old = operator.range_geometry().allocate(0)
        if not isinstance(self._y_old, BlockDataContainer): #This can be removed once #1863 is fixed
            self._y_old =BlockDataContainer(self._y_old)

        # initialize variable z corresponding to back-projected dual variable
        self._z = operator.domain_geometry().allocate(0)
        self._zbar = operator.domain_geometry().allocate(0)
        # relaxation parameter
        self._theta = 1

        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))

    def _deprecated_set_prob(self, deprecated_kwargs, prob_weights, sampler):
        """
        Handle deprecated keyword arguments for backward compatibility.

        Parameters
        ----------
        deprecated_kwargs : dict
            Dictionary of keyword arguments.
        prob_weights : list of floats
            List of probabilities for each operator.
        sampler : Sampler           
            Sampler class for selecting the next index for the SPDHG update.

        Notes
        -----
        This method is called by the set_up method.
        """
        
        prob = deprecated_kwargs.pop('prob', None)

        if prob is not None:
            if (prob_weights is None) and (sampler is None):
                warnings.warn('`prob` is being deprecated to be replaced with a sampler class and `prob_weights`. To randomly sample with replacement use "sampler=Sampler.randomWithReplacement(number_of_subsets,  prob=prob). To pass probabilities to the calculation for `sigma` and `tau` please use `prob_weights`. ', DeprecationWarning, stacklevel=2)
                self._prob_weights = prob
                self._sampler = Sampler.random_with_replacement(
                    len(self.operator), prob=prob)
            else:

                raise ValueError(
                    '`prob` is being deprecated to be replaced with a sampler class and `prob_weights`. You passed  a `prob` argument, and either a `prob_weights` argument or a sampler. Please remove the `prob` argument.')



    def _deprecated_set_norms(self, deprecated_kwargs):
        """
        Handle deprecated keyword arguments for backward compatibility.

        Parameters
        ----------
        deprecated_kwargs : dict
            Dictionary of keyword arguments.

        Notes
        -----
        This method is called by the set_up method.
        """
        norms = deprecated_kwargs.pop('norms', None)
        
        if norms is not None:
            self.operator.set_norms(norms)
            warnings.warn(
                ' `norms` is being deprecated, use instead the `BlockOperator` function `set_norms`', DeprecationWarning, stacklevel=2)

    def _deprecated_else(self, deprecated_kwargs):
        """
        Handle deprecated keyword arguments for backward compatibility.

        Parameters
        ----------
        deprecated_kwargs : dict
            Dictionary of keyword arguments.

        Notes
        -----
        This method is called by the set_up method.
        """
        if deprecated_kwargs:
            raise ValueError("Additional keyword arguments passed but not used: {}".format(deprecated_kwargs))
        
        
    @property
    def sigma(self):
        return self._sigma

    @property
    def tau(self):
        return self._tau

    def set_step_sizes_from_ratio(self, gamma=1.0, rho=0.99):
        r""" Sets gamma, the step-size ratio for the SPDHG algorithm. Currently gamma takes a scalar value.

        The step sizes `sigma` and `tau` are set using the equations:

        .. math:: \sigma_i= \frac{\gamma\rho }{\|K_i\|^2}

        .. math::  \tau = \rho\min_i([ \frac{p_i }{\sigma_i  \|K_i\|^2})


        Parameters
        ----------
            gamma : Positive float
                parameter controlling the trade-off between the primal and dual step sizes
            rho : Positive float
                 parameter controlling the size of the product :math: \sigma\tau :math:



        """
        if isinstance(gamma, Number):
            if gamma <= 0:
                raise ValueError(
                    "The step-sizes of SPDHG are positive, gamma should also be positive")

        else:
            raise ValueError(
                "We currently only support scalar values of gamma")
        if isinstance(rho, Number):
            if rho <= 0:
                raise ValueError(
                    "The step-sizes of SPDHG are positive, rho should also be positive")

        else:
            raise ValueError(
                "We currently only support scalar values of gamma")

        self._sigma = [gamma * rho / ni for ni in self._norms]
        values = [rho*pi / (si * ni**2) for pi, ni,
                  si in zip(self._prob_weights, self._norms, self._sigma)]
        self._tau = min([value for value in values if value > 1e-8])

    def set_step_sizes(self, sigma=None, tau=None):
        r""" Sets sigma and tau step-sizes for the SPDHG algorithm after the initial set-up. The step sizes can be either scalar or array-objects.

        When setting `sigma` and `tau`, there are 4 possible cases considered by setup function: 

        - Case 1: If neither `sigma` or `tau` are provided then `sigma` is set using the formula:

        .. math:: \sigma_i= \frac{0.99}{\|K_i\|^2}

        and `tau` is set as per case 2

        - Case 2: If `sigma` is provided but not `tau` then `tau` is calculated using the formula 

        .. math:: \tau = 0.99\min_i( \frac{p_i}{ (\sigma_i  \|K_i\|^2) })

        - Case 3: If `tau` is provided but not `sigma` then `sigma` is calculated using the formula

        .. math:: \sigma_i= \frac{0.99 p_i}{\tau\|K_i\|^2}

        - Case 4: Both `sigma` and `tau` are provided.


        Parameters
        ----------
            sigma : list of positive float, optional, default= see docstring
                List of Step size parameters for dual problem
            tau : positive float, optional, default= see docstring
                Step size parameter for primal problem

        """
        gamma = 1.
        rho = .99
        if sigma is not None:
            if len(sigma) == self._ndual_subsets:
                if all(isinstance(x, Number) and x > 0 for x in sigma):
                    pass
                else:
                    raise ValueError(
                        "Sigma expected to be a positive number.")

            else:
                raise ValueError(
                    "Please pass a list of floats to sigma with the same number of entries as number of operators")
            self._sigma = sigma

        elif tau is None:
            self._sigma = [gamma * rho / ni for ni in self._norms]
        else:
            self._sigma = [
                rho*pi / (tau*ni**2) for ni, pi in zip(self._norms, self._prob_weights)]

        if tau is None:
            values = [rho*pi / (si * ni**2) for pi, ni,
                      si in zip(self._prob_weights, self._norms, self._sigma)]
            self._tau = min([value for value in values if value > 1e-8])

        else:
            if not ( isinstance(tau, Number) and tau > 0):
                raise ValueError(
                    "The step-sizes of SPDHG must be positive, passed tau = {}".format(tau))

            self._tau = tau

    def check_convergence(self):
        """  Checks whether convergence criterion for SPDHG is satisfied with the current scalar values of tau and sigma

        Returns
        -------
        Boolean
            True if convergence criterion is satisfied. False if not satisfied or convergence is unknown. 

        Note
        -----
        Convergence criterion currently can only be checked for scalar values of tau.

        Note
        ----
        This checks the convergence criterion. Numerical errors may mean some sigma and tau values that satisfy the convergence criterion may not converge. 
        Alternatively, step sizes outside the convergence criterion may still allow (fast) convergence. 
        """
        for i in range(self._ndual_subsets):
            if isinstance(self._tau, Number) and isinstance(self._sigma[i], Number):
                if self._sigma[i] * self._tau * self._norms[i]**2 > self._prob_weights[i]:
                    return False
                return True
            else:
                raise ValueError('Convergence criterion currently can only be checked for scalar values of tau and sigma[i].')

    def update(self):
        """  Runs one iteration of SPDHG 

        """
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self._zbar.sapyb(self._tau,  self.x, -1., out=self._x_tmp )
        self._x_tmp*=-1

        self.g.proximal(self._x_tmp, self._tau, out=self.x)

        # Choose subset
        i = next(self._sampler)

        # Gradient ascent for the dual variable
        # y_k = y_old[i] + sigma[i] * K[i] x
        try:
            y_k = self.operator[i].direct(self.x)
        except IndexError:
            raise IndexError(
                'The sampler has outputted an index larger than the number of operators to sample from. Please ensure your sampler samples from {0,1,...,len(operator)-1} only.')

        y_k.sapyb(self._sigma[i], self._y_old[i], 1., out=y_k)

        y_k = self.f[i].proximal_conjugate(y_k, self._sigma[i])

        # Back-project
        # x_tmp = K[i]^*(y_k - y_old[i])
        y_k.subtract(self._y_old[i], out=self._y_old[i])

        self.operator[i].adjoint(self._y_old[i], out=self._x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self._z.add(self._x_tmp, out=self._z)
        # zbar = z + (theta/p[i]) * x_tmp

        self._z.sapyb(1., self._x_tmp, self._theta /
                      self._prob_weights[i], out=self._zbar)

        # save previous iteration
        self._save_previous_iteration(i, y_k)

    def update_objective(self):
        # p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        p1 = 0.
        for i, op in enumerate(self.operator.operators):
            p1 += self.f[i](op.direct(self.x))
        p1 += self.g(self.x)

        d1 = - self.f.convex_conjugate(self._y_old)
        tmp = self.operator.adjoint(self._y_old)
        tmp *= -1
        d1 -= self.g.convex_conjugate(tmp)

        self.loss.append([p1, d1, p1-d1])

    @property
    def objective(self):
        '''The saved primal objectives. 

        Returns
        -------
        list
            The saved primal objectives from `update_objective`. The number of saved values depends on the `update_objective_interval` kwarg. 
        '''
        return [x[0] for x in self.loss]

    @property
    def dual_objective(self):
        '''The saved dual objectives. 

        Returns
        -------
        list
            The saved dual objectives from `update_objective`. The number of saved values depends on the `update_objective_interval` kwarg. 
        '''
        return [x[1] for x in self.loss]

    @property
    def primal_dual_gap(self):
        '''The saved primal-dual gap. 

        Returns
        -------
        list
            The saved primal dual gap from `update_objective`. The number of saved values depends on the `update_objective_interval` kwarg. 
        '''
        return [x[2] for x in self.loss]

    def _save_previous_iteration(self, index, y_current):
        ''' Internal function used to save the previous iteration 
        '''
        self._y_old[index].fill(y_current)
