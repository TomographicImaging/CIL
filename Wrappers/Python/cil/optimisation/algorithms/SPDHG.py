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
from cil.optimisation.utilities import Sampler, StepSizeRule, SPDHGConstantStepSize
from numbers import Number
import warnings
from cil.framework import BlockDataContainer

log = logging.getLogger(__name__)


class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient (SPDHG) solves separable optimisation problems of the type: 

      .. math:: \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)

    where :math:`f_i` and the regulariser :math:`g` need to be proper, convex and lower semi-continuous.

    Parameters
    ----------
    f : BlockFunction
        Each must be a convex function with a "simple" proximal method of its conjugate
    g : Function
        A convex function with a "simple" proximal
    operator : BlockOperator
        BlockOperator must contain Linear Operators
    step_size : tuple of (tau, sigma), optional
        A tuple containing the step size parameters for the primal and dual problems. If `None` will be computed by algorithm, see note for details.
    initial : `DataContainer`, or `list` or `tuple` of `DataContainer`s, optional, default is a DataContainer of zeros for both primal and dual variables
        Initial point for the PDHG algorithm. If just one data container is provided, it is used for the primal and the dual variable is initialised as zeros.  If a list or tuple is passed,  the first element is used for the primal variable and the second one for the dual variable. If either of the two is not provided, it is initialised as a DataContainer of zeros.
    gamma : float, optional
            Parameter controlling the trade-off between the primal and dual step sizes
    sampler: `cil.optimisation.utilities.Sampler`, optional 
            A `Sampler` controllingthe selection of the next index for the SPDHG update. If `None`, a sampler will be created for uniform random sampling with replacement. See notes.  

    prob_weights: list of floats, optional,  
             Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. Note that this should not be passed if the provided sampler has it as an attribute: if the sampler has a `prob_weight` attribute it will take precedence on this parameter. Should be a list of floats of length `num_indices` that sum to 1. If no sampler with `prob_weights` is passed, it defaults to `[1/len(operator)]*len(operator)`.


    Note  
    -----  
    The `sampler` can be an instance of the `cil.optimisation.utilities.Sampler` class or a custom class with the `__next__(self)` method implemented, which outputs an integer index from {1, ..., len(operator)}. 

    Note  
    -----  
    "Random sampling with replacement" will select the next index with equal probability from  `1 - len(operator)`.  


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

    def __init__(self, f=None, g=None, operator=None, step_size=None,
                 initial=None, sampler=None, prob_weights=None,   **kwargs):

        self.initial = initial
        self._sigma = kwargs.pop('sigma', None)  # To be deprecated
        self._tau = kwargs.pop('tau', None)  # To be deprecated
    
        if step_size is not None:  # To be deprecated
            if self._sigma is not None or self._tau is not None:  # To be deprecated
                raise ValueError("The parameters `sigma` and `tau` are being deprecated in favour of `step_size`. You have passed both. Instead please pass these as part of the `step_size` argument, either as a tuple of (sigma, tau) or using a compatible step size rule.", DeprecationWarning)

        if self._sigma is not None or self._tau is not None:  # To be deprecated
            warnings.warn("The parameters `sigma` and `tau` are being deprecated. In the future, please pass these as part of the `step_size` argument, either as a tuple of (sigma, tau) or using a compatible step size rule.", DeprecationWarning)
            step_size = (self._tau, self._sigma)
            
            
        update_objective_interval = kwargs.pop('update_objective_interval', 1)
        super(SPDHG, self).__init__(
            update_objective_interval=update_objective_interval)

        self.set_up(f=f, g=g, operator=operator, step_size=step_size,
                    initial=initial,  sampler=sampler, prob_weights=prob_weights)

    def set_up(self, f, g, operator, step_size=None,
               initial=None,   sampler=None, prob_weights=None):
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

        self._prob_weights = getattr(sampler, 'prob_weights', prob_weights)

        if self._prob_weights is None:
            self._prob_weights = [1/self._ndual_subsets]*self._ndual_subsets

        if prob_weights is not None and self._prob_weights != prob_weights:
            raise ValueError(
                ' You passed a `prob_weights` argument and a sampler with a different attribute `prob_weights`, please remove the `prob_weights` argument.')

        if sampler is None:
            self._sampler = Sampler.random_with_replacement(
                len(operator), prob=self._prob_weights)
        else:
            self._sampler = sampler

        # Set the norms of the operators
        self._norms = operator.get_norms_as_list()

        if step_size is None:  # This line can be removed when sigma and tau deprecated
            step_size = (None, None)
        if isinstance(step_size, StepSizeRule):
            self.step_size_rule = step_size
        elif isinstance(step_size, (tuple, list)):
            self.step_size_rule = SPDHGConstantStepSize(step_size=step_size)
        else:
            raise ValueError("The `step_size` argument must be either None, a SPDHG compatible step size rule or a tuple of (sigma, tau) where sigma is the step size for the dual problem and tau is the step size for the primal problem.")


        if isinstance(initial, (tuple, list)):
            if initial[0] is not None:
                self.x = initial[0].copy()
            else:
                self.x = self.operator.domain_geometry().allocate(0)


            if len(initial) > 1 and initial[1] is not None:
                self._y_old = initial[1].copy()
            else:
                self._y_old = self.operator.range_geometry().allocate(0)

        else:
            self._y_old = self.operator.range_geometry().allocate(0)
            if initial is None:
                self.x = self.operator.domain_geometry().allocate(0)
            else:
                self.x = initial.copy()

        self._x_tmp = self.operator.domain_geometry().allocate(0)
        
        # This can be removed once #1863 is fixed
        if not isinstance(self._y_old, BlockDataContainer):
            self._y_old = BlockDataContainer(self._y_old)

        # initialize variable z corresponding to back-projected dual variable
        self._z = operator.domain_geometry().allocate(0)
        self._zbar = operator.domain_geometry().allocate(0)
        # relaxation parameter
        self._theta = 1

        self._tau, self._sigma = self.step_size_rule.get_initial_step_size(
            self)
        
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))

    @property
    def sigma(self):
        return self._sigma

    @property
    def tau(self):
        return self._tau

    

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
                raise ValueError(
                    'Convergence criterion currently can only be checked for scalar values of tau and sigma[i].')
                
    def _spdhg_update(self, i):
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self._zbar.sapyb(self._tau,  self.x, -1., out=self._x_tmp)
        self._x_tmp *= -1

        self.g.proximal(self._x_tmp, self._tau, out=self.x)

        

        # Gradient ascent for the dual variable
        # y_k = y_old[i] + sigma[i] * K[i] x
        try:
            self.y_k = self.operator[i].direct(self.x)
        except IndexError:
            raise IndexError(
                'The sampler has outputted an index larger than the number of operators to sample from. Please ensure your sampler samples from {0,1,...,len(operator)-1} only.')

        self.y_k.sapyb(self._sigma[i], self._y_old[i], 1., out=self.y_k)

        self.y_k = self.f[i].proximal_conjugate(self.y_k, self._sigma[i])

        # Back-project
        # x_tmp = K[i]^*(y_k - y_old[i])
        self.y_k.subtract(self._y_old[i], out=self._y_old[i])

        self.operator[i].adjoint(self._y_old[i], out=self._x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self._z.add(self._x_tmp, out=self._z)
        # zbar = z + (theta/p[i]) * x_tmp

        self._z.sapyb(1., self._x_tmp, self._theta /
                      self._prob_weights[i], out=self._zbar)

        

    def update(self):
        """  Runs one iteration of SPDHG 

        """
        # Choose subset
        i = next(self._sampler)
        
        self._spdhg_update(i)
        
        # save previous iteration
        self._save_previous_iteration(i, self.y_k)
        
        self._tau, self._sigma = self.step_size_rule.get_step_size(self)
        
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
