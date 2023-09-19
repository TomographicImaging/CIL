# -*- coding: utf-8 -*-
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
import numpy as np
import warnings
import logging
from cil.framework import Sampler
from numbers import Number

class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient

    Problem: 

    .. math::

      \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)

    Parameters
    ----------
    f : BlockFunction
        Each must be a convex function with a "simple" proximal method of its conjugate
    g : Function
        A convex function with a "simple" proximal
    operator : BlockOperator   
        BlockOperator must contain Linear Operators    
    tau : positive float, optional, default=None
        Step size parameter for Primal problem
    sigma : list of positive float, optional, default=None
        List of Step size parameters for Dual problem
    initial : DataContainer, optional, default=None
        Initial point for the SPDHG algorithm
    prob : list of floats, optional, default=None
        List of probabilities. If None each subset will have probability = 1/number of subsets
    gamma : float
        parameter controlling the trade-off between the primal and dual step sizes
    sampler: instance of the Sampler class
        Method of selecting the next mini-batch. If None, random sampling and each subset will have probability = 1/number of subsets
    **kwargs:
    norms : list of floats
        precalculated list of norms of the operators

    Example 
    -------

    Example of usage: See https://github.com/vais-ral/CIL-Demos/blob/master/Tomography/Simulated/Single%20Channel/PDHG_vs_SPDHG.py


    Note
    ----

    Convergence is guaranteed provided that [2, eq. (12)]:

    .. math:: 

    \|\sigma[i]^{1/2} * K[i] * tau^{1/2} \|^2  < p_i for all i

    Note
    ----

    Notation for primal and dual step-sizes are reversed with comparison
        to PDHG.py



    References
    ----------

    [1]"Stochastic primal-dual hybrid gradient algorithm with arbitrary 
    sampling and imaging applications",
    Chambolle, Antonin, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schonlieb,
    SIAM Journal on Optimization 28, no. 4 (2018): 2783-2808.   

    [2]"Faster PET reconstruction with non-smooth priors by randomization and preconditioning",
    Matthias J Ehrhardt, Pawel Markiewicz and Carola-Bibiane Schönlieb,
    Physics in Medicine & Biology, Volume 64, Number 22, 2019.
    '''

    def __init__(self, f=None, g=None,  operator=None, tau=None, sigma=None,
                 initial=None,  gamma=1., sampler=None, **kwargs):

        super(SPDHG, self).__init__(**kwargs)

        self._prob_weights = kwargs.get('prob', None)
        if self._prob_weights is not None:
            warnings.warn('prob is being deprecated to be replaced with a sampler class. To randomly sample with replacement use "sampler=Sampler.randomWithReplacement(number_of_subsets,  prob=prob)".\
                          If you have passed both prob and a sampler then prob will be')

        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma,
                        initial=initial,  gamma=gamma, sampler=sampler, norms=kwargs.get('norms', None))

    @property
    def norms(self):
        return self._norms

    def set_norms(self, norms=None):
        #TODO: write some checks for setting norms 

        if norms is None:
            # Compute norm of each sub-operator
            norms = [self.operator.get_item(i, 0).norm()
                     for i in range(self.ndual_subsets)]
        self._norms = norms

    @property
    def sigma(self):
        return self._sigma

    def set_sigma(self, sigma=None, norms=None):
        #TODO: check if this is correct for PSDHG 
        if sigma is not None:
            if isinstance(sigma, Number):
                if sigma <= 0:
                    raise ValueError("The step-sizes of PDHG are positive, passed sigma = {}".format(sigma))                  
            elif sigma.shape != self.operator.range_geometry().shape:  
                raise ValueError(" The shape of sigma = {0} is not the same as the shape of the range_geometry = {1}".format(sigma.shape, self.operator.range_geometry().shape))



        self.set_norms(norms)
        if sigma is None:
            self._sigma = [self.gamma * self.rho / ni for ni in self._norms]
        else:
            self._sigma = sigma

    @property
    def tau(self):
        return self._tau

    def set_tau(self, tau=None):
        #TODO: check if this is correct for SPDHG
        if tau is not None:          
            if isinstance(tau, Number):
                if tau <= 0:
                    raise ValueError("The step-sizes of PDHG must be positive, passed tau = {}".format(tau))                  
            elif tau.shape != self.operator.domain_geometry().shape:  
                raise ValueError(" The shape of tau = {0} is not the same as the shape of the domain_geometry = {1}".format(tau.shape, self.operator.domain_geometry().shape))



        if tau is None:
            self._tau = min([pi / (si * ni**2) for pi, ni,
                            si in zip(self._prob_weights, self._norms, self._sigma)])
            self._tau *= (self.rho / self.gamma)
        else:
            self._tau = tau

    def set_step_sizes(self):
        ''' If you update either the norms or the prob_weights run this to reset the default sigma and tau step-sizes'''
        self.set_sigma()
        self.set_tau()
        
    def check_convergence(self):
         #TODO: check if this is correct for SPDHG 
        """  Check whether convergence criterion for SPDHG is satisfied with scalar values of tau and sigma

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
        else:
            warnings.warn("Convergence criterion can only be checked for scalar values of tau and sigma.")   
            return False             


    @property
    def prob_weights(self):
        return self._prob_weights

    def set_prob_weights(self, prob_weights=None):
        if prob_weights is None:
            self._prob_weights = [1/self.ndual_subsets] * self.ndual_subsets
        else:
            self._prob_weights = prob_weights

    def set_up(self, f, g, operator, tau=None, sigma=None,
               initial=None,  gamma=1., sampler=None, norms=None):
        '''set-up of the algorithm
        Parameters
        ----------
        f : BlockFunction
            Each must be a convex function with a "simple" proximal method of its conjugate
        g : Function
            A convex function with a "simple" proximal
        operator : BlockOperator   
            BlockOperator must contain Linear Operators    
        tau : positive float, optional, default=None
            Step size parameter for Primal problem
        sigma : list of positive float, optional, default=None
            List of Step size parameters for Dual problem
        initial : DataContainer, optional, default=None
            Initial point for the SPDHG algorithm

        gamma : float
            parameter controlling the trade-off between the primal and dual step sizes
        sampler: instance of the Sampler class
            Method of selecting the next mini-batch. If None, random sampling and each subset will have probability = 1/number of subsets. 
        **kwargs:
        norms : list of floats
            precalculated list of norms of the operators
        '''
        logging.info("{} setting up".format(self.__class__.__name__, ))

        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator
        self.sampler = sampler
        self.gamma = gamma
        self.ndual_subsets = self.operator.shape[0]
        self.rho = .99

        # Remove this if statement once prob is deprecated
        if self._prob_weights is None or sampler is not None:
            self.set_prob_weights(sampler.prob)
        if self.sampler is None:
            self.sampler = Sampler.randomWithReplacement(
                self.ndual_subsets,  prob=self._prob_weights)
        self.set_norms(norms)
        self.set_sigma(sigma)
        self.set_tau(tau)

        # initialize primal variable
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.x_tmp = self.operator.domain_geometry().allocate(0)

        # initialize dual variable to 0
        self.y_old = operator.range_geometry().allocate(0)

        # initialize variable z corresponding to back-projected dual variable
        self.z = operator.domain_geometry().allocate(0)
        self.zbar = operator.domain_geometry().allocate(0)
        # relaxation parameter
        self.theta = 1
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))

    def update(self):
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self.x.sapyb(1., self.zbar,  -self._tau, out=self.x_tmp)

        self.g.proximal(self.x_tmp, self._tau, out=self.x)

        # Choose subset
        i = self.sampler.next()

        # Gradient ascent for the dual variable
        # y_k = y_old[i] + sigma[i] * K[i] x
        y_k = self.operator[i].direct(self.x)

        y_k.sapyb(self._sigma[i], self.y_old[i], 1., out=y_k)

        y_k = self.f[i].proximal_conjugate(y_k, self._sigma[i])

        # Back-project
        # x_tmp = K[i]^*(y_k - y_old[i])
        y_k.subtract(self.y_old[i], out=self.y_old[i])

        self.operator[i].adjoint(self.y_old[i], out=self.x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self.z.add(self.x_tmp, out=self.z)
        # zbar = z + (theta/p[i]) * x_tmp

        self.z.sapyb(1., self.x_tmp, self.theta /
                     self._prob_weights[i], out=self.zbar)

        # save previous iteration
        self.save_previous_iteration(i, y_k)

    def update_objective(self):
        # p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        p1 = 0.
        for i, op in enumerate(self.operator.operators):
            p1 += self.f[i](op.direct(self.x))
        p1 += self.g(self.x)

        d1 = - self.f.convex_conjugate(self.y_old)
        tmp = self.operator.adjoint(self.y_old)
        tmp *= -1
        d1 -= self.g.convex_conjugate(tmp)

        self.loss.append([p1, d1, p1-d1])

    @property
    def objective(self):
        '''alias of loss'''
        return [x[0] for x in self.loss]

    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]

    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]

    def save_previous_iteration(self, index, y_current):
        self.y_old[index].fill(y_current)
