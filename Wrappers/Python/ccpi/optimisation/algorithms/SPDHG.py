# -*- coding: utf-8 -*-
# Copyright 2020 Science Technology Facilities Council
# Copyright 2020 University of Manchester
# Copyright 2020 University of Bath
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ccpi.optimisation.algorithms import Algorithm
import numpy as np

class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient
    
    Problem: 
    
    .. math::
    
      \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)
        
    :param operator: BlockOperator of Linear Operators
    :param f: BlockFunction, each function with "simple" proximal of its conjugate 
    :param g: Convex function with "simple" proximal 
    :param sigma=(sigma_i): List of Step size parameters for Dual problem
    :param tau: Step size parameter for Primal problem
    :param x_init: Initial guess ( Default x_init = 0)
    :param prob: List of probabilities
        
    Remark: Convergence is guaranted provided that [2, eq. (12)]:
        
    .. math:: 
    
      \|\sigma[i]^{1/2} * K[i] * tau^{1/2} \|^2  < p_i for all i
      
    Remark: Notation for primal and dual step-sizes are reversed with comparison
            to PDGH.py
            
    Remark: this code implements serial sampling only, as presented in [2]
            (to be extended to more general case of [1] as future work)             
            
    References:
        
        [1]"Stochastic primal-dual hybrid gradient algorithm with arbitrary 
        sampling and imaging applications",
        Chambolle, Antonin, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schonlieb,
        SIAM Journal on Optimization 28, no. 4 (2018): 2783-2808.   
         
        [2]"Faster PET reconstruction with non-smooth priors by randomization and preconditioning",
        Matthias J Ehrhardt, Pawel Markiewicz and Carola-Bibiane Schönlieb,
        Physics in Medicine & Biology, Volume 64, Number 22, 2019.
    
    Example of usage: See https://github.com/vais-ral/CIL-Demos/blob/master/Tomography/Simulated/Single%20Channel/PDHG_vs_SPDHG.py
        
    '''
    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=None,
                 x_init=None, prob=None, gamma=1., use_axpby=True, 
                 norms=None, **kwargs):
        '''SPDHG algorithm creator

        Parameters
        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate 
        :param g: Convex function with "simple" proximal 
        :param sigma=(sigma_i): List of Step size parameters for Dual problem
        :param tau: Step size parameter for Primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities
        :param gamma: parameter controlling the trade-off between the primal and dual step sizes
        :param use_axpby: whether to use axpby or not
        :param norms: norms of the operators in operator
        :type norms: list, default None
        '''
        super(SPDHG, self).__init__(**kwargs)
        self._use_axpby = use_axpby
        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, 
                        x_init=x_init, prob=prob, gamma=gamma, norms=norms)
    
    def set_up(self, f, g, operator, tau=None, sigma=None, \
               x_init=None, prob=None, gamma=1., norms=None):
        '''initialisation of the algorithm

        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate.
        :param g: Convex function with "simple" proximal 
        :param sigma: list of Step size parameters for dual problem
        :param tau: Step size parameter for primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities'''
        print("{} setting up".format(self.__class__.__name__, ))
                    
        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator
        self.tau = tau
        self.sigma = sigma
        self.prob = prob
        self.ndual_subsets = len(self.operator)
        self.gamma = gamma
        self.rho = .99
        
        if self.prob is None:
            self.prob = [1/self.ndual_subsets] * self.ndual_subsets

        
        if self.sigma is None:
            if norms is None:
                # Compute norm of each sub-operator       
                norms = [operator.get_item(i,0).norm() for i in range(self.ndual_subsets)]
            self.norms = norms
            self.sigma = [self.gamma * self.rho / ni for ni in norms] 
        if self.tau is None:
            self.tau = min( [ pi / ( si * ni**2 ) for pi, ni, si in zip(self.prob, norms, self.sigma)] ) 
            self.tau *= (self.rho / self.gamma)

        # initialize primal variable 
        if x_init is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = x_init.copy()
        
        self.x_tmp = self.operator.domain_geometry().allocate(0)
        
        # initialize dual variable to 0
        self.y_old = operator.range_geometry().allocate(0)
        
        # initialize variable z corresponding to back-projected dual variable
        self.z = operator.domain_geometry().allocate(0)
        self.zbar= operator.domain_geometry().allocate(0)
        # relaxation parameter
        self.theta = 1
        self.update_objective()
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
    def update(self):
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        if self._use_axpby:
            self.x.axpby(1., -self.tau, self.zbar, out=self.x_tmp)
        else:
            self.zbar.multiply(self.tau, out=self.x_tmp)
            self.x.subtract(self.x_tmp, out=self.x_tmp)
            
        self.g.proximal(self.x_tmp, self.tau, out=self.x)
        
        # Choose subset
        i = int(np.random.choice(len(self.sigma), 1, p=self.prob))
        
        # Gradient ascent for the dual variable
        # y_k = y_old[i] + sigma[i] * K[i] x
        y_k = self.operator[i].direct(self.x)
        if self._use_axpby:
            y_k.axpby(self.sigma[i], 1., self.y_old[i], out=y_k)
        else:
            y_k.multiply(self.sigma[i], out=y_k)
            y_k.add(self.y_old[i], out=y_k)
            
        self.f[i].proximal_conjugate(y_k, self.sigma[i], out=y_k)
        
        # Back-project
        # x_tmp = K[i]^*(y_k - y_old[i])
        y_k.subtract(self.y_old[i], out=self.y_old[i])

        self.operator[i].adjoint(self.y_old[i], out = self.x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self.z.add(self.x_tmp, out =self.z)
        # zbar = z + (theta/p[i]) * x_tmp
        if self._use_axpby:
            self.z.axpby(1., self.theta / self.prob[i], self.x_tmp, out = self.zbar)
        else:
            self.x_tmp.multiply(self.theta / self.prob[i], out=self.x_tmp)
            self.z.add(self.x_tmp, out=self.zbar)
        
        # save previous iteration
        self.save_previous_iteration(i, y_k)
        
    def update_objective(self):
         p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
         d1 = - self.f.convex_conjugate(self.y_old)
         tmp = -1*self.operator.adjoint(self.y_old)
         d1 += self.g.convex_conjugate(tmp)

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
