# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
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
    
      \|\sigma[i]^{1/2} * K[i] * tau^{1/2} \|  <1 for all i
      
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
        
        
    '''
    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=None,
                 x_init=None, prob=None, **kwargs):
        '''SPDHG algorithm creator
        Optional parameters
        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate 
        :param g: Convex function with "simple" proximal 
        :param sigma=(sigma_i): List of Step size parameters for Dual problem
        :param tau: Step size parameter for Primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities
        '''
        super(SPDHG, self).__init__(**kwargs)
        
        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, 
                        x_init=x_init, prob=prob)
    def set_up(self, f, g, operator, tau=None, sigma=None, x_init=None, prob=None):
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
        
        # Compute norm of each sub-operator       
        norms = [operator.get_item(i,0).norm() for i in range(self.ndual_subsets)]
        
        if self.sigma is None and self.tau is None:
            self.sigma = [1.] * self.ndual_subsets
            self.tau = 1 / sum([si * ni**2 for si, ni in zip(self.sigma, norms)])
            
        if self.prob is None:
            self.prob = [1/self.ndual_subsets] * self.ndual_subsets
        
        # initialize primal variable 
        if x_init is None:
            self.x = self.operator.domain_geometry().allocate()
        else:
            self.x = x_init.copy()
            
        self.x_tmp = self.operator.domain_geometry().allocate()
        
        # initialize dual variable to 0
        self.y = operator.range_geometry().allocate()
        self.y_old = operator.range_geometry().allocate()
        
        # initialize variable z corresponding to back-projected dual variable
        self.z = operator.domain_geometry().allocate()
        self.zbar= operator.domain_geometry().allocate()
        # relaxation parameter
        self.theta = 1
        self.update_objective()
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
    def update(self):
        
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self.x.axpby(1., -self.tau, self.zbar, out=self.x_tmp)
        self.g.proximal(self.x_tmp, self.tau, out=self.x)
        
        # Choose subset
        i = int(np.random.choice(len(self.sigma), 1, p=self.prob))
        
        # save previous iteration
        self.y_old[i].fill(self.y[i])
        
        # Gradient ascent for the dual variable
        # y[i] = y_old[i] + sigma[i] * K[i] x
        self.operator.get_item(i,0).direct(self.x, out=self.y[i])
        self.y[i].axpby(self.sigma[i], 1., self.y_old[i], out=self.y[i])
        self.f[i].proximal_conjugate(self.y[i], self.sigma[i], out=self.y[i])
        
        # Back-project
        # x_tmp = K[i]^*(y[i] - y_old[i])
        self.operator.get_item(i,0).adjoint(self.y[i]-self.y_old[i], out = self.x_tmp)
        # Update backprojected dual variable and extrapolate
        # z = z + x_tmp
        self.z.add(self.x_tmp, out =self.z)
        # zbar = z + (1 + theta/p[i]) x_tmp
        self.z.axpby(1., (1 + self.theta / self.prob[i]), self.x_tmp, out = self.zbar)
        
        
    def update_objective(self):
         p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
         d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))
#
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