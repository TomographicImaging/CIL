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
from cil.optimisation.utilities import RandomSampling

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

    Note
    ----

    this code implements serial sampling only, as presented in [2]
        (to be extended to more general case of [1] as future work)             
        
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
    
    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=None,
                 initial=None, prob=None, gamma=1., selection=None, **kwargs):

        super(SPDHG, self).__init__(**kwargs)


        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, 
                        initial=initial, prob=prob, gamma=gamma, selection=None, norms=kwargs.get('norms', None))
    

    def set_up(self, f, g, operator, tau=None, sigma=None, \
               initial=None, prob=None, gamma=1., selection = None, norms=None):
        
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
        prob : list of floats, optional, default=None
            List of probabilities. If None each subset will have probability = 1/number of subsets
        gamma : float
            parameter controlling the trade-off between the primal and dual step sizes

        **kwargs:
        norms : list of floats
            precalculated list of norms of the operators
        '''
        logging.info("{} setting up".format(self.__class__.__name__, ))
                    
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
        self.selection = selection
        
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

        if self.selection is None:
            self.selection = RandomSampling(len(self.sigma), prob=self.prob)            

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
        self.zbar= operator.domain_geometry().allocate(0)
        # relaxation parameter
        self.theta = 1
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
        
    def update(self):
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self.x.sapyb(1., self.zbar,  -self.tau, out=self.x_tmp)
            
        self.g.proximal(self.x_tmp, self.tau, out=self.x)
        
        # Choose subset
        i = next(self.selection)
        # i = int(np.random.choice(len(self.sigma), 1, p=self.prob))
        
        # Gradient ascent for the dual variable
        # y_k = y_old[i] + sigma[i] * K[i] x
        y_k = self.operator[i].direct(self.x)

        y_k.sapyb(self.sigma[i], self.y_old[i], 1., out=y_k)
            
        y_k = self.f[i].proximal_conjugate(y_k, self.sigma[i])
        
        # Back-project
        # x_tmp = K[i]^*(y_k - y_old[i])
        y_k.subtract(self.y_old[i], out=self.y_old[i])

        self.operator[i].adjoint(self.y_old[i], out = self.x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self.z.add(self.x_tmp, out =self.z)
        # zbar = z + (theta/p[i]) * x_tmp

        self.z.sapyb(1., self.x_tmp, self.theta / self.prob[i], out = self.zbar)

        # save previous iteration
        self.save_previous_iteration(i, y_k)
        
    def update_objective(self):
        # p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        p1 = 0.
        for i,op in enumerate(self.operator.operators):
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


if __name__=="__main__":

    dd = RandomSampling(10, shuffle=False)
    for i in range(100):
    
        z = next(dd)
        print(z)