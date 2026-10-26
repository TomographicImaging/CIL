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
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.algorithms import Algorithm
import logging
import warnings
import numpy as np
from copy import deepcopy

log = logging.getLogger(__name__)

class ProxSkip(Algorithm):
    

    r"""Proximal Skip (ProxSkip) for convex optimisation.


        .. math:: \min_{x} f(x) + g(x),

        where :math:`f` has a Lipschitz-continuous gradient and :math:`g` is convex with a "simple" proximal operator. 
        ProxSkip can be particularly useful when the proximal operator is computationally expensive, as it reduces
        the frequency of proximal evaluations. The proximal operator is evaluated with probability `prob`. 
        As a convention, we always compute the proximal operator in the first iteration.
    
        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. 
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A convex function with a "simple" proximal operator.
        prob : positive :obj:`float`
            Probability of performing the proximal step. If :code:`prob=1`, proximal step is used in every iteration.             
        step_size : positive :obj:`float`
            Step size for the ProxSkip algorithm. It is equal to 1./L for strongly convex f and 2./L for convex f, where L is the  Lipschitz constant for the gradient of f.   
        seed : int or numpy.random.Generator-compatible seed, optional
            Seed for the independent random number generator.            


        See Mishchenko, Malinovsky, Stich and Richtarik (2022),
        ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication
        Acceleration! Finally!, ICML, https://proceedings.mlr.press/v162/mishchenko22b.html. 

        See Condat and Richtárik (2023),
        RandProx: Primal-Dual Optimization Algorithms with Randomized Proximal Updates, ICLR, https://arxiv.org/abs/2207.12891.                   

        Papoutsellis, Kereta and Papafitsoros (2025),
        Why Do We Regularise in Every Iteration for Imaging Inverse Problems?, Scale Space and Variational Methods in Computer Vision (SSVM 2025),
        pp. 43–55, https://doi.org/10.1007/978-3-031-92366-1_4.        
            
     """


    def __init__(self, initial, f, g, step_size, prob, seed=None, **kwargs):
        """ Set up of the algorithm
        """        

        super(ProxSkip, self).__init__(**kwargs)

        self.f = f # smooth function
        self.g = g # proximable
        self.step_size = step_size
        self.prob = prob
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        # get prox iterates after run
        self._initial_rng = deepcopy(self.rng) if seed is not None else None

        if self.prob<=0:
            raise ValueError("Need a positive probability")
        if self.prob == 1.0:
            warnings.warn("With prob=1, ProxSkip is equivalent to ISTA/PGD. Consider using ISTA instead.")

        self.set_up(initial, f, g, step_size, prob, **kwargs)
        
                  
    def set_up(self, initial, f, g, step_size, prob, **kwargs):

        log.info("%s setting up", self.__class__.__name__)
      
        self.x = initial.copy()   
        self.xhat_new = initial.copy()
        self.x_new = initial.copy()
        self.ht = initial.copy() 
        self.tmp_ = initial.copy()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)
                                  

    def update(self):
        r""" Performs a single iteration of the ProxSkip algorithm        
        """
        
        self.f.gradient(self.x, out=self.xhat_new)
        self.xhat_new -= self.ht
        self.x.sapyb(1., self.xhat_new, -self.step_size, out=self.xhat_new)

        theta = True if self.iteration == 0 else self.rng.random() < self.prob

        if theta:
            # Proximal step is used
            self.xhat_new.sapyb(1.0, self.ht, -(self.step_size/self.prob), out=self.tmp_)
            self.g.proximal(self.tmp_, (self.step_size/self.prob), out=self.x_new)
            
            self.x_new.subtract(self.xhat_new, out=self.tmp_)
            self.ht.sapyb(1., self.tmp_, (self.prob/self.step_size), out=self.ht) 
           
        else:
            self.x_new.fill(self.xhat_new)

    def _update_previous_solution(self):
        self.x, self.x_new = self.x_new, self.x            

    def get_output(self):
        " Returns the current solution. "
        return self.x        
                        
    def update_objective(self):

        """ Updates the objective

        .. math:: f(x) + g(x)

        """        

        self.loss.append( self.f(self.x) + self.g(self.x) )

    def get_proximal_iterates(self):

        if self.prob == 1.0:
            return [True] * self.iteration

        prox_iterates = []
        rng = deepcopy(self._initial_rng)
        for k in range(self.iteration):
            theta = True if k == 0 else rng.random() < self.prob
            prox_iterates.append(theta)
        return prox_iterates
               
        

