# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
import math 
import time 
class Sampler():
    
    r"""Takes an integer number of subsets and a sampling type and returns a class object with a next function. On each call of next, an integer value between 0 and the number of subsets is returned, the next sample."""
    
        

    @staticmethod
    def hermanMeyer(num_subsets):
        @staticmethod
        def _herman_meyer_order(n):
            # Assuming that the subsets are in geometrical order
            n_variable = n
            i = 2
            factors = []
            while i * i <= n_variable:
                if n_variable % i:
                    i += 1
                else:
                    n_variable //= i
                    factors.append(i)
            if n_variable > 1:
                factors.append(n_variable)
            n_factors = len(factors)
            order =  [0 for _ in range(n)]
            value = 0
            for factor_n in range(n_factors):
                n_rep_value = 0
                if factor_n == 0:
                    n_change_value = 1
                else:
                    n_change_value = math.prod(factors[:factor_n])
                for element in range(n):
                    mapping = value
                    n_rep_value += 1
                    if n_rep_value >= n_change_value:
                        value = value + 1
                        n_rep_value = 0
                    if value == factors[factor_n]:
                        value = 0
                    order[element] = order[element] + math.prod(factors[factor_n+1:]) * mapping
            return order

        order=_herman_meyer_order(num_subsets)
        sampler=Sampler(num_subsets, sampling_type='herman_meyer', order=order)
        return sampler 

    @staticmethod
    def sequential(num_subsets):
        order=list(range(num_subsets))
        sampler=Sampler(num_subsets, sampling_type='sequential', order=order)
        return sampler 

    @staticmethod
    def randomWithReplacement(num_subsets, prob=None, seed=None):
        if prob==None:
            prob = [1/num_subsets] *num_subsets
        else:   
            prob=prob
        sampler=Sampler(num_subsets, sampling_type='random_with_replacement', prob=prob, seed=seed)
        return sampler 
    
    @staticmethod
    def randomWithoutReplacement(num_subsets, seed=None):
        order=list(range(num_subsets))
        sampler=Sampler(num_subsets, sampling_type='random_without_replacement', order=order, shuffle=True, seed=seed)
        return sampler 


    def __init__(self, num_subsets, sampling_type, shuffle=False, order=None, prob=None, seed=None):
        self.type=sampling_type
        self.num_subsets=num_subsets
        if seed !=None:
            self.seed=seed
        else:
            self.seed=int(time.time())
        self.generator=np.random.RandomState(self.seed)
        self.order=order
        self.initial_order=order
        if order!=None:
            self.iterator=self._next_order
        self.prob=prob
        if prob!=None:
            self.iterator=self._next_prob
        self.shuffle=shuffle
        self.last_subset=self.num_subsets-1
        


    
    def _next_order(self):
      #  print(self.last_subset)
        if self.shuffle==True and self.last_subset==self.num_subsets-1:
                self.order=self.generator.permutation(self.order)
                print(self.order)
        self.last_subset= (self.last_subset+1)%self.num_subsets
        return(self.order[self.last_subset])
    
    def _next_prob(self):
        return int(self.generator.choice(self.num_subsets, 1, p=self.prob))

    def next(self):
        return (self.iterator())


    def show_epochs(self, num_epochs=2):
        save_generator=self.generator
        save_order=self.order
        self.order=self.initial_order
        self.generator=np.random.RandomState(self.seed)
        for i in range(num_epochs):
            print('Epoch {}: '.format(i), [self.next() for _ in range(self.num_subsets)])
        self.generator=save_generator
        self.order=save_order
   
