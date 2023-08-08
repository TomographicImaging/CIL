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
class Sampling():
    
    r"""Takes an integer number of subsets and a sampling type and returns a class object with a next function. On each call of next, an integer value between 0 and the number of subsets is returned, the next sample."""
    
    def __init__(self, num_subsets, sampling_type='sequential', prob=None, seed=99):
        self.type=sampling_type
        self.num_subsets=num_subsets
        self.seed=seed
        
        self.last_subset=-1
        if self.type=='sequential':
            pass
        elif self.type=='random':
            if prob==None:
                self.prob = [1/self.num_subsets] * self.num_subsets
            else:
                self.prob=prob
        elif self.type=='herman_meyer':
            
            self.order=self.herman_meyer_order(self.num_subsets)
        else:
            raise NameError('Please choose from sequential, random, herman_meyer')

    
    def herman_meyer_order(self, n):
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

    def next(self):
        if self.type=='sequential':
            self.last_subset= (self.last_subset+1)%self.num_subsets
            return self.last_subset
        elif self.type=='random':
            if self.last_subset==-1:
                np.random.seed(self.seed)
                self.last_subset=0
            return int(np.random.choice(self.num_subsets, 1, p=self.prob))
        elif self.type=='herman_meyer':
            self.last_subset= (self.last_subset+1)%self.num_subsets
            return(self.order[self.last_subset])


    def show_epochs(self, num_epochs=2):
        if self.type=='sequential':
            for i in range(num_epochs):
                print('Epoch {}: '.format(i), [j for j in range(self.num_subsets)])
        elif self.type=='random':
            np.random.seed(self.seed)
            for i in range(num_epochs):
                print('Epoch {}: '.format(i), np.random.choice(self.num_subsets, self.num_subsets, p=self.prob))
        elif self.type=='herman_meyer':
            for i in range(num_epochs):
                print('Epoch {}: '.format(i), self.order)
   
