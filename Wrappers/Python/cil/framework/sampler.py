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
    
    r"""
    A class to select from a list of integers {0, 1, …, S-1}, with each integer representing the index of a subset
    The function next() outputs a single next index from the {0,1,…,S-1} subset list. Different orders possible incl with and without replacement. To be run again and again, depending on how many iterations/epochs the users asks for.
    
    Calls are organised into epochs:  The single index outputs can be organised into length-S lists. Each length-S list is called an epoch. The user can in principle ask for an infinite number of epochs to be run. Denote by E the number of epochs.
    Each epoch always has a list of length S.  It may contain the same subset index s multiple times or not at all.

    Parameters
    ----------
    num_subsets: int
        The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 
    
    sampling_type:str
        The sampling type used. 

    order: list of integers
        The list of integers the method selects from using next. 
    
    shuffle= bool, default=False
        If True, after each epoch (num_subsets calls of next), the sampling order is shuffled randomly. 

    prob: list of floats of length num_subsets that sum to 1. 
        For random sampling with replacement, this is the probability for each integer to be called by next. 

    seed:int, default=None
        Random seed for the methods that use a random number generator.  
    


    Example
    -------

    >>> sampler=Sampler.sequential(10)
    >>> sampler.show_epochs(5)
    >>> for _ in range(11):
            print(sampler.next())

    Epoch 0:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Epoch 1:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Epoch 2:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Epoch 3:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Epoch 4:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    0
    1

    Example
    -------
    >>> sampler=Sampler.randomWithReplacement(11)
    >>> for _ in range(12):
    >>>     print(next(sampler))
    >>> sampler.show_epochs(5)
    
    10
    5
    10
    1
    6
    7
    10
    0
    0
    2
    5
    3
    Epoch 0:  [10, 5, 10, 1, 6, 7, 10, 0, 0, 2, 5]
    Epoch 1:  [3, 10, 7, 7, 8, 7, 4, 7, 8, 4, 9]
    Epoch 2:  [0, 0, 0, 1, 3, 8, 6, 5, 7, 7, 0]
    Epoch 3:  [8, 8, 6, 4, 0, 2, 7, 2, 8, 3, 8]
    Epoch 4:  [10, 9, 3, 6, 6, 9, 5, 2, 8, 4, 0]



    """
    
    @staticmethod
    def sequential(num_subsets):
        """
        Function that outputs a sampler that outputs sequentially. 

        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 

        Example
        -------

        >>> sampler=Sampler.sequential(10)
        >>> sampler.show_epochs(5)
        >>> for _ in range(11):
                print(sampler.next())

        Epoch 0:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Epoch 1:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Epoch 2:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Epoch 3:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Epoch 4:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        0
        1
        """
        order=list(range(num_subsets))
        sampler=Sampler(num_subsets, sampling_type='sequential', order=order)
        return sampler 
    
    @staticmethod
    def customOrder( customlist):
        """
        Function that outputs a sampler that outputs from a list, one entry at a time before cycling back to the beginning. 

        customlist: list of integers
            The list that will be sampled from in order. 
        """
        num_subsets=len(customlist)
        sampler=Sampler(num_subsets, sampling_type='custom_order', order=customlist)
        return  sampler

    @staticmethod
    def hermanMeyer(num_subsets):
        
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
    def staggered(num_subsets, offset):
        indices=list(range(num_subsets))
        order=[]
        [order.extend(indices[i::offset]) for i in range(offset)]
       # order=[indices[i::offset] for i in range(offset)]
        print(order)
        sampler=Sampler(num_subsets, sampling_type='staggered', order=order)
        return sampler 
    
    

    @staticmethod
    def randomWithReplacement(num_subsets, prob=None, seed=None):
        if prob==None:
            prob = [1/num_subsets] *num_subsets
        else:   
            prob=prob
        if len(prob)!=num_subsets:
            raise ValueError("Length of the list of probabilities should equal the number of subsets")
        if sum(prob)!=1.:
            raise ValueError("Probabilites should sum to 1.")
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
                #print(self.order)
        self.last_subset= (self.last_subset+1)%self.num_subsets
        return(self.order[self.last_subset])
    
    def _next_prob(self):
        return int(self.generator.choice(self.num_subsets, 1, p=self.prob))

    def next(self):
        return (self.iterator())

    def __next__(self):
        return(self.next())

    def show_epochs(self, num_epochs=2):
        save_generator=self.generator
        save_last_subset=self.last_subset
        self.last_subset=self.num_subsets-1
        save_order=self.order
        self.order=self.initial_order
        self.generator=np.random.RandomState(self.seed)
        for i in range(num_epochs):
            print('Epoch {}: '.format(i), [self.next() for _ in range(self.num_subsets)])
        self.generator=save_generator
        self.order=save_order
        self.last_subset=save_last_subset

    def get_epochs(self, num_epochs=2):
        save_generator=self.generator
        save_last_subset=self.last_subset
        self.last_subset=self.num_subsets-1
        save_order=self.order
        self.order=self.initial_order
        self.generator=np.random.RandomState(self.seed)
        output=[]
        for i in range(num_epochs):
            output.append( [self.next() for _ in range(self.num_subsets)])
        self.generator=save_generator
        self.order=save_order
        self.last_subset=save_last_subset
        return(output)

