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
import logging

class RandomSampling():
    
    r"""RandomSampling.

    RandomSampling is a class tha generates randomly indices or batches from a list of integers of length = `num_indices`, e.g., :code:`np.arange(num_indices)`.
    
    Parameters
    ----------

    num_indices : {int}
            A list of length :code:`num_indices`, e.g., :code:`np.arange(num_indices)` is generated.
    num_batches : {int}
            The number of batches to split the generated list. Default = num_indices.
            A warning is raised when :code:`num_batches` is not a divisor of :code:`num_indices`.
    prob : 1-D array_like, optional
            A list of probabilities of length :code:`num_indices`. Default = None.
            If :code:`None`, a uniform distribution is assumed for all entries in the generated list.
    replace : {bool}
            Whether to use replacement or not when an item from the generated list is selected. Default = True.
            If :code:`True`, an element from the list can be selected multiple times.  
    shuffle : {bool}
            Whether to shuffle the generated list at the end of each epoch. Default = True 
    seed : {int} 
            A seed to initialize the random generator.   
                 

    See also
    --------
    `np.random.choice <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice>`_ 

    """

    
    def __new__(cls, num_indices, num_batches=None, prob = None, replace = True, shuffle=False, seed = None):
        
        cls.num_batches = num_batches

        if cls.num_batches is None:
            cls.num_batches = num_indices
        
        if cls.num_batches == num_indices:
            return super(RandomSampling, cls).__new__(RandomIndex)
        else:
            return super(RandomSampling, cls).__new__(RandomBatch)
    
    def __init__(self, num_indices, num_batches=None, prob = None, replace = True, shuffle=False, seed = None ):
        
        self.num_indices = num_indices
        self.num_batches = num_batches

        if self.num_batches is None:
            self.num_batches = num_indices

        self.prob = prob
        self.replace = replace
        self.shuffle = shuffle
        self.indices_used = []
        self.index = 0
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.list_of_indices =  self.rng.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=self.replace) 
                    
        self.equal_size_batches = self.num_indices%self.num_batches==0        
        if self.equal_size_batches:
            self.batch_size = self.num_indices//self.num_batches
        else:
            logging.warning("Batch size is not constant")
            self.batch_size = (self.num_indices//self.num_batches)+1  

        if self.batch_size>1: 
            self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)]             
            
    def show_epochs(self, epochs):
        
        total_its = epochs * self.num_indices
        for _ in range(total_its):
            next(self)
                    
        if self.batch_size==1:
            k = 0
            for i in range(epochs):
                print(" Epoch : {}, indices used : {} ".format(i, self.indices_used[k:k+self.num_indices]))    
                k+=self.num_indices 
            print("")   
        else:
            k=0
            for i in range(epochs):
                print(" Epoch : {}, batches used : {} ".format(i, self.indices_used[k:k+self.num_batches]))                 
                k += self.num_batches
                
            
    @staticmethod    
    def uniform(num_indices, num_batches = None, replace = True, seed=None):
        
        return RandomSampling(num_indices,  num_batches=num_batches, prob=None, replace = replace, shuffle = False, seed = seed)
    
    @staticmethod
    def non_uniform(num_indices, prob, num_batches = None, replace = True, seed=None):
        return RandomSampling(num_indices, num_batches=num_batches, replace = replace, prob=prob, shuffle = False, seed = seed) 
    
    # @staticmethod    
    # def uniform_no_replacement(num_indices, num_batches = None, shuffle=True, seed=None):
    #     return RandomSampling(num_indices, num_batches=num_batches, prob=None, replace = False, shuffle = shuffle, seed = seed) 
    
    # @staticmethod    
    # def non_uniform_no_replacement(num_indices, prob, num_batches = None, shuffle=False, seed=None):
    #     return RandomSampling(num_indices, num_batches=num_batches, prob=prob, replace = False, shuffle = shuffle, seed = seed)     
        
    @staticmethod
    def single_shuffle(num_indices, num_batches = None, prob=None, seed=None):
        return RandomSampling(num_indices, num_batches = num_batches, prop=prob, replace = False, shuffle = False, seed = seed)
    
    @staticmethod
    def random_shuffle(num_indices, num_batches = None, seed=None):
        return RandomSampling(num_indices, num_batches = num_batches, replace = False, shuffle = True, seed = seed)              
            

class RandomBatch(RandomSampling):
            
    def __next__(self):
        
        tmp_list = list(self.partition_list[self.index])
        self.indices_used.append(tmp_list)         
        self.index+=1
        
        if self.index==len(self.partition_list):
            self.index=0            
            
            if self.shuffle is True:
                self.list_of_indices = self.rng.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=self.replace)        
                self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)]
                                               
        return tmp_list
    
class RandomIndex(RandomSampling):   
    
    def __next__(self):

        index_num = self.list_of_indices[self.index]
        self.index+=1   

        if self.index == self.num_indices:
            self.index = 0                
            if self.shuffle is True:                    
                self.list_of_indices = self.rng.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=self.replace)         

        self.indices_used.append(index_num)

        return index_num  
        
if __name__ == "__main__":

    sq1 = RandomSampling([1,2,3,4,5,6,7],num_batches=10)
    sq1.show_epochs(1)