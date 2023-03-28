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
from itertools import islice

class RandomSampling():
    
    r"""RandomSampling.

    RandomSampling is an iterator that generates randomly indices or batches from a list of integers of length = `num_indices`, e.g., :code:`np.arange(num_indices)`.
    
    Parameters
    ----------

    num_indices : int
            A list of length :code:`num_indices`, e.g., :code:`np.arange(num_indices)` is generated.
    num_batches : int, optional
            The number of batches to split the generated list. Default = num_indices.
            A warning is raised when :code:`num_batches` is not a divisor of :code:`num_indices`.
    prob : 1-D array_like, optional
            A list of probabilities of length :code:`num_indices`. Default = None.
            If :code:`None`, a uniform distribution is assumed for all entries in the generated list.
    replace : bool, optional
            Whether to use replacement or not, when an item from the generated list is selected. Default = True.
            If :code:`True`, an element from the list can be selected multiple times.  
    shuffle : bool, optional
            Whether to shuffle the generated list at the end of each epoch. Default = True.
    batch_size : list, optional
            The batch size for each `num_batches`. Default = None .
            If the `num_batches` is a divisor of `num_indices` then `batch_size` is `self.num_indices//self.num_batches`, by default.
            If the `num_batches` is not a divisor of `num_indices`, then `batch_size` is `self.num_indices//self.num_batches + 1`, by default. Otherwise, a list can be passed for each size of the `num_batches`.                    
    seed : int, optional
            A seed to initialize the random generator.   

    See also
    --------
    :class:`.SequentialSampling`
    
    `np.random.choice <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice>`_ 

  
    Examples
    --------

    Create a RandomSampling iterator for a list of 10 indices, np.arange(10) with replacement and shuffle:
    >>> rs1 = RandomSampling(10, seed=19)
    >>> rs1.show_epochs(2)
    Epoch : 0, indices used : [5, 4, 3, 9, 3, 2, 8, 0, 4, 3] 
    Epoch : 1, indices used : [0, 7, 9, 7, 3, 5, 8, 3, 4, 9] 

    Create a RandomSampling iterator for a list of 10 indices, np.arange(10) with replacement, shuffle and 2 batches:
    >>> rs1 = RandomSampling(10, num_batches = 2, seed=19)
    >>> rs1.show_epochs(2)
    Epoch : 0, batches used : [[5, 4, 3, 9, 3], [2, 8, 0, 4, 3]] 
    Epoch : 1, batches used : [[0, 7, 9, 7, 3], [5, 8, 3, 4, 9]]   

    Create a RandomSampling iterator for a list of 10 indices, np.arange(10) without replacement and shuffle and 2 batches:
    >>> rs1 = RandomSampling(10, num_batches = 2, replace=False, shuffle = False, seed=19)
    >>> rs1.show_epochs(2)
    Epoch : 0, batches used : [[7, 8, 9, 2, 1], [4, 6, 5, 0, 3]] 
    Epoch : 1, batches used : [[7, 8, 9, 2, 1], [4, 6, 5, 0, 3]]     

    Create a RandomSampling iterator for a list of 10 indices, np.arange(10) with 3 batches of batch_size = [1,3,6]:
    >>> rs1 = RandomSampling(10, num_batches = 3, batch_size = [1,3,6], seed=19)
    >>> rs1.show_epochs(2)
    Epoch : 0, batches used : [[5], [4, 3, 9], [3, 2, 8, 0, 4, 3]] 
    Epoch : 1, batches used : [[0], [7, 9, 7], [3, 5, 8, 3, 4, 9]]                

    """

    
    def __new__(cls, num_indices, num_batches=None, prob = None, replace = True, shuffle=True, batch_size = None, seed = None):
        
        cls.num_batches = num_batches

        if cls.num_batches is None:
            cls.num_batches = num_indices
        
        if cls.num_batches == num_indices:
            return super(RandomSampling, cls).__new__(RandomIndex)
        else:
            return super(RandomSampling, cls).__new__(RandomBatch)
    
    def __init__(self, num_indices, num_batches=None, prob = None, replace = True, shuffle=True, batch_size = None, seed = None ):
        
        self.num_indices = num_indices
        self.num_batches = num_batches
        self.batch_size = batch_size

        if self.num_batches is None:
            self.num_batches = num_indices

        self.prob = prob
        self.replace = replace
        self.shuffle = shuffle
        # store indices
        self.indices_used = []
        self.index = 0
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.list_of_indices =  self.rng.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=self.replace) 

        ########################################
        # check if equal batches
        # if equal batch_size then batch_size = self.num_indices//self.num_batches
        # if not equal batch_size then default batch_size = (self.num_indices//self.num_batches)+1
        # another uneven splitting is created if batch_size is a list.
        self.equal_size_batches = self.num_indices%self.num_batches==0    

        if self.equal_size_batches:
            logging.warning("Batch size is (constant) self.num_indices//self.num_batches ") 
            self.batch_size = self.num_indices//self.num_batches
        else:
            if self.batch_size is None:
                logging.warning("Batch size is not constant. Default maximum batch_size = num_indices//num_batches + 1 ")                
                self.batch_size = (self.num_indices//self.num_batches)+1 
            else:
                self.batch_size = batch_size # list
                if isinstance(self.batch_size, list):
                    if len(self.batch_size)!=self.num_batches:                        
                        raise ValueError(" The list of sizes for the uneven batch_size case should be equal to num_batches. ")
                    if sum(self.batch_size)<self.num_indices:
                        raise ValueError(" The sum of each element in batch_size should be greater or equal to num_indices. ")
                else:
                    raise ValueError(" With uneven batch sizes, a list is required for the size of each batch_size. ")
                    
        # create partition for batch_size>1
        if isinstance(self.batch_size, list):
            iterator_list = iter(self.list_of_indices)
            self.partition_list = [list(islice(iterator_list, 0, i)) for i in self.batch_size] # list of lists
        else: 
            self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)] # list of lists
                            
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
        return RandomSampling(num_indices,  num_batches=num_batches, prob=None, replace = replace, shuffle = True, seed = seed)
    
    @staticmethod
    def non_uniform(num_indices, prob, num_batches = None, replace = True, seed=None):
        if prob is None:
            raise ValueError("List of probabilities is required for the non-uniform sampling. For uniform sampling use RandomSampling.uniform. ")
        return RandomSampling(num_indices, num_batches=num_batches, replace = replace, prob=prob, shuffle = True, seed = seed) 
     
    @staticmethod
    def single_shuffle(num_indices, num_batches = None, prob=None, seed=None):
        return RandomSampling(num_indices, num_batches = num_batches, prop=prob, replace = False, shuffle = False, seed = seed)
     
    @staticmethod
    def random_shuffle(num_indices, num_batches = None, prob=None, seed=None):
        return RandomSampling(num_indices, num_batches = num_batches, prob = prob, replace = False, shuffle = True, seed = seed)              
            

class RandomBatch(RandomSampling):
            
    def __next__(self):
        
        tmp_list = list(self.partition_list[self.index])
        self.indices_used.append(tmp_list)         
        self.index+=1
        
        if self.index==len(self.partition_list):
            self.index=0            

            if self.shuffle is True:     
         
                self.list_of_indices = self.rng.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=self.replace)        
                if isinstance(self.batch_size, list):
                    iterator_list = iter(self.list_of_indices)
                    self.partition_list = [list(islice(iterator_list, 0, i)) for i in self.batch_size] # list of lists
                else: 
                    self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)] # list of lists
                                  
                                               
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

     
        
   