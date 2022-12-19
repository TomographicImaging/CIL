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

class RandomSampling():
    
    def __new__(cls, num_indices, batch_size=1, prob = None, replace = True, shuffle=False, seed = None):
        
        cls.batch_size = batch_size
        
        if cls.batch_size == 1:
            return super(RandomSampling, cls).__new__(RandomIndex)
        else:
            return super(RandomSampling, cls).__new__(RandomBatch)
    
    def __init__(self, num_indices, batch_size=1, prob = None, replace = True, shuffle=False, seed = None ):
        
        self.num_indices = num_indices
        self.batch_size = batch_size
        self.equal_size_batches = self.num_indices%self.batch_size==0        
        if self.equal_size_batches:
            self.num_batches = self.num_indices//self.batch_size
        else:
            self.num_batches = (self.num_indices//self.batch_size)+1        
        self.prob = prob
        self.replace = replace
        self.shuffle = shuffle
        self.indices_used = []
        self.index = 0
        np.random.seed(seed)
                            
        if self.replace is False: 
            self.list_of_indices = np.random.choice(num_indices, size=self.num_indices, p=prob, replace=False)                                        
        else:
            if shuffle is True and self.batch_size==1:
                raise ValueError("Shuffle is used only with replace=False")   
            self.list_of_indices = np.random.choice(num_indices, size=self.num_indices, p=prob, replace=True)         
            
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
    def uniform(num_indices, batch_size = 1, seed=None):
        return RandomSampling(num_indices,  batch_size=batch_size, prob=None, replace = True, shuffle = False, seed = seed)
    
    @staticmethod
    def non_uniform(num_indices, prob, batch_size = 1, seed=None):
        return RandomSampling(num_indices, batch_size=batch_size, replace = True, prob=prob, shuffle = False, seed = seed) 
    
    @staticmethod    
    def uniform_no_replacement(num_indices, batch_size = 1, shuffle=True, seed=None):
        return RandomSampling(num_indices, batch_size=batch_size, prob=None, replace = False, shuffle = shuffle, seed = seed) 
    
    @staticmethod    
    def non_uniform_no_replacement(num_indices, prob, batch_size = 1, shuffle=False, seed=None):
        return RandomSampling(num_indices, batch_size=batch_size, prob=prob, replace = False, shuffle = shuffle, seed = seed)     
        
    @staticmethod
    def single_shuffle(num_indices, batch_size = 1, seed=None):
        return RandomSampling(num_indices, batch_size = batch_size, replace = False, shuffle = False, seed = seed)
    
    @staticmethod
    def random_shuffle(num_indices, batch_size = 1, seed=None):
        return RandomSampling(num_indices, batch_size = batch_size, replace = False, shuffle = True, seed = seed)              
            

class RandomBatch(RandomSampling):
            
    def __next__(self):
        
        tmp = list(self.partition_list[self.index])
        self.indices_used.append(tmp)         
        self.index+=1
        
        if self.index==len(self.partition_list):
            self.index=0            
            
            if self.shuffle is True:
                self.list_of_indices = np.random.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=self.replace)        
                self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)]
                                               
        return tmp
    
class RandomIndex(RandomSampling):   
    
    def __next__(self):
        
        if self.replace is False:

            index_num = self.list_of_indices[self.index]
            self.index+=1                

            if self.index == self.num_indices:
                self.index = 0                
                if self.shuffle is True:                    
                    self.list_of_indices = np.random.choice(self.num_indices, size=self.num_indices, p=self.prob, replace=False)                                                                                         
        else:

            index_num = np.random.choice(self.num_indices, size=1, p=self.prob, replace=True).item()

        self.indices_used.append(index_num)

        return index_num  
        
    