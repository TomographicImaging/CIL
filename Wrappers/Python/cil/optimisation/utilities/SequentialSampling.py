import numpy as np
import logging
from itertools import islice

class SequentialSampling:

    r"""SequentialSampling.

    SequentialSampling is an iterator that generates ordered indices or batches from a list of integers of length = `num_indices`, e.g., :code:`np.arange(num_indices)`.

        
    Parameters
    ----------

    num_indices : int
            A list of length :code:`num_indices`, e.g., :code:`np.arange(num_indices)` is generated.
    num_batches : int, optional
            The number of batches to split the generated list. Default = num_indices.
            A warning is raised when :code:`num_batches` is not a divisor of :code:`num_indices`.
    step_size : int, optional 
            The step size for the next selected item in the list. Default = `num_batches`
            A warning is raised when :code:`step_size` is not a divisor of :code:`num_indices`.
    batch_size : list, optional
            The batch size for each `num_batches`. Default = None .
            If the `num_batches` is a divisor of `num_indices` then `batch_size` is `self.num_indices//self.num_batches`, by default.
            If the `num_batches` is not a divisor of `num_indices`, then `batch_size` is `self.num_indices//self.num_batches + 1`, by default. Otherwise, a list can be passed for each size of the `num_batches`.    
        
    See also
    --------
    :class:`.RandomSampling`

    Examples
    --------

    Create a SequentialSampling iterator for a list of 10 indices, np.arange(10):
    >>> sq1 = SequentialSampling(10)
    >>> sq1.show_epochs(1)
    Epoch : 0, indices used : [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] 

    Create a SequentialSampling iterator for a list of 10 indices, np.arange(10) with 2 batches:
    >>> sq1 = SequentialSampling(10, num_batches = 2)
    >>> sq1.show_epochs(1)
    Epoch : 0, indices used : [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]    

    Create a SequentialSampling iterator for a list of 10 indices, np.arange(10) with 2 batches and step_size = 5:
    >>> sq1 = SequentialSampling(10, num_batches = 2, step_size = 5)
    >>> sq1.show_epochs(1)
    Epoch : 0, indices used : [[0, 5, 1, 6, 2], [7, 3, 8, 4, 9]] 

    Create a SequentialSampling iterator for a list of 10 indices, np.arange(10) with 3 batches:
    >>> sq1 = SequentialSampling(10, num_batches = 3)
    >>> sq1.show_epochs(1)
    Epoch : 0, indices used : [[0, 3, 6, 9], [1, 4, 7, 2], [5, 8]]   

    Create a SequentialSampling iterator for a list of 10 indices, np.arange(10) with 3 batches of batch_size = [1,3,6]:
    >>> sq1 = SequentialSampling(10, num_batches = 3, batch_size = [1,3,6])
    >>> sq1.show_epochs(1)
    Epoch : 0, indices used : [[0], [3, 6, 9], [1, 4, 7, 2, 5, 8]]                


    """    

    def __new__(cls, num_indices, num_batches=None, step_size = None, batch_size = None):
        
        cls.num_batches = num_batches

        if cls.num_batches is None:
            cls.num_batches = num_indices
        
        if cls.num_batches == num_indices:
            return super(SequentialSampling, cls).__new__(SequentialIndex)
        else:
            return super(SequentialSampling, cls).__new__(SequentialBatch)

                
    def __init__(self, num_indices, num_batches = None, step_size = None, batch_size = None):
        
        self.num_indices = num_indices
        self.num_batches = num_batches
        self.step_size = step_size
        self.batch_size = batch_size
        # store indices
        self.indices_used = []
        self.index = 0 
        

        # default values
        if self.num_batches is None:
            self.num_batches = self.num_indices

        # default values
        if self.step_size is None:
            self.step_size = self.num_batches
        

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

        if self.num_indices%self.step_size!=0:
            logging.warning("Step size is not constant")
        
        # create new list
        self.list_of_indices = []
        tmp_list_indices = list(np.arange(num_indices))

        for i in range(0,self.step_size):
            self.list_of_indices += tmp_list_indices[i:self.num_indices:self.step_size] # list
            
        # create partition for batch_size>1
        if isinstance(self.batch_size, list):
            iterator_list = iter(self.list_of_indices)
            self.partition_list = [list(islice(iterator_list, 0, i)) for i in self.batch_size] # list of lists
        else: 
            self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)] # list of lists

class SequentialIndex(SequentialSampling):   
    
    def __next__(self):

        index_num = self.list_of_indices[self.index]
        self.index+=1   

        if self.index == self.num_indices:
            self.index = 0                

        self.indices_used.append(index_num)

        return index_num  

class SequentialBatch(SequentialIndex):
            
    def __next__(self):
        
        tmp_list = list(self.partition_list[self.index])
        self.indices_used.append(tmp_list)         
        self.index+=1
        
        if self.index==len(self.partition_list):
            self.index=0            
                                               
        return tmp_list        

        
                
                                         
    # def __next__(self):
        
    #     if isinstance(self.batch_size, (list,int)):
    #         tmp = self.partition_list[self.index]
    #     else:
    #         tmp = self.list_of_indices[self.index]
    #     self.indices_used.append(tmp)  
    #     self.index += 1
        
    #     if self.index == self.num_batches:
    #         self.index=0 
                    
    #     return tmp  

    def show_epochs(self, epochs):
        
        total_its = epochs * self.num_indices
        for _ in range(total_its):
            next(self)

        k = 0
        for i in range(epochs):
            print(" Epoch : {}, indices used : {} ".format(i, self.indices_used[k:k+self.num_batches]))    
            k+=self.num_batches 
        print("")  



    
