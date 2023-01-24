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

    """    
                
    def __init__(self, num_indices, num_batches = None, step_size = None, batch_size = None):
        
        self.num_indices = num_indices
        self.num_batches = num_batches
        self.step_size = step_size
        self.batch_size = batch_size

        # default values
        if self.num_batches is None:
            self.num_batches = self.num_indices

        # default values
        if self.step_size is None:
            self.step_size = self.num_batches
        
        # store indices
        self.indices_used = []
        
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
        
        self.index = 0
                
                                         
    def __next__(self):
        
        if isinstance(self.batch_size, (list,int)):
            tmp = self.partition_list[self.index]
        else:
            tmp = self.list_of_indices[self.index]
        self.indices_used.append(tmp)  
        self.index += 1
        
        if self.index == self.num_batches:
            self.index=0 
                    
        return tmp  

    def show_epochs(self, epochs):
        
        total_its = epochs * self.num_indices
        for _ in range(total_its):
            next(self)

        k = 0
        for i in range(epochs):
            print(" Epoch : {}, indices used : {} ".format(i, self.indices_used[k:k+self.num_batches]))    
            k+=self.num_batches 
        print("")  

                          


    

