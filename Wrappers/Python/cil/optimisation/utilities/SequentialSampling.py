import numpy as np
import logging

class SequentialSampling:
                
    def __init__(self, num_indices, num_batches = None, step_size = None):
        
        self.num_indices = num_indices
        self.num_batches = num_batches
        self.step_size = step_size

        # default values
        if self.num_batches is None:
            self.num_batches = self.num_indices

        # default values
        if self.step_size is None:
            self.step_size = self.num_batches
        
        # store indices
        self.indices_used = []
        
        # check if equal batches
        self.equal_size_batches = self.num_indices%self.num_batches==0    
        if self.equal_size_batches:
            self.batch_size = self.num_indices//self.num_batches
        else:
            logging.warning("Batch size is not constant")
            self.batch_size = (self.num_indices//self.num_batches)+1 

        if self.num_indices%self.step_size!=0:
            logging.warning("Step size is not constant")
            
        
        # create new list
        self.list_of_indices = []
        tmp_list_indices = list(np.arange(num_indices))
        for i in range(0,self.step_size):
            self.list_of_indices += tmp_list_indices[i:self.num_indices:self.step_size] 
            
        # create partition for batch_size>1
        if self.batch_size>1: 
            self.partition_list = [self.list_of_indices[i:i + self.batch_size] for i in range(0, self.num_indices, self.batch_size)]             
            
        self.index = 0
                
                                         
    def __next__(self):
        
        if self.batch_size>1:
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


if __name__ == "__main__":

    a = 10
    # sq1 = SequentialSampling(a, num_batches=10, step_size = 3)
    sq1 = SequentialSampling(a, num_batches=1, step_size=3)
    print(sq1.num_batches, sq1.batch_size, sq1.step_size)
    # for i in range(10):
    #     next(sq1)
    # print(sq1.indices_used)
    sq1.show_epochs(1)
    