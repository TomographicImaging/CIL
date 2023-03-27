import unittest
from utils import initialise_tests
from cil.optimisation.utilities import SequentialSampling
import numpy as np                  
                  
initialise_tests()


class TestSequentialSampling(unittest.TestCase):
    
    def setUp(self):
            
        self.num_indices = 10     
 
    def test_seq_sampling_diff_num_batches(self):

        # test sequential sampling with different num_batches

        sq1 = SequentialSampling(self.num_indices, self.num_indices)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, list(np.arange(self.num_indices)))

        sq1 = SequentialSampling(self.num_indices, 2)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,2,4,6,8],[1,3,5,7,9]])  

        sq1 = SequentialSampling(self.num_indices, 5)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,5],[1,6],[2,7],[3,8],[4,9]])                

    def test_seq_sampling_diff_num_batches_diff_step_size(self):

        sq1 = SequentialSampling(self.num_indices, 5, step_size = 2)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,2],[4,6],[8,1],[3,5],[7,9]])  

        sq1 = SequentialSampling(self.num_indices, 5, step_size = 3)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,3],[6,9],[1,4],[7,2],[5,8]])          

    def test_seq_sampling_ueven_batch_size(self):

        # default batch_size in the ueven case
        sq1 = SequentialSampling(self.num_indices, num_batches = 3)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,3,6,9],[1,4,7,2],[5,8]] )   

    def test_seq_sampling_list_batch_size(self): 

        # batch_size in the ueven case as a list
        sq1 = SequentialSampling(self.num_indices, num_batches = 4, batch_size = [2,3,3,2])         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,4],[8,1,5],[9,2,6],[3,7]] )                 

        # batch_size in the ueven case as a list
        sq1 = SequentialSampling(self.num_indices, num_batches = 3, batch_size = [2,3,5])         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,3],[6,9,1],[4,7,2,5,8]] )    

        # batch_size in the ueven case as a list, with step_size
        sq1 = SequentialSampling(self.num_indices, num_batches = 3, step_size=1, batch_size = [2,3,5])         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[0,1],[2,3,4],[5,6,7,8,9]] )  

    def test_seq_sampling_list_batch_size_error(self):     

        # when the sum of each batch_size in the list does not sum to num_indices
        with self.assertRaises(ValueError) as err:            
            sq1 = SequentialSampling(self.num_indices, num_batches = 4, batch_size = [2,3,3,1])         

        # when the len of the list batch_size < num_batches
        with self.assertRaises(ValueError) as err:            
            sq1 = SequentialSampling(self.num_indices, num_batches = 4, batch_size = [2,3,5])                 
        
                 


    
                                                       
