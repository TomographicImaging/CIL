import unittest
from utils import initialise_tests
from cil.optimisation.utilities import SequentialSampling
import numpy as np                  
                  
initialise_tests()


class TestSequentialSampling(unittest.TestCase):
    
    def setUp(self):
            
        self.num_indices = 10     
 
    def test_seq_sampling_unit_step_size(self):

        sq1 = SequentialSampling(self.num_indices)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, list(np.arange(self.num_indices)))

    
    def test_seq_sampling_num_batches(self):
        
        sq1 = SequentialSampling(self.num_indices, num_batches=2)
        tmp_list = [[0, 2, 4, 6, 8],[1,3,5,7,9]]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list)  

        sq1 = SequentialSampling(self.num_indices, 5)
        tmp_list = [[0, 5],[1,6],[2,7],[3,8],[4,9]]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list)  

        sq1 = SequentialSampling(self.num_indices, num_batches=3)
        tmp_list = [[0, 3, 6, 9], [1, 4, 7, 2], [5, 8]]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list) 

        sq1 = SequentialSampling(self.num_indices, num_batches=10)
        tmp_list = [0,1,2,3,4,5,6,7,8,9]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list)         

    def test_seq_sampling_num_batches_step_size(self):
        
        sq1 = SequentialSampling(self.num_indices, num_batches=2, step_size = 3)
        tmp_list = [[0, 3, 6, 9, 1],[4,7,2,5,8]]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list) 

        sq1 = SequentialSampling(self.num_indices, num_batches=5, step_size = 3)
        tmp_list = [[0, 3], [6, 9], [1,4],[7,2],[5,8]]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list) 

        sq1 = SequentialSampling(self.num_indices, num_batches=2, step_size = 5)
        tmp_list = [[0, 5, 1, 6, 2], [7, 3, 8, 4, 9]]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list)    

        sq1 = SequentialSampling(self.num_indices, num_batches=10, step_size = 5)
        tmp_list = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, tmp_list)                                                        

