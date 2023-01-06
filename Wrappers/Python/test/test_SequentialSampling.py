import unittest
from utils import initialise_tests
from cil.optimisation.utilities import SequentialSampling
import numpy as np                  
                  
initialise_tests()


class TestSequentialSampling(unittest.TestCase):
    
    def setUp(self):
            
        self.num_indices = 10     
        self.batch_size = 5
        self.epochs = 3
 

        # # step_size = 3, batch_size = 2
        # self.seqsam_ss_3 = SequentialSampling(self.num_indices, step_size=3, batch_size=2)                 

        # # step_size = 3, batch_size = 4 (not divisible)
        # self.seqsam_ss_3 = SequentialSampling(self.num_indices, step_size=3, batch_size=4)                 


    def test_seq_sampling_unit_step_size(self):

        # step_size = 1
        seqsam = SequentialSampling(self.num_indices) 
        
        for _ in range(seqsam.num_batches):
            next(seqsam)

        self.assertListEqual(seqsam.indices_used, list(np.arange(self.num_indices)))

    def test_seq_sampling_no_unit_step_size(self):

        seqsam = SequentialSampling(self.num_indices, step_size=3) 
        tmp_list = [0,3,6,9,1,4,7,2,5,8]

        for _ in range(seqsam.num_batches):
            next(seqsam)

        self.assertListEqual(seqsam.indices_used, tmp_list)

    def test_seq_sampling_no_unit_step_size_divisible_batch_size(self):

        seqsam = SequentialSampling(self.num_indices, step_size=3, batch_size=self.batch_size) 
        tmp_list = [[0,3,6,9,1],[4,7,2,5,8]]

        for _ in range(seqsam.num_batches):
            next(seqsam)

        self.assertListEqual(seqsam.indices_used, tmp_list) 

    def test_seq_sampling_no_unit_step_size_not_divisible_batch_size(self):

        seqsam = SequentialSampling(self.num_indices, step_size=3, batch_size=4) 
        tmp_list = [[0,3,6,9],[1,4,7,2],[5,8]]

        for _ in range(seqsam.num_batches):
            next(seqsam)

        self.assertListEqual(seqsam.indices_used, tmp_list)                

