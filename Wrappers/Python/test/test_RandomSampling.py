import unittest
from utils import initialise_tests
from cil.optimisation.utilities import RandomSampling, RandomIndex, RandomBatch
import numpy as np                  
                  
initialise_tests()

class TestRandomSampling(unittest.TestCase):
                    
    def setUp(self):
            
        self.len_list = 10     
        self.seed = 19  
        self.batch_size = 2
        self.epochs = 3

        # uniform with replacement
        self.rs_uniform = RandomSampling(self.len_list, seed=self.seed) 

        # uniform without replacement
        self.rs_uniform_without_replacement = RandomSampling(self.len_list, replace=False, seed=self.seed) 

        # uniform with replacement batch_size>1
        self.rs_uniform_batch = RandomSampling(self.len_list, batch_size=self.batch_size, seed=self.seed) 

        # uniform without replacement batch_size>1
        self.rs_uniform_without_replacement_batch = RandomSampling(self.len_list, batch_size=self.batch_size, replace=False, seed=self.seed) 

        # uniform without replacement unequal batch_size
        self.rs_uniform_unequal_batch = RandomSampling(self.len_list, batch_size=3, seed=self.seed) 


    def test_random_sampling_uniform_with_replacement(self):

        list_generated_seed_19 = [5, 4, 3, 9, 3, 2, 8, 0, 4, 3]

        # check static method
        sm_rs_uniform = RandomSampling.uniform(self.len_list, seed=self.seed)
        
        for _ in range(10):
            next(self.rs_uniform)
            next(sm_rs_uniform)

        self.assertListEqual(self.rs_uniform.indices_used, list_generated_seed_19)
        self.assertListEqual(self.rs_uniform.indices_used, sm_rs_uniform.indices_used)


    def test_random_sampling_uniform_without_replacement(self):

        list_generated_seed_19 = [7, 8, 9, 2, 1, 4, 6, 5, 0, 3]

        # check static method
        sm_rs_uniform_without_replacement = RandomSampling.uniform_no_replacement(self.len_list, seed=self.seed)        
        
        for _ in range(10):
            next(self.rs_uniform_without_replacement)
            next(sm_rs_uniform_without_replacement)

        self.assertListEqual(self.rs_uniform_without_replacement.indices_used, list_generated_seed_19)        
        self.assertListEqual(self.rs_uniform_without_replacement.indices_used, sm_rs_uniform_without_replacement.indices_used)        

    def test_random_sampling_uniform_with_replacement_batch(self):

        batches_generated_seed_19 = [[5, 4], [3, 9], [3, 2], [8, 0], [4, 3], [5, 4], [3, 9], [3, 2], [8, 0], [4, 3]]
        
        for _ in range(10):
            next(self.rs_uniform_batch)

        for i in range(int(self.len_list/self.batch_size)):
            self.assertListEqual(self.rs_uniform_batch.indices_used[i], batches_generated_seed_19[i]) 

    def test_random_sampling_uniform_with_replacement_batch(self):

        unequal_batches_generated_seed_19 = [[5, 4, 3], [9, 3, 2], [8, 0, 4], [3]]
        
        
        for _ in range(int(self.len_list/self.batch_size)+1):
            next(self.rs_uniform_unequal_batch)

        for i in range(int(self.len_list/3)+1):
            self.assertListEqual(self.rs_uniform_unequal_batch.indices_used[i], unequal_batches_generated_seed_19[i])                

    def test_random_sampling_uniform_without_replacement_batch(self):

        batches_generated_seed_19 = [[7, 8], [9, 2], [1, 4], [6, 5], [0, 3]]
        
        
        for _ in range(int(self.len_list/self.batch_size)):
            next(self.rs_uniform_without_replacement_batch)

        for i in range(int(self.len_list/self.batch_size)):
            self.assertListEqual(self.rs_uniform_without_replacement_batch.indices_used[i], batches_generated_seed_19[i])                





    