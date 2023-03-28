import unittest
from utils import initialise_tests
from cil.optimisation.utilities import RandomSampling, RandomIndex, RandomBatch
import numpy as np                  
                  
initialise_tests()

class TestRandomSampling(unittest.TestCase):
                    
    def setUp(self):
            
        self.num_indices = 10     
        self.seed = 19  


    def test_random_sampling_diff_num_batches(self):

        # test random sampling with different num_batches, with default options

        sq1 = RandomSampling(self.num_indices, self.num_indices, seed=self.seed)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [5, 4, 3, 9, 3, 2, 8, 0, 4, 3])

        sq1 = RandomSampling(self.num_indices, 2, seed=self.seed)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[5, 4, 3, 9, 3], [2, 8, 0, 4, 3]])        

        sq1 = RandomSampling(self.num_indices, 5, seed=self.seed)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[5, 4], [3, 9], [3, 2], [8, 0], [4, 3]])        

    def test_random_sampling_ueven_batch_size(self):

        # test random sampling with ueven batch_size

        sq1 = RandomSampling(self.num_indices, 3, seed=self.seed)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[5, 4, 3, 9], [3, 2, 8, 0], [4, 3]])        


    def test_random_sampling_list_batch_size(self): 

        # batch_size in the ueven case as a list
        sq1 = RandomSampling(self.num_indices, num_batches = 4, batch_size = [2,3,3,2], seed=self.seed)         
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[5, 4], [3, 9, 3], [2, 8, 0], [4, 3]])                  

        # batch_size in the ueven case as a list
        sq1 = RandomSampling(self.num_indices, num_batches = 3, batch_size = [2,3,5], seed=self.seed)                 
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[5, 4], [3, 9, 3], [2, 8, 0, 4, 3]])   
       
    def test_random_sampling_list_batch_size_error(self):     

        # when the sum of each batch_size in the list does not sum to num_indices
        with self.assertRaises(ValueError) as err:            
            sq1 = RandomSampling(self.num_indices, num_batches = 4, batch_size = [2,3,3,1])         

        # when the len of the list batch_size < num_batches
        with self.assertRaises(ValueError) as err:            
            sq1 = RandomSampling(self.num_indices, num_batches = 4, batch_size = [2,3,5])         
       

    def test_random_sampling_without_replacement(self):

        sq1 = RandomSampling(self.num_indices, num_batches = 2, replace = False, seed=self.seed)                 
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[7, 8, 9, 2, 1], [4, 6, 5, 0, 3]])   
                

    def test_random_sampling_without_replacement_with_shuffle(self):

        sq1 = RandomSampling(self.num_indices, num_batches = 2, replace = False, seed=self.seed)                 
        for _ in range(2*sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[7, 8, 9, 2, 1], [4, 6, 5, 0, 3], 
                                                [7, 0, 2, 1, 9], [3, 4, 5, 8, 6]])   
           

    def test_random_sampling_without_replacement_without_shuffle(self):

        sq1 = RandomSampling(self.num_indices, num_batches = 2, replace = False, shuffle = False, seed=self.seed)                 
        for _ in range(2*sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[7, 8, 9, 2, 1], [4, 6, 5, 0, 3], 
                                                [7, 8, 9, 2, 1], [4, 6, 5, 0, 3]])   


    def test_random_sampling_with_prob(self):

        prob = [1./self.num_indices]*self.num_indices #uniform
        sq1 = RandomSampling(self.num_indices, num_batches = 5, prob=prob, seed=self.seed)                 
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[4, 9], [2, 0], [3, 7], [7, 5], [3, 9]])                                                   
           
        np.random.seed(self.seed)
        tmp_p = [np.random.random() for i in range(self.num_indices)]
        prob = [i/sum(tmp_p) for i in tmp_p] #non uniform
        sq1 = RandomSampling(self.num_indices, num_batches = 5, prob=prob, seed=self.seed)                 
        for _ in range(sq1.num_batches):
            next(sq1)
        self.assertListEqual(sq1.indices_used, [[6, 9], [4, 1], [4, 8], [8, 7], [4, 9]])                                                   
                      