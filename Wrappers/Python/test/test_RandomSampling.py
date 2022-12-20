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

        list_generated_seed_19 = [0, 7, 9, 7, 3, 5, 8, 3, 4, 9]

        # check static method
        sm_rs_uniform = RandomSampling.uniform(self.len_list, seed=19)
        
        tmp1 = []
        for _ in range(10):
            next(self.rs_uniform)
            next(sm_rs_uniform)

        self.assertListEqual(self.rs_uniform.indices_used, list_generated_seed_19)
        self.assertListEqual(self.rs_uniform.indices_used, sm_rs_uniform.indices_used)


    def test_random_sampling_uniform_without_replacement(self):

        list_generated_seed_19 = [7, 8, 9, 2, 1, 4, 6, 5, 0, 3]
        
        tmp1 = []
        for _ in range(10):
            next(self.rs_uniform_without_replacement)

        self.assertListEqual(self.rs_uniform_without_replacement.indices_used, list_generated_seed_19)        

    def test_random_sampling_uniform_with_replacement_batch(self):

        batches_generated_seed_19 = [[5, 4], [3, 9], [3, 2], [8, 0], [4, 3], [5, 4], [3, 9], [3, 2], [8, 0], [4, 3]]
        
        tmp1 = []
        for _ in range(10):
            next(self.rs_uniform_batch)

        for i in range(int(self.len_list/self.batch_size)):
            self.assertListEqual(self.rs_uniform_batch.indices_used[i], batches_generated_seed_19[i]) 

    def test_random_sampling_uniform_with_replacement_batch(self):

        unequal_batches_generated_seed_19 = [[5, 4, 3], [9, 3, 2], [8, 0, 4], [3]]
        
        print(int(self.len_list/3)+1)
        for _ in range(int(self.len_list/self.batch_size)+1):
            next(self.rs_uniform_unequal_batch)

        for i in range(int(self.len_list/3)+1):
            self.assertListEqual(self.rs_uniform_unequal_batch.indices_used[i], unequal_batches_generated_seed_19[i])                

    def test_random_sampling_uniform_without_replacement_batch(self):

        batches_generated_seed_19 = [[7, 8], [9, 2], [1, 4], [6, 5], [0, 3]]
        
        print(int(self.len_list/self.batch_size))
        for _ in range(int(self.len_list/self.batch_size)):
            next(self.rs_uniform_without_replacement_batch)

        for i in range(int(self.len_list/self.batch_size)):
            self.assertListEqual(self.rs_uniform_without_replacement_batch.indices_used[i], batches_generated_seed_19[i])                





    

    # def test_random_sampling_uniform_with_replacement_batch(self):
        
    #     tmp1 = []
    #     for _ in range(10):
    #         next(self.rs_batch)
    #         next(self.rs_with_batch_class)

    #     self.assertListEqual(self.rs_batch.indices_used[0], self.rs_with_batch_class.indices_used[0])        
           

    #     # tmp2 = []
    #     # for _ in range(10):
    #     #     tmp2.append(next(self.rs2)) 

    #     # print(tmp1)
    #     # print(tmp2)        

    #     # epochs = 1
    #     # list_of_selected_ind = []
    #     # for i in range(self.len_list*epochs):
    #     #     next(rs1)
    #     #     next(rs2)

    #     # np.random.seed(19)
    #     # tmp = np.random.choice(self.len_list,size=self.len_list,replace=True)
    #     # for i in range(self.len_list*epochs):

    #     #     list_of_selected_ind.append(tmp[i])

        
    #     # print(self.rs1.indices_used)
    #     # print(self.rs2.indices_used)
    #     # print(list_of_selected_ind)        
                
                
    
    # # def test_random_sampling_uniform_without_replacement(self):
        


    # #     epochs = 1
    # #     list_of_selected_ind = []
    # #     for i in range(self.len_list*epochs):
    # #         next(rs1)
    # #         next(rs2)

    # #     np.random.seed(19)
    # #     tmp = np.random.choice(self.len_list,size=self.len_list,replace=False)
    # #     for i in range(self.len_list*epochs):

    # #         list_of_selected_ind.append(tmp[i])

    # #     self.assertListEqual(rs1.indices_used, rs2.indices_used)
    # #     self.assertListEqual(list_of_selected_ind, rs2.indices_used)
    # #     self.assertListEqual(rs1.indices_used, list_of_selected_ind)
        

    