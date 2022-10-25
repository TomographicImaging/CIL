import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SubsetSumFunction
from cil.framework import VectorData
import numpy as np                  
                  

initialise_tests()

class TestSubsetSumFunction(unittest.TestCase):
                    
    def setUp(self):   

        np.random.seed(10)
        n = 50
        m = 500
        self.n_subsets = 10

        Anp = np.random.uniform(0,1, (m, n)).astype('float32')
        xnp = np.random.uniform(0,1, (n,)).astype('float32')
        bnp = Anp.dot(xnp)

        Ai = np.vsplit(Anp, self.n_subsets) 
        bi = [bnp[i:i+n] for i in range(0, m, n)]     

        self.Aop = MatrixOperator(Anp)
        self.bop = VectorData(bnp) 
        ig = self.Aop.domain        
        self.x_cil = ig.allocate('random')

        self.fi_cil = []
        for i in range(self.n_subsets):   
            Ai_cil = MatrixOperator(Ai[i])
            bi_cil = VectorData(bi[i])
            self.fi_cil.append(LeastSquares(Ai_cil, bi_cil, c=1.0))

        self.f = LeastSquares(self.Aop, b=self.bop, c=1.0)
        self.f_subset_sum_function = SubsetSumFunction(self.fi_cil) # default with replacement
        self.f_subset_sum_function_sequential = SubsetSumFunction(self.fi_cil, sampling="sequential")              

        self.f_subset_sum_function_random_suffle = SubsetSumFunction(self.fi_cil, sampling="random", replacement=False, suffle="random") 
        self.f_subset_sum_function_single_suffle = SubsetSumFunction(self.fi_cil, sampling="random", replacement=False, suffle="single")         
 
    def test_call_method(self):
        
        res1 = self.f(self.x_cil)
        res2 = self.f_subset_sum_function(self.x_cil)
        np.testing.assert_allclose(res1, res2)

    def test_full_gradient(self):
        
        res1 = self.f.gradient(self.x_cil)
        res2 = self.f_subset_sum_function.full_gradient(self.x_cil)
        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3)        

    def test_sampling_sequential(self):

        # check sequential selection
        for i in range(self.n_subsets):
            self.f_subset_sum_function_sequential.next_subset()
            np.testing.assert_equal(self.f_subset_sum_function_sequential.subset_num, i)

    def test_sampling_random_with_replacement(self):

        # check random selection with replacement
        epochs = 3
        choices = []
        for i in range(epochs):
            for j in range(self.n_subsets):
                self.f_subset_sum_function.next_subset()
                choices.append(self.f_subset_sum_function.subset_num)
        self.assertTrue( choices == self.f_subset_sum_function.subsets_used)          

    def test_sampling_random_without_replacement_random_suffle(self):

        # check random selection with no replacement
        epochs = 3
        choices = []
        for i in range(epochs):
            for j in range(self.n_subsets):
                self.f_subset_sum_function_random_suffle.next_subset()
                choices.append(self.f_subset_sum_function_random_suffle.subset_num)
        self.assertTrue( choices == self.f_subset_sum_function_random_suffle.subsets_used)  

    def test_sampling_random_without_replacement_single_suffle(self):

        # check random selection with no replacement
        epochs = 3
        choices = []
        for i in range(epochs):
            for j in range(self.n_subsets):
                self.f_subset_sum_function_single_suffle.next_subset()
                choices.append(self.f_subset_sum_function_single_suffle.subset_num)
        self.assertTrue( choices == self.f_subset_sum_function_single_suffle.subsets_used)         





                      












           