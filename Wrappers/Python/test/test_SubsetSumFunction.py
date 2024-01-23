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

        self.f = (1/self.n_subsets) * LeastSquares(self.Aop, b=self.bop, c=1.0)
        self.f_subset_sum_function = SubsetSumFunction(self.fi_cil)
        self.f_subset_sum_function_no_replacement = SubsetSumFunction(self.fi_cil, sampling="random", replacement=False) 
        self.f_subset_sum_function_sequential = SubsetSumFunction(self.fi_cil, sampling="sequential")              

    def test_call_method(self):
        
        res1 = self.f(self.x_cil)
        res2 = self.f_subset_sum_function(self.x_cil)
        np.testing.assert_allclose(res1, res2)

    def test_full_gradient(self):
        
        res1 = self.f.gradient(self.x_cil)
        res2 = self.f_subset_sum_function._full_gradient(self.x_cil)
        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3)        

    def test_sampling(self):

        # check sequential selection
        for i in range(self.n_subsets):
            self.f_subset_sum_function_sequential.next_subset()
            np.testing.assert_equal(self.f_subset_sum_function_sequential.subset_num, i)

        # check random selection with no replacement
        epochs = 2
        choices = [[],[]]
        for i in range(epochs):
            for j in range(self.n_subsets):
                self.f_subset_sum_function_no_replacement.next_subset()
                choices[i].append(self.f_subset_sum_function_no_replacement.subset_num)
        self.assertTrue( len(set(choices[0]))== len(set(choices[1])))

        # check random selection with replacement
        epochs = 2
        choices = [[],[]]
        for i in range(epochs):
            for j in range(self.n_subsets):
                self.f_subset_sum_function.next_subset()
                choices[i].append(self.f_subset_sum_function.subset_num)
        self.assertTrue( len(set(choices[0]))!= len(set(choices[1])))  

        with self.assertRaises(NotImplementedError):
            f=SubsetSumFunction(self.fi_cil, sampling="not implemented")
            f.next_subset()

              




                      












           