import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SubsetSumFunction, SGDFunction
from cil.framework import VectorData
import numpy as np                  
                  

initialise_tests()

class TestSGDFunction(unittest.TestCase):
                    
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
        self.f_SGD = SGDFunction(self.fi_cil, sampling="sequential")

        precond = ig.allocate(1.0)
        self.f_SGD_precond = SGDFunction(self.fi_cil, sampling="sequential", precond=precond)

    def test_gradient(self):

        out1 = self.x_cil.geometry.allocate()
        out2 = self.x_cil.geometry.allocate()

        # No preconditioning
        self.f_SGD.gradient(self.x_cil, out=out1)

        self.f_SGD[self.f_SGD.subset_num].gradient(self.x_cil, out=out2)
        np.testing.assert_allclose(out1.array, out2.array, atol=1e-3) 

        out3 = self.x_cil.geometry.allocate()
        out4 = self.x_cil.geometry.allocate()

        # With preconditioning
        self.f_SGD_precond.gradient(self.x_cil, out=out1)

        self.f_SGD_precond[self.f_SGD.subset_num].gradient(self.x_cil, out=out2)
        out2.multiply(self.f_SGD_precond.precond, out=out2)
        np.testing.assert_allclose(out1.array, out2.array, atol=1e-3)         





              




                      












           