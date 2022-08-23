import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SubsetSumFunction, SAGFunction
from cil.optimisation.algorithms import GD
from cil.framework import VectorData
import numpy as np                  
                  
initialise_tests()

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestSAGFunction(unittest.TestCase):
                    
    def setUp(self):
        
        np.random.seed(10)
        n = 50
        m = 500

        A = np.random.uniform(0,1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 
        
        Anp = np.random.uniform(0,1, (m, n)).astype('float32')
        xnp = np.random.uniform(0,1, (n,)).astype('float32')
        bnp = Anp.dot(xnp)

        # split data, operators, functions
        self.n_subsets = 10

        Ai = np.vsplit(A, self.n_subsets) 
        bi = [b[i:i+n] for i in range(0, m, n)]     

        self.fi_cil = []
        for i in range(self.n_subsets):   
            self.Ai_cil = MatrixOperator(Ai[i])
            self.bi_cil = VectorData(bi[i])
            self.fi_cil.append(LeastSquares(self.Ai_cil, self.bi_cil, c = 0.5))
            
        self.F = LeastSquares(self.Aop, b=self.bop, c = 0.5/self.n_subsets) 
        self.ig = self.Aop.domain
        self.precond = lambda i, x: 3./self.ig.allocate(2.5)
        self.F_SAG = SAGFunction(self.fi_cil, replacement = True, precond = self.precond)           

        self.initial = self.ig.allocate()          

    def test_gradient(self):

        out1 = self.ig.allocate()
        out2 = self.ig.allocate()

        x = self.ig.allocate('random', seed = 10)

        # use the gradient method for one iteration
        self.F_SAG.gradient(x, out=out1)
        
        # run all steps of the SAG gradient method, one iteration
        tmp_sag = SAGFunction(self.fi_cil, replacement = True)     

        # x is passed but the gradient initial point = None, hence initial is 0
        tmp_sag.initialise_memory(self.ig.allocate()) 
        tmp_sag.next_subset()
        tmp_sag.functions[tmp_sag.subset_num].gradient(x, out=tmp_sag.tmp1)
        tmp_sag.tmp1.sapyb(1., tmp_sag.subset_gradients[tmp_sag.subset_num], -1., out=tmp_sag.tmp2)
        tmp_sag.tmp2.sapyb(1./tmp_sag.num_subsets, tmp_sag.full_gradient, 1.,  out=out2)
        out2 *= self.precond(tmp_sag.tmp2.subset_num, 3./self.ig.allocate(2.5))

        # update subset_gradient in the subset_num
        # update full gradient
        tmp_sag.subset_gradients[tmp_sag.subset_num].fill(tmp_sag.tmp1)
        tmp_sag.full_gradient.sapyb(1., tmp_sag.tmp2, 1./tmp_sag.num_subsets, out=tmp_sag.full_gradient)

        np.testing.assert_allclose(tmp_sag.subset_gradients[tmp_sag.subset_num].array, 
                                   self.n_subsets * out1.array, atol=1e-3)

        np.testing.assert_allclose(tmp_sag.full_gradient.array, 
                                   out1.array, atol=1e-3)                                     

        np.testing.assert_allclose(out1.array, out2.array, atol=1e-3)                                     

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        
        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize( 0.5/self.n_subsets * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 

        step_size = 1./self.F.L # in theory is 1./16*self.L, in practive 1./self.L works better
        epochs = 100
        sag = GD(initial = self.initial, objective_function = self.F_SAG, step_size = step_size,
                    max_iteration = epochs * self.n_subsets, 
                    update_objective_interval = epochs * self.n_subsets)
        sag.run(verbose=0)    

        np.testing.assert_allclose(p.value, sag.objective[-1], atol=1e-1)



        


              




                      










