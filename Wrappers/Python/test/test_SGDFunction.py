import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SubsetSumFunction, SGDFunction
from cil.optimisation.algorithms import GD
from cil.framework import VectorData
import numpy as np                  
                  
initialise_tests()

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestSGDFunction(unittest.TestCase):
                    

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
        self.precond = lambda i, x: self.ig.allocate(1.)
        self.F_SGD = SGDFunction(self.fi_cil, replacement = True, precond = self.precond)           

        self.initial = self.ig.allocate()  

    def test_gradient(self):

        out1 = self.ig.allocate()
        out2 = self.ig.allocate()

        x = self.ig.allocate('random')

        # No preconditioning
        self.F_SGD.gradient(x, out=out1)

        self.F_SGD[self.F_SGD.subset_num].gradient(x, out=out2)
        np.testing.assert_allclose(out1.array, out2.array, atol=1e-3) 

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        
        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize( 0.5/self.n_subsets * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 

        step_size = 1./self.F.L # Theoretical step size = 1./(16*self.F_SAG.L) 
        epochs = 100
        sgd = GD(initial = self.initial, objective_function = self.F_SGD, step_size = step_size,
                    max_iteration = epochs * self.n_subsets, 
                    update_objective_interval = epochs * self.n_subsets)
        sgd.run(verbose=0)    

        np.testing.assert_allclose(p.value, sgd.objective[-1], atol=1e-1)

        np.testing.assert_allclose(u_cvxpy.value, sgd.solution.array, atol=1e-1)             





              




                      












           