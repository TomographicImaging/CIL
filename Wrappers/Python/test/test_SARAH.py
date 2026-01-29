import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.algorithms import SARAH
from cil.optimisation.functions import LeastSquares, L2NormSquared, ZeroFunction, ApproximateGradientSumFunction
from cil.framework import VectorData
import numpy as np            
from cil.optimisation.utilities import RandomSampling     
                  
initialise_tests()


from utils import has_cvxpy

if has_cvxpy:
    import cvxpy


class TestSARAH(unittest.TestCase):

    def setUp(self):
        
        np.random.seed(10)
        n = 10
        m = 200
        A = np.random.uniform(0,1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 

        self.n_subsets = 5

        Ai = np.vsplit(A, self.n_subsets) 
        bi = [b[i:i+int(m/self.n_subsets)] for i in range(0, m, int(m/self.n_subsets))]     

        self.fi_cil = []
        for i in range(self.n_subsets):   
            self.Ai_cil = MatrixOperator(Ai[i])
            self.bi_cil = VectorData(bi[i])
            self.fi_cil.append(LeastSquares(self.Ai_cil, self.bi_cil, c = 0.5))
            
        self.F = LeastSquares(self.Aop, b=self.bop, c = 0.5) 
        self.G = ZeroFunction()

        self.ig = self.Aop.domain

        self.sampling = RandomSampling.uniform(self.n_subsets)
        self.fi = ApproximateGradientSumFunction(functions=self.fi_cil, selection=self.sampling, data_passes=[0.])           

        self.initial = self.ig.allocate()   


    def test_signature(self):

        # required args
        with np.testing.assert_raises(TypeError):
            sarah = SARAH(initial = self.initial, f = self.fi)            

        with np.testing.assert_raises(TypeError):
            sarah = SARAH(initial = self.initial, f = self.fi)            

        with np.testing.assert_raises(TypeError):
            sarah = SARAH(initial = self.initial, g = self.G) 

        with np.testing.assert_raises(ValueError):
            sarah = SARAH(initial = self.initial, f = L2NormSquared(), g = self.G)             

        tmp_step_size = 10
        tmp_update_frequency = 3
        sarah = SARAH(initial = self.initial, g = self.G, f = self.fi, step_size=tmp_step_size, update_frequency=tmp_update_frequency) 
        np.testing.assert_equal(sarah.step_size.initial, tmp_step_size)
        np.testing.assert_equal(sarah.update_frequency, tmp_update_frequency)

        self.assertTrue( id(sarah.x)!=id(sarah.initial))   
        self.assertTrue( id(sarah.x_old)!=id(sarah.initial))

    def test_data_passes(self):

        sampling = RandomSampling.uniform(self.n_subsets)
        fi = ApproximateGradientSumFunction(functions=self.fi_cil, 
                                            selection=sampling, 
                                            data_passes=[0.])           

        sarah = SARAH(f=fi, g=self.G, update_objective_interval=1, 
                      initial=self.initial, max_iteration=6)
        sarah.run(verbose=0)

        correct_passes = [1., 1+1/self.n_subsets, 
                             1.+2./self.n_subsets, 1+3./self.n_subsets, 1+4/self.n_subsets, 2+4/self.n_subsets]
        np.testing.assert_equal(correct_passes, sarah.f.data_passes)        


    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        epochs = 300        
        initial = self.ig.allocate()
        sarah = SARAH(f=self.fi, g=self.G, update_objective_interval=200, initial=initial, max_iteration=epochs*self.n_subsets)
        sarah.run(verbose=0)

        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=False, solver=cvxpy.SCS, eps=1e-4)
        np.testing.assert_allclose(p.value, sarah.objective[-1], rtol=5e-3)
        np.testing.assert_allclose(u_cvxpy.value, sarah.solution.array, rtol=5e-3)    

    def test_update(self):
        
        initial = self.ig.allocate()
        sarah = SARAH(f=self.fi, g=self.G, update_objective_interval=1, 
                      initial=initial, max_iteration=2)
        # this should use indices 0 and 1
        sarah.run(verbose=0)

        x = initial.copy()
        x_old = initial.copy()

        step_size = sarah.step_size.initial
        F_new = ApproximateGradientSumFunction(functions=self.fi_cil, selection=self.sampling, data_passes=[0.])

        gradient_estimator = F_new.full_gradient(x)        
        x.sapyb(1., gradient_estimator, -step_size, out = x)
        x = self.G.proximal(x, step_size) # not sure if this makes sense

        function_num = sarah.f.function_num
        stoch_grad_at_iterate = F_new.functions[function_num].gradient(x)
        stochastic_grad_difference = stoch_grad_at_iterate.sapyb(1., F_new.functions[function_num].gradient(x_old), -1.)
        gradient_estimator =  stochastic_grad_difference.sapyb(self.n_subsets, gradient_estimator, 1.)
        x_old = x.copy()
        x.sapyb(1., gradient_estimator, -step_size, out = x)
        x = self.G.proximal(x, step_size) # not sure if this makes sense

        np.testing.assert_allclose(sarah.solution.array, x.array, atol=1e-2)      

        res1 = sarah.objective[-1]
        res2 = F_new(x) + self.G(x)
        np.testing.assert_allclose(res1, res2, rtol=1e-5)         