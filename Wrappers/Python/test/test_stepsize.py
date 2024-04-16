import unittest
from utils import initialise_tests
from cil.optimisation.algorithms import GD, FISTA
from cil.framework import VectorData, ImageGeometry
from cil.optimisation.utilities import ArmijoStepSize, ConstantStepSize
from cil.optimisation.functions import Rosenbrock, ZeroFunction, LeastSquares
from cil.optimisation.operators import IdentityOperator
from scipy.optimize import minimize, rosen
import numpy as np
from testclass import CCPiTestClass
initialise_tests()


class TestStepSize(CCPiTestClass):

    def setUp(self):

        x0_1 = 1.1
        x0_2 = 1.1
        # x0_1 = 0.5
        # x0_2 = 0.5
        self.x0 = np.array([x0_1, x0_2])

        self.initial = VectorData(np.array(self.x0))
        method = 'Nelder-Mead'# or "BFGS"
        # self.scipy_opt_low = minimize(rosen, self.x0, method=method, tol=1e-3, options={"maxiter":50})
        self.scipy_opt_high = minimize(rosen, self.x0, method=method, tol=1e-2) # (1., 1.)
        self.f =  Rosenbrock(alpha=1, beta=100) #fixed (alpha=1, beta=100) same to Scipy, min at (alpha,alpha^2)
                
        
    def tearDown(self):
        pass   
    
    def test_gd_init(self):
        gd = GD(initial = self.initial, objective_function = self.f, step_size = 0.002)
        self.assertEqual(gd.step_size_rule.step_size, 0.002)
        
        gd = GD(initial = self.initial, objective_function = self.f)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e6)
        self.assertEqual(gd.step_size_rule.beta, 0.5)
        self.assertEqual(gd.step_size_rule.kmax, np.ceil (2 * np.log10(1e6) / np.log10(2)))
       
        gd = GD(initial = self.initial, objective_function = self.f, alpha=1e2, beta=0.25)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e2)
        self.assertEqual(gd.step_size_rule.beta, 0.25)
        self.assertEqual(gd.step_size_rule.kmax, np.ceil (2 * np.log10(1e2) / np.log10(2)))
       
        with self.assertRaises(TypeError):
            gd = GD(initial = self.initial,objective_function = self.f, step_size=0.1, step_size_rule=ConstantStepSize(0.5))
        
        
    def test_constant_step_size_init(self):
        rule=ConstantStepSize(0.4)
        self.assertEqual(rule.step_size, 0.4)
        gd = GD(initial = self.initial, objective_function = self.f, step_size_rule=rule)
        self.assertEqual(gd.step_size_rule.step_size, 0.4)
        
    
    def test_gd_fixed_step_size_rosen(self):

        gd = GD(initial = self.initial, objective_function = self.f, step_size = 0.002,
                    max_iteration = 3000, 
                    update_objective_interval =  500)
        gd.run(verbose=0)    
        np.testing.assert_allclose(gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)
        
    def test_armijo_step_size_init(self):
        rule=ArmijoStepSize()
        self.assertEqual(rule.alpha_orig, 1e6)
        self.assertEqual(rule.beta, 0.5)
        self.assertEqual(rule.kmax, np.ceil (2 * np.log10(1e6) / np.log10(2)))
       
        gd = GD(initial = self.initial, objective_function = self.f, step_size_rule=rule)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e6)
        self.assertEqual(gd.step_size_rule.beta, 0.5)
        self.assertEqual(gd.step_size_rule.kmax, np.ceil (2 * np.log10(1e6) / np.log10(2)))
       
        rule=ArmijoStepSize(5e5,0.2,5)
        self.assertEqual(rule.alpha_orig, 5e5)
        self.assertEqual(rule.beta, 0.2)
        self.assertEqual(rule.kmax, 5)
       
        gd = GD(initial = self.initial, objective_function = self.f, step_size_rule=rule)
        self.assertEqual(gd.step_size_rule.alpha_orig, 5e5)
        self.assertEqual(gd.step_size_rule.beta, 0.2)
        self.assertEqual(gd.step_size_rule.kmax, 5)
       
       

        

    def test_GDArmijo(self):
        ig = ImageGeometry(12,13,14)
        initial = ig.allocate()
        # b = initial.copy()
        # fill with random numbers
        # b.fill(np.random.random(initial.shape))
        b = ig.allocate('random')
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)

        alg = GD(initial=initial, objective_function=norm2sq)
        alg.max_iteration = 100
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg = GD(initial=initial, objective_function=norm2sq,
                 max_iteration=20, update_objective_interval=2)
        #alg.max_iteration = 20
        self.assertTrue(alg.max_iteration == 20)
        self.assertTrue(alg.update_objective_interval==2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        
        

    def test_gd_armijo_rosen(self):        
        armj = ArmijoStepSize(alpha=50, kmax=150)
        gd = GD(initial = self.initial, objective_function = self.f, step_size_rule = armj,
                    max_iteration = 2500, 
                    update_objective_interval =  500)
        gd.run(verbose=0)  
        np.testing.assert_allclose(gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2) 

                    