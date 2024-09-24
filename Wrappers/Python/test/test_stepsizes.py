from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA
from cil.optimisation.functions import LeastSquares, IndicatorBox
from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.operators import IdentityOperator, MatrixOperator

from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner, ConstantStepSize, ArmijoStepSizeRule
import numpy as np

from testclass import CCPiTestClass
from unittest.mock import MagicMock


class TestStepSizes(CCPiTestClass):

    def test_step_sizes_called(self):

        ig = ImageGeometry(2, 1, 4)
        data = ig.allocate(1)
        A = IdentityOperator(ig)
        step_size_test = ConstantStepSize(3)
        step_size_test.get_step_size = MagicMock(return_value=.1)
        f = LeastSquares(A=A, b=data, c=0.5)
        alg = GD(initial=ig.allocate('random', seed=10), objective_function=f, step_size=step_size_test,
                 max_iteration=100, update_objective_interval=1)

        alg.run(5)

        self.assertEqual(len(step_size_test.get_step_size.mock_calls), 5)

        step_size_test = ConstantStepSize(3)
        step_size_test.get_step_size = MagicMock(return_value=.1)
        alg = ISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), step_size=step_size_test,
                   max_iteration=100, update_objective_interval=1)
        alg.run(5)
        self.assertEqual(len(step_size_test.get_step_size.mock_calls), 5)

        step_size_test = ConstantStepSize(3)
        step_size_test.get_step_size = MagicMock(return_value=.1)
        alg = FISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), step_size=step_size_test,
                    max_iteration=100, update_objective_interval=1)
        alg.run(5)
        self.assertEqual(len(step_size_test.get_step_size.mock_calls), 5)

class TestStepSizeConstant(CCPiTestClass):
    def test_constant(self):
        test_stepsize = ConstantStepSize(0.3)
        self.assertEqual(test_stepsize.step_size, 0.3)

class TestStepSizeArmijo(CCPiTestClass):
    
    def setUp(self):
        self.ig = VectorGeometry(2)
        self.data = self.ig.allocate('random')
        self.data.fill(np.array([3.5, 3.5]))
        self.A = MatrixOperator(np.diag([1., 1.]))
        self.f = LeastSquares(self.A, self.data)

        
    def test_armijo_init(self):
        test_stepsize = ArmijoStepSizeRule(alpha=1e3, beta=0.4, max_iterations=40, warmstart=False)
        self.assertFalse(test_stepsize.warmstart)        
        self.assertEqual(test_stepsize.alpha_orig, 1e3)
        self.assertEqual(test_stepsize.beta, 0.4)
        self.assertEqual(test_stepsize.max_iterations, 40)

        test_stepsize = ArmijoStepSizeRule()
        self.assertTrue(test_stepsize.warmstart)
        self.assertEqual(test_stepsize.alpha_orig, 1e6)
        self.assertEqual(test_stepsize.beta, 0.5)
        self.assertEqual(test_stepsize.max_iterations, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))

    def test_armijo_calculation(self):
        test_stepsize = ArmijoStepSizeRule(alpha=8, beta=0.5, max_iterations=100)

        alg = GD(initial=self.ig.allocate(0), objective_function=self.f,
                 max_iteration=100, update_objective_interval=1, step_size=test_stepsize)
        alg.gradient_update = self.ig.allocate(-1)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 4)

        alg.gradient_update = ig.allocate(-.5)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 8)

        alg.gradient_update = ig.allocate(-2)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 2)

    def test_warmstart_true(self):
        
        rule = ArmijoStepSizeRule(warmstart=True, alpha=5000)
        self.assertTrue(rule.warmstart)
        self.assertTrue(rule.alpha_orig == 5000)
        alg = GD(initial=self.ig.allocate(0), objective_function=self.f,
                 max_iteration=100, update_objective_interval=1, step_size=rule)
        alg.update()
        self.assertFalse(rule.alpha_orig == 5000)
        self.assertTrue(rule.alpha_orig == rule.alpha)  

    def test_warmstart_false(self):
        rule = ArmijoStepSizeRule(warmstart=False,  alpha=5000)
        self.assertFalse(rule.warmstart)
        self.assertTrue(rule.alpha_orig == 5000)
        alg = GD(initial=self.ig.allocate(0), objective_function=self.f,
                 max_iteration=100, update_objective_interval=1, step_size=rule)
        alg.update()
        self.assertTrue(rule.alpha_orig == 5000)
        self.assertFalse(rule.alpha_orig == rule.alpha)  
