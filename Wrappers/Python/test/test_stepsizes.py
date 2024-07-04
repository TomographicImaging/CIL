from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA
from cil.optimisation.functions import LeastSquares, IndicatorBox
from cil.framework import ImageGeometry, VectorGeometry, VectorData
from cil.optimisation.operators import IdentityOperator, MatrixOperator

from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner, ConstantStepSize, ArmijoStepSizeRule, BarzilaiBorweinStepSizeRule
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

    def test_constant(self):
        test_stepsize = ConstantStepSize(0.3)
        self.assertEqual(test_stepsize.step_size, 0.3)

    def test_armijo_init(self):
        test_stepsize = ArmijoStepSizeRule(alpha=1e3, beta=0.4, max_iterations=40)
        self.assertEqual(test_stepsize.alpha_orig, 1e3)
        self.assertEqual(test_stepsize.beta, 0.4)
        self.assertEqual(test_stepsize.max_iterations, 40)

        test_stepsize = ArmijoStepSizeRule()
        self.assertEqual(test_stepsize.alpha_orig, 1e6)
        self.assertEqual(test_stepsize.beta, 0.5)
        self.assertEqual(test_stepsize.max_iterations, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))

    def test_armijo_calculation(self):
        test_stepsize = ArmijoStepSizeRule(alpha=8, beta=0.5, max_iterations=100)
        ig = VectorGeometry(2)
        data = ig.allocate('random')
        data.fill(np.array([3.5, 3.5]))
        A = MatrixOperator(np.diag([1., 1.]))
        f = LeastSquares(A, data)
        alg = GD(initial=ig.allocate(0), objective_function=f,
                 max_iteration=100, update_objective_interval=1, step_size=test_stepsize)
        alg.gradient_update = ig.allocate(-1)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 4)

        alg.gradient_update = ig.allocate(-.5)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 8)

        alg.gradient_update = ig.allocate(-2)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 2)

    def test_bb_init(self):
        pass #TODO:
    
    def test_bb_converge(self):
        n = 10
        m = 5

        A = np.random.uniform(0, 1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1 *
             np.random.randn(m)).astype('float32')

        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig=Aop.domain
        initial = ig.allocate()
        f = LeastSquares(Aop, b=bop, c=0.5)
        
        ss_rule=ArmijoStepSizeRule(max_iterations=40)
        alg_true = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg_true .run(300, verbose=0)
        
        
        
        
        
        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'short')
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg.run(40, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), alg_true.x.as_array(), decimal=4)
        

        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'long')
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg.run(40, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), alg_true.x.as_array(), decimal=4)
        

        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'alternate')
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)

        alg.run(40, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), alg_true.x.as_array(), decimal=4)
        
        
