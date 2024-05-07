from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA
from cil.optimisation.functions import  LeastSquares, IndicatorBox
from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.operators import IdentityOperator, MatrixOperator

from cil.optimisation.utilities import  Sensitivity, AdaptiveSensitivity, Adam, AdaGrad, Preconditioner, ConstantStepSize, ArmijoStepSize
import numpy as np 

from testclass import CCPiTestClass
from unittest.mock import MagicMock


class TestPreconditioners(CCPiTestClass):
    
    def test_step_sizes_called(self):
        
        
        ig = ImageGeometry(2,1,4)
        data = ig.allocate(1)
        A= IdentityOperator(ig)
        step_size_test = MagicMock(return_value=.1)
        f = LeastSquares(A=A, b=data, c=0.5)
        alg = GD(initial=ig.allocate('random', seed=10), objective_function=f, step_size_rule=step_size_test,
               max_iteration=100, update_objective_interval=1)
        
        alg.run(5)
        
        self.assertEqual(len(step_size_test.mock_calls),5)
        
        step_size_test = MagicMock(return_value=.1)
        alg = ISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), step_size_rule=step_size_test ,
               max_iteration=100, update_objective_interval=1)
        alg.run(5)
        self.assertEqual(len(step_size_test.mock_calls),5)
    
        step_size_test = MagicMock(return_value=.1)
        alg = FISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), step_size_rule=step_size_test ,
               max_iteration=100, update_objective_interval=1)
        alg.run(5)
        self.assertEqual(len(step_size_test.mock_calls),5)
        
        
    def test_constant(self):
        test_precon = ConstantStepSize(0.3)
        self.assertEqual(test_precon.step_size, 0.3)
        
    
    def test_armijo_init(self):
        test_precon = ArmijoStepSize(alpha = 1e3, beta =0.4, kmax = 40)
        self.assertEqual(test_precon.alpha_orig, 1e3)
        self.assertEqual(test_precon.beta, 0.4)
        self.assertEqual(test_precon.kmax, 40)
        
        test_precon = ArmijoStepSize()
        self.assertEqual(test_precon.alpha_orig, 1e6)
        self.assertEqual(test_precon.beta, 0.5)
        self.assertEqual(test_precon.kmax, np.ceil (2 * np.log10(1e6) / np.log10(2)))
        
    def test_armijo_calculation(self):
        test_precon = ArmijoStepSize(alpha = 8, beta=0.5, kmax=100)
        ig = VectorGeometry(2)
        data = ig.allocate('random')
        data.fill(np.array([3.5,3.5]))
        A= MatrixOperator( np.diag([1.,1.]))
        f=LeastSquares(A, data)
        alg = GD(initial=ig.allocate(0), objective_function=f, 
               max_iteration=100, update_objective_interval=1, step_size_rule=test_precon)  
        alg.gradient_update= ig.allocate(-1)
        step_size=test_precon(alg)
        self.assertAlmostEqual(step_size, 4)
        
        alg.gradient_update= ig.allocate(-.5)
        step_size=test_precon(alg)
        self.assertAlmostEqual(step_size, 8)
        
        alg.gradient_update= ig.allocate(-2)
        step_size=test_precon(alg)
        self.assertAlmostEqual(step_size, 2)


       