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
                  update_objective_interval=1)

        alg.run(5)

        self.assertEqual(len(step_size_test.get_step_size.mock_calls), 5)

        step_size_test = ConstantStepSize(3)
        step_size_test.get_step_size = MagicMock(return_value=.1)
        alg = ISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), step_size=step_size_test,
                    update_objective_interval=1)
        alg.run(5)
        self.assertEqual(len(step_size_test.get_step_size.mock_calls), 5)

        step_size_test = ConstantStepSize(3)
        step_size_test.get_step_size = MagicMock(return_value=.1)
        alg = FISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), step_size=step_size_test,
                     update_objective_interval=1)
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
        test_stepsize = ArmijoStepSizeRule(alpha=8, beta=0.5, max_iterations=100, warmstart=False)

        alg = GD(initial=self.ig.allocate(0), objective_function=self.f,
                  update_objective_interval=1, step_size=test_stepsize)
        alg.gradient_update = self.ig.allocate(-1)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 4)

        alg.gradient_update = self.ig.allocate(-.5)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 8)

        alg.gradient_update = self.ig.allocate(-2)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 2)

    def test_armijo_ISTA_and_FISTA(self):
        test_stepsize = ArmijoStepSizeRule(alpha=8, beta=0.5, max_iterations=100, warmstart=False)

        alg = ISTA(initial=self.ig.allocate(0), f=self.f, g=IndicatorBox(lower=0),
                     update_objective_interval=1, step_size=test_stepsize)
        alg.gradient_update = self.ig.allocate(-1)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 4)

        alg.gradient_update = self.ig.allocate(-.5)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 8)

        alg.gradient_update = self.ig.allocate(-2)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 2)

        alg = FISTA(initial=self.ig.allocate(0), f=self.f, g=IndicatorBox(lower=0),
                     update_objective_interval=1, step_size=test_stepsize)
        alg.gradient_update = self.ig.allocate(-1)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 4)

        alg.gradient_update = self.ig.allocate(-.5)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 8)

        alg.gradient_update = self.ig.allocate(-2)
        step_size = test_stepsize.get_step_size(alg)
        self.assertAlmostEqual(step_size, 2)

    def test_warmstart_true(self):
        
        rule = ArmijoStepSizeRule(warmstart=True, alpha=5000)
        self.assertTrue(rule.warmstart)
        self.assertTrue(rule.alpha_orig == 5000)
        alg = GD(initial=self.ig.allocate(0), objective_function=self.f,
                  update_objective_interval=1, step_size=rule)
        alg.update()
        self.assertFalse(rule.alpha == 5000)

    def test_warmstart_false(self):
        rule = ArmijoStepSizeRule(warmstart=False,  alpha=5000)
        self.assertFalse(rule.warmstart)
        self.assertTrue(rule.alpha_orig == 5000)
        alg = GD(initial=self.ig.allocate(0), objective_function=self.f,
                  update_objective_interval=1, step_size=rule)
        alg.update()
        self.assertTrue(rule.alpha_orig == 5000)
        self.assertFalse(rule.alpha_orig == rule.alpha)  

class TestStepSizeBB(CCPiTestClass):
    def test_bb(self):
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
        
        ss_rule=BarzilaiBorweinStepSizeRule(2 )
        self.assertEqual(ss_rule.mode, 'short')
        self.assertEqual(ss_rule.initial, 2)
        self.assertEqual(ss_rule.adaptive, True)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        
        #Check the right errors are raised for incorrect parameters 
        
        with self.assertRaises(TypeError):
            ss_rule=BarzilaiBorweinStepSizeRule(2,'short',-4, )
        with self.assertRaises(TypeError):
            ss_rule=BarzilaiBorweinStepSizeRule(2,'long', 'banana', )
        with self.assertRaises(ValueError):
            ss_rule=BarzilaiBorweinStepSizeRule(2, 'banana',3 )
            
        
        #Check stabilisation parameter unchanged if fixed 
        ss_rule=BarzilaiBorweinStepSizeRule(2, 'long',3 )
        self.assertEqual(ss_rule.mode, 'long')
        self.assertFalse(ss_rule.adaptive)
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param,3)
        alg.run(2)
        self.assertEqual(ss_rule.stabilisation_param,3)
        
        #Check infinity can be passed 
        ss_rule=BarzilaiBorweinStepSizeRule(2, 'short',"off" )
        self.assertEqual(ss_rule.mode, 'short')
        self.assertFalse(ss_rule.adaptive)
        self.assertEqual(ss_rule.stabilisation_param,np.inf)
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg.run(2)
           
        n = 5
        m = 5

        A = np.eye(5).astype('float32')
        b = (np.array([.5,.5,.5,.5,.5])).astype('float32')

        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig=Aop.domain
        initial = ig.allocate(0)
        f = LeastSquares(Aop, b=bop, c=0.5)
        ss_rule=BarzilaiBorweinStepSizeRule(0.22, 'long',np.inf )
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        self.assertFalse(ss_rule.is_short)
        #Check the initial step size was used
        alg.run(1)
        self.assertNumpyArrayAlmostEqual( np.array([.11,.11,.11,.11,.11]), alg.x.as_array() )
        self.assertFalse(ss_rule.is_short)
        #check long 
        alg.run(1)
        x_change= np.array([.11,.11,.11,.11,.11])-np.array([0,0,0,0,0])
        grad_change = -np.array([.39,.39,.39,.39,.39])+np.array([.5,.5,.5,.5,.5])
        step= x_change.dot(x_change)/x_change.dot(grad_change)
        self.assertNumpyArrayAlmostEqual( np.array([.11,.11,.11,.11,.11])+step*np.array([.39,.39,.39,.39,.39]), alg.x.as_array() )
        self.assertFalse(ss_rule.is_short)
        
        ss_rule=BarzilaiBorweinStepSizeRule(0.22, 'short',np.inf )
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        self.assertTrue(ss_rule.is_short)
        #Check the initial step size was used
        alg.run(1)
        self.assertNumpyArrayAlmostEqual( np.array([.11,.11,.11,.11,.11]), alg.x.as_array() )
        self.assertTrue(ss_rule.is_short)
        #check short
        alg.run(1)
        x_change= np.array([.11,.11,.11,.11,.11])-np.array([0,0,0,0,0])
        grad_change = -np.array([.39,.39,.39,.39,.39])+np.array([.5,.5,.5,.5,.5])
        step= x_change.dot(grad_change)/grad_change.dot(grad_change)
        self.assertNumpyArrayAlmostEqual( np.array([.11,.11,.11,.11,.11])+step*np.array([.39,.39,.39,.39,.39]), alg.x.as_array() )
        self.assertTrue(ss_rule.is_short)
        
        #check stop iteration 
        ss_rule=BarzilaiBorweinStepSizeRule(1, 'long',np.inf )
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg.run(500)
        self.assertEqual(alg.iteration, 1)
        
        #check adaptive
        ss_rule=BarzilaiBorweinStepSizeRule(0.001, 'long',"auto")
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg.run(2)
        self.assertNotEqual(ss_rule.stabilisation_param, np.inf)
        
        #check stops being adaptive 
        
        ss_rule=BarzilaiBorweinStepSizeRule(0.0000001, 'long',"auto" )
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg.run(4)
        self.assertNotEqual(ss_rule.stabilisation_param, np.inf)
        a=ss_rule.stabilisation_param
        alg.run(1)
        self.assertEqual(ss_rule.stabilisation_param, a)
        
        #Test alternating
        ss_rule=BarzilaiBorweinStepSizeRule(0.0000001, 'alternate',"auto" )
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        self.assertFalse(ss_rule.is_short)
        alg.run(2)
        self.assertTrue(ss_rule.is_short)
        alg.run(1)
        self.assertFalse(ss_rule.is_short)
        alg.run(1)
        self.assertTrue(ss_rule.is_short)

        
        
        
    def test_bb_converge(self):
        n = 10
        m = 5
        np.random.seed(4)
        A = np.random.uniform(0, 1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1 *
             np.random.randn(m)).astype('float32')

        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig=Aop.domain
        initial = ig.allocate()
        f = LeastSquares(Aop, b=bop, c=2)
        
        ss_rule=ArmijoStepSizeRule(max_iterations=40, warmstart=False)
        alg_true = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg_true .run(300, verbose=0)
        
       
        
        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'short')
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg.run(80, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), alg_true.x.as_array(), decimal=3)
        

        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'long')
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)
        alg.run(80, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), alg_true.x.as_array(), decimal=3)
        

        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'alternate')
        alg = GD(initial=initial, objective_function=f, step_size=ss_rule)

        alg.run(80, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), alg_true.x.as_array(), decimal=3)
        
        

