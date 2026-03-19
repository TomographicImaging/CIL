from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA, PDHG
from cil.optimisation.functions import LeastSquares, IndicatorBox, ZeroFunction, L2NormSquared
from cil.framework import ImageGeometry, VectorGeometry, VectorData
from cil.optimisation.operators import IdentityOperator, MatrixOperator, LinearOperator

from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner, ConstantStepSize, ArmijoStepSizeRule, BarzilaiBorweinStepSizeRule, PDHGStronglyConvexUpdate, PDHGConstantStepSize, PDHGAdaptiveStepSize2013, PDHGAdaptiveStepSize2015
import numpy as np

from testclass import CCPiTestClass
from unittest.mock import MagicMock

import warnings

class TestStepSizes(CCPiTestClass):

    def test_step_sizes_called(self):

        ig = ImageGeometry(2, 1, 4)
        data = ig.allocate(1)
        A = IdentityOperator(ig)
        step_size_test = ConstantStepSize(3)
        step_size_test.get_step_size = MagicMock(return_value=.1)
        f = LeastSquares(A=A, b=data, c=0.5)
        alg = GD(initial=ig.allocate('random', seed=10), f=f, step_size=step_size_test,
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
        self.data = self.ig.allocate('random', seed=3)
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

        alg = GD(initial=self.ig.allocate(0), f=self.f,
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
        alg = GD(initial=self.ig.allocate(0), f=self.f,
                  update_objective_interval=1, step_size=rule)
        alg.update()
        self.assertFalse(rule.alpha == 5000)

    def test_warmstart_false(self):
        rule = ArmijoStepSizeRule(warmstart=False,  alpha=5000)
        self.assertFalse(rule.warmstart)
        self.assertTrue(rule.alpha_orig == 5000)
        alg = GD(initial=self.ig.allocate(0), f=self.f,
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
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param,3)
        alg.run(2)
        self.assertEqual(ss_rule.stabilisation_param,3)
        
        #Check infinity can be passed 
        ss_rule=BarzilaiBorweinStepSizeRule(2, 'short',"off" )
        self.assertEqual(ss_rule.mode, 'short')
        self.assertFalse(ss_rule.adaptive)
        self.assertEqual(ss_rule.stabilisation_param,np.inf)
        alg = GD(initial=initial, f=f, step_size=ss_rule)
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
        alg = GD(initial=initial, f=f, step_size=ss_rule)
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
        alg = GD(initial=initial, f=f, step_size=ss_rule)
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
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        alg.run(500)
        self.assertEqual(alg.iteration, 1)
        
        #check adaptive
        ss_rule=BarzilaiBorweinStepSizeRule(0.001, 'long',"auto")
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg.run(2)
        self.assertNotEqual(ss_rule.stabilisation_param, np.inf)
        
        #check stops being adaptive 
        
        ss_rule=BarzilaiBorweinStepSizeRule(0.0000001, 'long',"auto" )
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg.run(4)
        self.assertNotEqual(ss_rule.stabilisation_param, np.inf)
        a=ss_rule.stabilisation_param
        alg.run(1)
        self.assertEqual(ss_rule.stabilisation_param, a)
        
        #Test alternating
        ss_rule=BarzilaiBorweinStepSizeRule(0.0000001, 'alternate',"auto" )
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertFalse(ss_rule.is_short)
        alg.run(2)
        self.assertTrue(ss_rule.is_short)
        alg.run(1)
        self.assertFalse(ss_rule.is_short)
        alg.run(1)
        self.assertTrue(ss_rule.is_short)

class TestPDHGConstantStepSize(CCPiTestClass):
    
    def test_deprecated_parameters(self): #TODO: remove when deprecated parameters are removed from PDHG
        
        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        sigma=0.5)
        self.assertEqual(pdhg.sigma, 0.5)
        self.assertEqual(pdhg.step_size_rule.sigma, 0.5)
        self.assertEqual(pdhg.step_size_rule.tau, 1./(0.5 * pdhg.operator.norm()**2))
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))
        
        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        tau=0.5)
        self.assertEqual(pdhg.tau, 0.5)
        self.assertEqual(pdhg.step_size_rule.tau, 0.5)
        self.assertEqual(pdhg.step_size_rule.sigma, 1./(0.5 * pdhg.operator.norm()**2))
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))
        
        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        sigma=0.5, tau=0.5)
        self.assertEqual(pdhg.sigma, 0.5)
        self.assertEqual(pdhg.tau, 0.5)
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))
        
        
        
    
    def test_PDHG_step_sizes(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = 3*IdentityOperator(ig)

        # check if sigma, tau are None
        pdhg = PDHG(f=f, g=g, operator=operator)
        self.assertAlmostEqual(pdhg.sigma, 1./operator.norm())
        self.assertAlmostEqual(pdhg.tau, 1./operator.norm())
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))

        # check if sigma is negative
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        step_size=(None, -1))

        # check if tau is negative
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(-1, None))

        # check if tau is None
        sigma = 3.0
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=(None, sigma))
        self.assertAlmostEqual(pdhg.sigma, sigma)
        self.assertAlmostEqual(pdhg.tau, 1./(sigma * operator.norm()**2))
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))
        
        # check if sigma is None
        tau = 3.0
        pdhg = PDHG(f=f, g=g, operator=operator, step_size= (tau, None))
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, 1./(tau * operator.norm()**2))
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))


        # check if sigma/tau are not None
        tau = 1.0
        sigma = 1.0
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau,sigma))
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, sigma)
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))


        # check sigma/tau as arrays, sigma wrong shape
        ig1 = ImageGeometry(2, 2)
        sigma = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size =(None, sigma))

        # check sigma/tau as arrays, tau wrong shape
        tau = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size =(tau, None))

        # check sigma not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        step_size = ("sigma", None))

        # check tau not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        step_size =( "tau", None))

        # check warning message if condition is not satisfied
        sigma = 4/operator.norm()
        tau = 1/3
        with self.assertWarnsRegex(UserWarning, "Convergence criterion"):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau,sigma))

        # check no warning message if check convergence is false
        sigma = 4/operator.norm()
        tau = 1/3
        with warnings.catch_warnings(record=True) as warnings_log:
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau,sigma) , check_convergence=False)
        self.assertEqual(warnings_log, [])

        # check no warning message if condition is satisfied
        sigma = 1/operator.norm()
        tau = 1/3
        with warnings.catch_warnings(record=True) as warnings_log:
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=[tau, sigma])
        self.assertEqual(warnings_log, [])

  
        

class TestStepSizePDHGStronglyConvex(CCPiTestClass):

    def test_deprecated_parameters(self): #TODO: remove when deprecated parameters are removed from PDHG
        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        gamma_g=0.5)        

        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        gamma_fconj=0.5)
        self.assertEqual(pdhg.step_size_rule.gamma_fconj, 0.5)
            

    def test_PDHG_strongly_convex_gamma_g(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0

        step_size_rule = PDHGStronglyConvexUpdate(initial_step_size = (tau,sigma), gamma_g=0.5)
        pdhg = PDHG(f=f, g=g, operator=operator, 
                    step_size=step_size_rule)
        pdhg.run(1, verbose=0)
        self.assertAlmostEqual(
            pdhg.theta, 1.0 / np.sqrt(1 + 2 * step_size_rule.gamma_g * tau))
        self.assertAlmostEqual(pdhg.tau, tau * pdhg.theta)
        self.assertAlmostEqual(pdhg.sigma, sigma / pdhg.theta)
        pdhg.run(4, verbose=0)
        self.assertNotEqual(pdhg.sigma, sigma)
        self.assertNotEqual(pdhg.tau, tau)

        # check negative strongly convex constant
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        gamma_g=-0.5)

        # check strongly convex constant not a number
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        gamma_g="-0.5")

    def test_PDHG_strongly_convex_gamma_fcong(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0
        step_size_rule = PDHGStronglyConvexUpdate(initial_step_size=(tau,sigma), gamma_fconj=0.5)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=step_size_rule)
        pdhg.run(1, verbose=0)
        self.assertEqual(pdhg.theta, 1.0 / np.sqrt(1 +
                         2 * step_size_rule .gamma_fconj * sigma))
        self.assertEqual(pdhg.tau, tau / pdhg.theta)
        self.assertEqual(pdhg.sigma, sigma * pdhg.theta)
        pdhg.run(4, verbose=0)
        self.assertNotEqual(pdhg.sigma, sigma)
        self.assertNotEqual(pdhg.tau, tau)

        # check negative strongly convex constant
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        gamma_fconj=-0.5)


        # check strongly convex constant not a number
        with self.assertRaises(ValueError):
            
            step_size_rule = PDHGStronglyConvexUpdate(gamma_fconj="-0.5")



    def test_PDHG_strongly_convex_both_fconj_and_g(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        with self.assertRaises(NotImplementedError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        gamma_g=0.5, gamma_fconj=0.5)
            pdhg.run(verbose=0)

      
        
class TestPDHGAdaptive2013(CCPiTestClass):
    


    def test_init(self):
        rule = PDHGAdaptiveStepSize2013(initial_step_size=[1.0, 2.0])
        self.assertEqual(rule.initial_step_size, [1.0, 2.0])
        self.assertAlmostEqual(rule.alpha, 0.95)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            PDHGAdaptiveStepSize2013(initial_step_size=[1.0])

    def test_initial_step_size_defaults(self):
        rule = PDHGAdaptiveStepSize2013(initial_step_size=[None, None])
        tau, sigma = rule.get_initial_step_size(None)
        self.assertEqual(tau, 1e5)
        self.assertEqual(sigma, 1e5)
        
        rule = PDHGAdaptiveStepSize2013(initial_step_size=[3.2, None])
        tau, sigma = rule.get_initial_step_size(None)
        self.assertEqual(tau, 3.2)
        self.assertEqual(sigma, 1e5)
        
        rule = PDHGAdaptiveStepSize2013(initial_step_size=[None, 3.2])
        tau, sigma = rule.get_initial_step_size(None)
        self.assertEqual(tau, 1e5)
        self.assertEqual(sigma, 3.2)
        
    def test_backtracking_calculation(self): 
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)
        
        rule = PDHGAdaptiveStepSize2013(initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        self.assertEqual(rule.gamma, 1)
        self.assertEqual(pdhg.sigma, 1)
        self.assertEqual(pdhg.tau, 1)
        b = rule._calculate_backtracking(pdhg)
        self.assertEqual(rule.x_resid.norm(), 3)
        self.assertEqual(rule.y_resid.norm(), 3)
        self.assertNumpyArrayAlmostEqual(rule.x_resid.as_array(), pdhg.y_tmp.as_array())
        self.assertEqual(rule.y_resid.dot(pdhg.y_tmp), 9)
        self.assertEqual(b, 1)
        

    def test_backtracking(self): #TODO:
        pass 

    def test_changing_ration(self): #TODO:
        pass

    def test_stopping_criterion(self): #TODO:
        pass

