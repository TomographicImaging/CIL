from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA, PDHG, SPDHG
from cil.optimisation.functions import BlockFunction, LeastSquares, IndicatorBox, ZeroFunction, L2NormSquared
from cil.framework import ImageGeometry, VectorGeometry, VectorData
from cil.optimisation.operators import BlockOperator,  IdentityOperator, MatrixOperator, LinearOperator

from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner, ConstantStepSize, ArmijoStepSizeRule, BarzilaiBorweinStepSizeRule, PDHGStronglyConvexUpdate, PDHGConstantStepSize, PDHGAdaptiveStepSize2013, PDHGAdaptiveStepSize2015, PDHGBayesOptimisationStepSize, StepSizeRule
from cil.optimisation.utilities import SPDHGConstantStepSize, SPDHGBayesOptimisationStepSize, SPDHGStepSizesFromRatio, Sampler
from cil.optimisation.utilities import SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle
import numpy as np

from cil.utilities import dataexample
from testclass import CCPiTestClass
from utils import has_skopt
import unittest
from unittest.mock import MagicMock
from unittest.mock import Mock
from numbers import Number
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
        test_stepsize = ArmijoStepSizeRule(
            alpha=1e3, beta=0.4, max_iterations=40, warmstart=False)
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
        test_stepsize = ArmijoStepSizeRule(
            alpha=8, beta=0.5, max_iterations=100, warmstart=False)

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
        test_stepsize = ArmijoStepSizeRule(
            alpha=8, beta=0.5, max_iterations=100, warmstart=False)

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
        ig = Aop.domain
        initial = ig.allocate()
        f = LeastSquares(Aop, b=bop, c=0.5)

        ss_rule = BarzilaiBorweinStepSizeRule(2)
        self.assertEqual(ss_rule.mode, 'short')
        self.assertEqual(ss_rule.initial, 2)
        self.assertEqual(ss_rule.adaptive, True)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)

        # Check the right errors are raised for incorrect parameters

        with self.assertRaises(TypeError):
            ss_rule = BarzilaiBorweinStepSizeRule(2, 'short', -4, )
        with self.assertRaises(TypeError):
            ss_rule = BarzilaiBorweinStepSizeRule(2, 'long', 'banana', )
        with self.assertRaises(ValueError):
            ss_rule = BarzilaiBorweinStepSizeRule(2, 'banana', 3)

        # Check stabilisation parameter unchanged if fixed
        ss_rule = BarzilaiBorweinStepSizeRule(2, 'long', 3)
        self.assertEqual(ss_rule.mode, 'long')
        self.assertFalse(ss_rule.adaptive)
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, 3)
        alg.run(2)
        self.assertEqual(ss_rule.stabilisation_param, 3)

        # Check infinity can be passed
        ss_rule = BarzilaiBorweinStepSizeRule(2, 'short', "off")
        self.assertEqual(ss_rule.mode, 'short')
        self.assertFalse(ss_rule.adaptive)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        alg.run(2)

        n = 5
        m = 5

        A = np.eye(5).astype('float32')
        b = (np.array([.5, .5, .5, .5, .5])).astype('float32')

        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig = Aop.domain
        initial = ig.allocate(0)
        f = LeastSquares(Aop, b=bop, c=0.5)
        ss_rule = BarzilaiBorweinStepSizeRule(0.22, 'long', np.inf)
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertFalse(ss_rule.is_short)
        # Check the initial step size was used
        alg.run(1)
        self.assertNumpyArrayAlmostEqual(
            np.array([.11, .11, .11, .11, .11]), alg.x.as_array())
        self.assertFalse(ss_rule.is_short)
        # check long
        alg.run(1)
        x_change = np.array([.11, .11, .11, .11, .11]) - \
            np.array([0, 0, 0, 0, 0])
        grad_change = -np.array([.39, .39, .39, .39, .39]) + \
            np.array([.5, .5, .5, .5, .5])
        step = x_change.dot(x_change)/x_change.dot(grad_change)
        self.assertNumpyArrayAlmostEqual(np.array(
            [.11, .11, .11, .11, .11])+step*np.array([.39, .39, .39, .39, .39]), alg.x.as_array())
        self.assertFalse(ss_rule.is_short)

        ss_rule = BarzilaiBorweinStepSizeRule(0.22, 'short', np.inf)
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertTrue(ss_rule.is_short)
        # Check the initial step size was used
        alg.run(1)
        self.assertNumpyArrayAlmostEqual(
            np.array([.11, .11, .11, .11, .11]), alg.x.as_array())
        self.assertTrue(ss_rule.is_short)
        # check short
        alg.run(1)
        x_change = np.array([.11, .11, .11, .11, .11]) - \
            np.array([0, 0, 0, 0, 0])
        grad_change = -np.array([.39, .39, .39, .39, .39]) + \
            np.array([.5, .5, .5, .5, .5])
        step = x_change.dot(grad_change)/grad_change.dot(grad_change)
        self.assertNumpyArrayAlmostEqual(np.array(
            [.11, .11, .11, .11, .11])+step*np.array([.39, .39, .39, .39, .39]), alg.x.as_array())
        self.assertTrue(ss_rule.is_short)

        # check stop iteration
        ss_rule = BarzilaiBorweinStepSizeRule(1, 'long', np.inf)
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        alg.run(500)
        self.assertEqual(alg.iteration, 1)

        # check adaptive
        ss_rule = BarzilaiBorweinStepSizeRule(0.001, 'long', "auto")
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg.run(2)
        self.assertNotEqual(ss_rule.stabilisation_param, np.inf)

        # check stops being adaptive

        ss_rule = BarzilaiBorweinStepSizeRule(0.0000001, 'long', "auto")
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertEqual(ss_rule.stabilisation_param, np.inf)
        alg.run(4)
        self.assertNotEqual(ss_rule.stabilisation_param, np.inf)
        a = ss_rule.stabilisation_param
        alg.run(1)
        self.assertEqual(ss_rule.stabilisation_param, a)

        # Test alternating
        ss_rule = BarzilaiBorweinStepSizeRule(0.0000001, 'alternate', "auto")
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        self.assertFalse(ss_rule.is_short)
        alg.run(2)
        self.assertTrue(ss_rule.is_short)
        alg.run(1)
        self.assertFalse(ss_rule.is_short)
        alg.run(1)
        self.assertTrue(ss_rule.is_short)


class TestPDHGConstantStepSize(CCPiTestClass):

    # TODO: remove when deprecated parameters are removed from PDHG
    def test_deprecated_parameters(self):

        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        sigma=0.5)
        self.assertEqual(pdhg.sigma, 0.5)
        self.assertEqual(pdhg.step_size_rule.sigma, 0.5)
        self.assertEqual(pdhg.step_size_rule.tau, 1. /
                         (0.5 * pdhg.operator.norm()**2))
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))

        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        tau=0.5)
        self.assertEqual(pdhg.tau, 0.5)
        self.assertEqual(pdhg.step_size_rule.tau, 0.5)
        self.assertEqual(pdhg.step_size_rule.sigma, 1. /
                         (0.5 * pdhg.operator.norm()**2))
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
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau, None))
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, 1./(tau * operator.norm()**2))
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))

        # check if sigma/tau are not None
        tau = 1.0
        sigma = 1.0
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau, sigma))
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, sigma)
        self.assertTrue(isinstance(pdhg.step_size_rule, PDHGConstantStepSize))

        # check sigma/tau as arrays, sigma wrong shape
        ig1 = ImageGeometry(2, 2)
        sigma = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(None, sigma))

        # check sigma/tau as arrays, tau wrong shape
        tau = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau, None))

        # check sigma not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        step_size=("sigma", None))

        # check tau not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        step_size=("tau", None))

        # check warning message if condition is not satisfied
        sigma = 4/operator.norm()
        tau = 1/3
        with self.assertWarnsRegex(UserWarning, "Convergence criterion"):
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(tau, sigma))

        # check no warning message if check convergence is false
        sigma = 4/operator.norm()
        tau = 1/3
        with warnings.catch_warnings(record=True) as warnings_log:
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=(
                tau, sigma), check_convergence=False)
        self.assertEqual(warnings_log, [])

        # check no warning message if condition is satisfied
        sigma = 1/operator.norm()
        tau = 1/3
        with warnings.catch_warnings(record=True) as warnings_log:
            warnings.simplefilter("always")
            pdhg = PDHG(f=f, g=g, operator=operator, step_size=[tau, sigma])
        self.assertTrue(pdhg.sigma * pdhg.tau * pdhg.operator.norm()**2 < 4/3)
        self.assertTrue(isinstance(pdhg.sigma, Number))
        self.assertTrue(isinstance(pdhg.tau, Number))
        self.assertEqual(warnings_log, [])

    def test_step_size_and_deprecated_sigma_tau_raises_valueerror(self):
        # Passing both `step_size` and the deprecated `sigma`/`tau` must raise a
        # clear ValueError. Regression: this branch previously passed keyword
        # arguments to ValueError, which raised a TypeError instead.
        operator = IdentityOperator(ImageGeometry(2, 2))
        with self.assertRaises(ValueError):
            PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=operator,
                 step_size=(0.1, 0.1), sigma=0.5)
        with self.assertRaises(ValueError):
            PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=operator,
                 step_size=(0.1, 0.1), tau=0.5)

    def test_incompatible_gd_rule_raises_valueerror(self):
        # A GD-style step-size rule returns a single scalar and has no
        # get_initial_step_size, so it is not compatible with PDHG and must
        # raise a clear ValueError rather than an opaque AttributeError.
        operator = IdentityOperator(ImageGeometry(2, 2))
        with self.assertRaises(ValueError):
            PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=operator,
                 step_size=ConstantStepSize(0.1))

    def test_wrong_shape_rule_raises_valueerror(self):
        # A rule that provides get_initial_step_size but returns wrong-shaped
        # step sizes (here a list where PDHG expects a scalar/array) must raise
        # a clear ValueError from the PDHG step-size validation.
        class _BadRule(StepSizeRule):
            def get_initial_step_size(self, algorithm):
                return 0.1, [0.1, 0.1]

            def get_step_size(self, algorithm):
                return 0.1, [0.1, 0.1]

        operator = IdentityOperator(ImageGeometry(2, 2))
        with self.assertRaises(ValueError):
            PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=operator,
                 step_size=_BadRule())


class TestStepSizePDHGStronglyConvex(CCPiTestClass):

    # TODO: remove when deprecated parameters are removed from PDHG
    def test_deprecated_parameters(self):
        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        gamma_g=0.5)

        with self.assertWarns(DeprecationWarning):
            pdhg = PDHG(f=ZeroFunction(), g=ZeroFunction(), operator=IdentityOperator(ImageGeometry(2, 2)),
                        gamma_fconj=0.5)
        self.assertEqual(pdhg.step_size_rule.gamma_fconj, 0.5)

    def test_init_invalid_initial_step_size_length(self):
        # initial_step_size must be a length-two list/tuple. A wrong length must
        # raise ValueError (regression: the error path previously referenced an
        # undefined `step_size` and raised NameError instead).
        with self.assertRaises(ValueError):
            PDHGStronglyConvexUpdate(initial_step_size=(1.0,), gamma_g=0.5)
        with self.assertRaises(ValueError):
            PDHGStronglyConvexUpdate(
                initial_step_size=(1.0, 2.0, 3.0), gamma_g=0.5)

    def test_PDHG_strongly_convex_gamma_g(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0

        step_size_rule = PDHGStronglyConvexUpdate(
            initial_step_size=(tau, sigma), gamma_g=0.5)
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
        step_size_rule = PDHGStronglyConvexUpdate(
            initial_step_size=(tau, sigma), gamma_fconj=0.5)
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
        self.assertEqual(rule.gamma, 0.9)
        self.assertEqual(rule.inner_iterations, 50)
        self.assertEqual(rule.tolerance, 1e-06)
        self.assertEqual(rule.count, 0)
        self.assertEqual(rule.beta, 0.95)
        self.assertEqual(rule.delta, 1.5)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            PDHGAdaptiveStepSize2013(initial_step_size=[1.0])

    def test_no_false_nonconvergence_warning(self):
        # Regression: the non-convergence warning used `k == inner_iterations-1`,
        # which false-fired when backtracking legitimately converged on the last
        # inner iteration. With inner_iterations=1 a converging step (b <= 1)
        # breaks at that last index, so no "did not converge" warning should be
        # logged.
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)
        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], inner_iterations=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        # Force backtracking to report convergence (b <= 1) on the first (and
        # only, hence last) inner iteration.
        rule._calculate_backtracking = MagicMock(return_value=0.5)

        logger = 'cil.optimisation.utilities.StepSizeMethods'
        with self.assertNoLogs(logger, level='WARNING'):
            pdhg.run(3)

    def test_initial_step_size_defaults(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)
        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(initial_step_size=[None, None])
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        tau, sigma = rule.get_initial_step_size(pdhg)
        self.assertEqual(tau, 10)
        self.assertEqual(sigma, 10)

        rule = PDHGAdaptiveStepSize2013(initial_step_size=[3.2, None])
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        tau, sigma = rule.get_initial_step_size(pdhg)
        self.assertEqual(tau, 3.2)
        self.assertEqual(sigma, 10)

        rule = PDHGAdaptiveStepSize2013(initial_step_size=[None, 3.2])
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        tau, sigma = rule.get_initial_step_size(pdhg)
        self.assertEqual(tau, 10)
        self.assertEqual(sigma, 3.2)

    def test_backtracking_calculation(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)
        rule.x_resid = operator.domain.allocate(0)
        rule.y_resid = operator.range.allocate(0)
        rule.x_store = operator.domain.allocate(0)
        rule.y_old = operator.range.allocate(0)
        self.assertEqual(rule.gamma, 1)
        self.assertEqual(pdhg.sigma, 1)
        self.assertEqual(pdhg.tau, 1)
        b = rule._calculate_backtracking(pdhg)
        self.assertEqual(rule.x_resid.norm(), 3)
        self.assertEqual(rule.y_resid.norm(), 3)
        self.assertNumpyArrayAlmostEqual(
            rule.x_resid.as_array(), pdhg.y_tmp.as_array())
        self.assertEqual(rule.y_resid.dot(pdhg.y_tmp), 9)
        self.assertEqual(b, 1)

    def test_backtracking_no_change_returns_zero(self):
        # When the iterate does not change (x == x_old and y == y_old) the
        # backtracking denominator is zero; ensure we return 0 (accept) rather
        # than 0/0 = nan, which would poison the step sizes.
        ig = ImageGeometry(3, 3)
        f = L2NormSquared(b=ig.allocate('random', seed=3))
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(1)
        pdhg.x = operator.domain.allocate(1)          # x == x_old  -> no primal change
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)
        rule.x_resid = operator.domain.allocate(0)
        rule.y_resid = operator.range.allocate(0)
        rule.y_old = operator.range.allocate(1)        # y == y_old  -> no dual change

        b = rule._calculate_backtracking(pdhg)
        self.assertTrue(np.isfinite(b))
        self.assertEqual(b, 0.0)
        self.assertLessEqual(b, 1)                     # caller would accept, not backtrack

    def test_backtracking(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[2.0, 1.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 3
            rule.d_norm = 3
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("Before adaptive step-size step", log_output)
        self.assertIn("Backtracking step", log_output)
        self.assertIn("Finished backtracking step", log_output)

        self.assertAlmostEqual(pdhg.sigma, 0.95**2 / (2*1.5))
        self.assertAlmostEqual(pdhg.tau, 0.95**2 / (2*1.5))
        self.assertEqual(mock_func.call_count, 3)

    def test_changing_ratio(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[0.5, 0.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 4
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)
        # Assertions
        self.assertIn("p_norm < ", log_output)
        self.assertAlmostEqual(pdhg.sigma, 1/0.05)
        self.assertAlmostEqual(pdhg.tau, 0.05)
        self.assertAlmostEqual(rule.alpha, 0.95*0.95)
        self.assertEqual(mock_func.call_count, 1)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[0.5, 0.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("No change", log_output)
        self.assertAlmostEqual(pdhg.sigma, 1)
        self.assertAlmostEqual(pdhg.tau, 1)
        self.assertAlmostEqual(rule.alpha, 0.95)
        self.assertEqual(mock_func.call_count, 1)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[0.5, 0.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 4
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)
        print(log_output)
        # Assertions
        self.assertIn("< d_norm ", log_output)
        self.assertAlmostEqual(pdhg.tau, 1/0.05)
        self.assertAlmostEqual(pdhg.sigma, 0.05)
        self.assertAlmostEqual(rule.alpha, 0.95*0.95)
        self.assertEqual(mock_func.call_count, 1)

    def test_inner_iteration_stopping_criterion(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(return_value=1.5)

        rule._calculate_backtracking = mock_func

    # Capture logs
        with self.assertLogs(level='WARNING') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("Backtracking step did not converge", log_output)

        self.assertEqual(mock_func.call_count, 51)

    def test_stopping_criterion(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)


        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, gamma=1, auto_stop=True)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(return_value=0.5)

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            for i in range(15):
                rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn('Finished backtracking step', log_output)
        self.assertIn("No change", log_output)
        self.assertAlmostEqual(pdhg.sigma, 1)
        self.assertAlmostEqual(pdhg.tau, 1)
        self.assertAlmostEqual(rule.alpha, 0.95)
        self.assertEqual(rule.count, 11)
        self.assertEqual(rule.adaptive, False)
        self.assertEqual(mock_func.call_count, 11)
        with self.assertRaises(AttributeError):
            rule.x_resid
        with self.assertRaises(AttributeError):
            rule.y_resid
        with self.assertRaises(AttributeError):
            rule.y_old


class TestPDHGAdaptive2015(CCPiTestClass):

    def test_init(self):
        rule = PDHGAdaptiveStepSize2015(initial_step_size=[1.0, 2.0])
        self.assertEqual(rule.initial_step_size, [1.0, 2.0])
        self.assertAlmostEqual(rule.alpha, 0.95)
        self.assertEqual(rule.c, 0.9)
        self.assertEqual(rule.inner_iterations, 50)
        self.assertEqual(rule.tolerance, 1e-06)
        self.assertEqual(rule.count, 0)
        self.assertEqual(rule.eta, 0.95)
        self.assertTrue(rule.adaptive)
        self.assertTrue(rule.auto_stop)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            PDHGAdaptiveStepSize2015(initial_step_size=[1.0])

    def test_initial_step_size_defaults(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)
        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2015(initial_step_size=[None, None])
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        tau, sigma = rule.get_initial_step_size(pdhg)
        self.assertEqual(tau, 10)
        self.assertEqual(sigma, 10)

        rule = PDHGAdaptiveStepSize2015(initial_step_size=[3.2, None])
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        tau, sigma = rule.get_initial_step_size(pdhg)
        self.assertEqual(tau, 3.2)
        self.assertEqual(sigma, 10)

        rule = PDHGAdaptiveStepSize2015(initial_step_size=[None, 3.2])
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        tau, sigma = rule.get_initial_step_size(pdhg)
        self.assertEqual(tau, 10)
        self.assertEqual(sigma, 3.2)

    def test_backtracking_calculation(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0, c=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)
        rule.x_resid = operator.domain.allocate(0)
        rule.y_resid = operator.range.allocate(0)
        rule.x_store = operator.domain.allocate(0)
        rule.y_old = operator.range.allocate(0)
        self.assertEqual(rule.c, 1)
        self.assertEqual(pdhg.sigma, 1)
        self.assertEqual(pdhg.tau, 1)
        b = rule._calculate_backtracking(pdhg)
        self.assertEqual(rule.x_resid.norm(), 3)
        self.assertEqual(rule.y_resid.norm(), 3)
        self.assertNumpyArrayAlmostEqual(
            rule.x_resid.as_array(), pdhg.y_tmp.as_array())
        self.assertEqual(rule.y_resid.dot(pdhg.y_tmp), 9)
        self.assertEqual(b, -18)

    def test_backtracking(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[-2,  -2, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 3
            rule.d_norm = 3
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("Before adaptive step-size step", log_output)
        self.assertIn("Backtracking step", log_output)
        self.assertIn("Finished backtracking step", log_output)

        self.assertAlmostEqual(pdhg.sigma, 0.5**2)
        self.assertAlmostEqual(pdhg.tau, 0.5**2)
        self.assertEqual(mock_func.call_count, 3)

    def test_changing_ratio(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, c=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[0.5, 0.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 4
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)
        # Assertions
        self.assertIn("< d_norm", log_output)
        self.assertAlmostEqual(pdhg.sigma, 1/0.05)
        self.assertAlmostEqual(pdhg.tau, 0.05)
        self.assertAlmostEqual(rule.alpha, 0.95*0.95)
        self.assertEqual(mock_func.call_count, 1)

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, c=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[0.5, 0.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("No change", log_output)
        self.assertAlmostEqual(pdhg.sigma, 1)
        self.assertAlmostEqual(pdhg.tau, 1)
        self.assertAlmostEqual(rule.alpha, 0.95)
        self.assertEqual(mock_func.call_count, 1)

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, c=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(side_effect=[0.5, 0.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 4
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)
        print(log_output)
        # Assertions
        self.assertIn("< p_norm ", log_output)
        self.assertAlmostEqual(pdhg.tau, 1/0.05)
        self.assertAlmostEqual(pdhg.sigma, 0.05)
        self.assertAlmostEqual(rule.alpha, 0.95*0.95)
        self.assertEqual(mock_func.call_count, 1)

    def test_inner_iteration_stopping_criterion(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0, c=1)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(return_value=-1)

        rule._calculate_backtracking = mock_func

    # Capture logs
        with self.assertLogs(level='WARNING') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("Backtracking step did not converge", log_output)

        self.assertEqual(mock_func.call_count, 51)

    def test_stopping_criterion(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)


        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, c=1, auto_stop=True)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        pdhg.x_old = operator.domain.allocate(0)
        pdhg.y_old = operator.range.allocate(0)
        pdhg.x = operator.domain.allocate(1)
        pdhg.y = operator.range.allocate(1)
        pdhg.y_tmp = operator.range.allocate(0)

        mock_func = Mock(return_value=0.5)

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        rule.delta = 1

    # Capture logs
        with self.assertLogs(level='DEBUG') as log:
            for i in range(15):
                rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn('Finished backtracking step', log_output)
        self.assertIn("No change", log_output)
        self.assertAlmostEqual(pdhg.sigma, 1)
        self.assertAlmostEqual(pdhg.tau, 1)
        self.assertAlmostEqual(rule.alpha, 0.95)
        self.assertEqual(rule.count, 11)
        self.assertEqual(rule.adaptive, False)
        self.assertEqual(mock_func.call_count, 11)
        with self.assertRaises(AttributeError):
            rule.x_resid
        with self.assertRaises(AttributeError):
            rule.y_resid
        with self.assertRaises(AttributeError):
            rule.y_old


class TestPDHGBayesOpt(CCPiTestClass):

    def test_init(self):
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[0.1, 2], n_initial_points=5, n_calls=5,  n_iterations=10, seed=42)
        self.assertEqual(rule.gamma_bounds, [0.1, 2])
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 5)
        self.assertEqual(rule.n_iterations, 10)
        self.assertEqual(rule.seed, 42)

    def test_init_default(self):
        rule = PDHGBayesOptimisationStepSize()
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 20)
        self.assertEqual(rule.n_iterations, 10)
        self.assertEqual(rule.seed, None)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            PDHGBayesOptimisationStepSize(gamma_bounds=[
                (0.1, 10)], n_initial_points=5)

        with self.assertRaises(ValueError):
            PDHGBayesOptimisationStepSize(gamma_bounds=[
                -0, 1.1], n_calls=-5)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_get_gamma_bounded(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[5, 100], n_initial_points=5, n_calls=5,  n_iterations=10, seed=42)
        pdhg = PDHG(f=f, g=g, operator=operator,
                    step_size=rule, update_objective_interval=4)

        gamma = np.sqrt(pdhg.sigma / pdhg.tau)
        self.assertTrue(5 <= gamma <= 5.5)

        self.assertEqual(pdhg.iteration, -1)
        self.assertEqual(pdhg.update_objective_interval, 4)
        self.assertEqual(pdhg.objective, [])

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_get_gamma(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random', seed=3)

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=5, n_calls=10,  n_iterations=10, seed=42)
        pdhg = PDHG(f=f, g=g, operator=operator, step_size=rule)
        gamma = np.sqrt(pdhg.sigma / pdhg.tau)
        self.assertAlmostEqual(gamma, 1.7, places=1)

class TestSPDHGConstantStepSize(CCPiTestClass):
    def setUp(self):
        self.subsets = 10

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get(size=(16, 16))

        partitioned_data = data.partition(self.subsets, 'sequential')
        self.A = BlockOperator(
            *[IdentityOperator(partitioned_data[i].geometry) for i in range(self.subsets)])


        # block function
        self.F = BlockFunction(*[L2NormSquared(b=partitioned_data[i])
                                for i in range(self.subsets)])
        alpha = 0.025
        self.G = alpha * IndicatorBox(lower=0)
        
    def test_init_and_constant_step_size(self):
        gamma = 1.
        rho = .99
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A)
        
        self.assertListEqual(
            spdhg.sigma, [rho / ni for ni in spdhg._norms])
        self.assertEqual(spdhg.tau, min([rho*pi / (si * ni**2) for pi, ni,
                                         si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))
        self.assertNumpyArrayEqual(
            spdhg.x.as_array(), self.A.domain_geometry().allocate(0).as_array())
        self.assertEqual(spdhg.update_objective_interval, 1)

        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=(100, [1]*self.subsets))
        self.assertListEqual(spdhg.sigma, [1]*self.subsets)
        self.assertEqual(spdhg.tau, 100)

        # Test SPDHG setters - set_step_sizes with sigma
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=(None, [1]*self.subsets))
        self.assertListEqual(spdhg.sigma, [1]*self.subsets)
        self.assertEqual(spdhg.tau, min([(rho*pi / (si * ni**2)) for pi, ni,
                                         si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))

        # Test SPDHG setters - set_step_sizes with tau
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=(100, None))
        self.assertListEqual(spdhg.sigma, [
                             gamma * rho*pi / (spdhg.tau*ni**2) for ni, pi in zip(spdhg._norms, spdhg._prob_weights)])
        self.assertEqual(spdhg.tau, 100)
        

    def test_init_and_constant_step_size_invalid(self):
        with self.assertRaises(ValueError):
            SPDHGConstantStepSize(step_size=[1.0])
        with self.assertRaises(ValueError):
            SPDHGConstantStepSize(step_size=[1.0, 2.0]).get_initial_step_size(SPDHG(f=self.F, g=self.G, operator=self.A))

    def test_step_size_and_deprecated_sigma_tau_raises_valueerror(self):
        # Passing both `step_size` and the deprecated `sigma`/`tau` must raise a
        # clear ValueError.
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A,
                  step_size=(100, [1]*self.subsets), sigma=0.5)
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A,
                  step_size=(100, [1]*self.subsets), tau=0.5)

    def test_incompatible_gd_rule_raises_valueerror(self):
        # A GD-style step-size rule has no get_initial_step_size and is not
        # compatible with SPDHG; it must raise a clear ValueError.
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A,
                  step_size=ConstantStepSize(0.1))
            
    def test_step_sizes_from_ratio(self):
        gamma = 3.7
        rho = 5.6
        rule = SPDHGStepSizesFromRatio(gamma,rho)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        self.assertListEqual(
            spdhg.sigma, [gamma * rho / ni for ni in spdhg._norms])
        self.assertEqual(spdhg.tau, min([pi*rho / (si * ni**2) for pi, ni,
                                            si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))


class TestSPDHGBayesStepSize(CCPiTestClass):
    def setUp(self):
        self.subsets = 4

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get(size=(4, 4))

        partitioned_data = data.partition(self.subsets, 'sequential')
        self.A = BlockOperator(
            *[IdentityOperator(partitioned_data[i].geometry) for i in range(self.subsets)])


        # block function
        self.F = BlockFunction(*[L2NormSquared(b=partitioned_data[i])
                                for i in range(self.subsets)])
        alpha = 0.025
        self.G = alpha * IndicatorBox(lower=0)
        
    def test_init(self):
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[0.1, 2], n_initial_points=5, n_calls=10,  n_iterations=10, seed=42)
        self.assertEqual(rule.gamma_bounds, [0.1, 2])
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 10)
        self.assertEqual(rule.n_iterations, 10)
        self.assertEqual(rule.seed, 42)

    def test_init_default(self):
        rule = SPDHGBayesOptimisationStepSize()
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 20)
        self.assertEqual(rule.n_iterations, None)
        self.assertEqual(rule.seed, None)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            SPDHGBayesOptimisationStepSize(gamma_bounds=[
                (0.1, 10)], n_initial_points=5)

        with self.assertRaises(ValueError):
            SPDHGBayesOptimisationStepSize(gamma_bounds=[
                -0, 1.1], n_calls=-5)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_get_gamma_bounded(self):

        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[5, 6], n_initial_points=5, n_calls=10,  n_iterations=None, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A,
                    step_size=rule, update_objective_interval=4)
        self.assertEqual(rule.n_iterations, 10*self.subsets)
        gamma = min([ pi/(spdhg.tau *ni) for pi, ni in zip(spdhg._prob_weights, spdhg._norms)])
        self.assertTrue(5.8 <= gamma <= 6)

        self.assertEqual(spdhg.iteration, -1)
        self.assertEqual(spdhg.update_objective_interval, 4)
        self.assertEqual(spdhg.objective, [])

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_get_gamma(self):
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=5, n_calls=10,  seed=42)
        sampler = Sampler.sequential(self.subsets)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule, sampler=sampler)
        gamma = min([ pi/(spdhg.tau *ni) for pi, ni in zip(spdhg._prob_weights, spdhg._norms)])
        self.assertAlmostEqual(gamma, 0.4, places=1)

class TestSPDHGAdaptiveStepSize(CCPiTestClass):
    """Tests for the adaptive SPDHG step-size rules based on arXiv:2301.02511."""

    def setUp(self):
        self.subsets = 10

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get(size=(16, 16))

        partitioned_data = data.partition(self.subsets, 'sequential')
        self.A = BlockOperator(
            *[IdentityOperator(partitioned_data[i].geometry) for i in range(self.subsets)])

        self.F = BlockFunction(*[L2NormSquared(b=partitioned_data[i])
                                 for i in range(self.subsets)])
        alpha = 0.025
        self.G = alpha * IndicatorBox(lower=0)


    def test_balancing_init(self):
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[1.0, 2.0], initial_alpha=0.9, eta=0.99,
            delta=2.0, s=3.0, auto_stop=False, auto_stop_patience=5)
        self.assertEqual(rule.initial_step_size, [1.0, 2.0])
        self.assertEqual(rule.alpha, 0.9)
        self.assertEqual(rule.eta, 0.99)
        self.assertEqual(rule.delta, 2.0)
        self.assertEqual(rule.s, 3.0)
        self.assertFalse(rule.auto_stop)
        self.assertEqual(rule.auto_stop_patience, 5)
        self.assertTrue(rule.adaptive)
        self.assertEqual(rule.count, 0)

    def test_balancing_init_defaults(self):
        rule = SPDHGAdaptiveStepSizeBalancing()
        self.assertEqual(rule.initial_step_size, [None, None])
        self.assertAlmostEqual(rule.alpha, 0.95)
        self.assertAlmostEqual(rule.eta, 0.995)
        self.assertEqual(rule.delta, 1.5)
        self.assertIsNone(rule.s)
        self.assertTrue(rule.auto_stop)
        self.assertEqual(rule.auto_stop_patience, 10)

    def test_angle_init(self):
        rule = SPDHGAdaptiveStepSizeAngle(
            initial_step_size=[1.0, 2.0], initial_alpha=0.5, eta=0.9,
            c=0.9, auto_stop=False, auto_stop_patience=3)
        self.assertEqual(rule.initial_step_size, [1.0, 2.0])
        self.assertEqual(rule.alpha, 0.5)
        self.assertEqual(rule.eta, 0.9)
        self.assertEqual(rule.c, 0.9)
        self.assertFalse(rule.auto_stop)
        self.assertEqual(rule.auto_stop_patience, 3)

    def test_angle_init_defaults(self):
        rule = SPDHGAdaptiveStepSizeAngle()
        self.assertEqual(rule.initial_step_size, [None, None])
        self.assertAlmostEqual(rule.alpha, 1.0)
        self.assertAlmostEqual(rule.eta, 0.995)
        self.assertEqual(rule.c, 0.999)
        self.assertTrue(rule.auto_stop)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            SPDHGAdaptiveStepSizeBalancing(initial_step_size=[1.0])
        with self.assertRaises(ValueError):
            SPDHGAdaptiveStepSizeAngle(initial_step_size=[1.0, 2.0, 3.0])


    def test_initial_step_size_defaults(self):
        rho = 0.99
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            rule = RuleCls()
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
            # tau scalar, sigma list of length n, all positive
            self.assertTrue(isinstance(spdhg.tau, Number))
            self.assertEqual(len(spdhg.sigma), self.subsets)
            self.assertTrue(all(s > 0 for s in spdhg.sigma))
            # matches the standard SPDHG relations
            self.assertListEqual(spdhg.sigma, [rho / ni for ni in spdhg._norms])
            self.assertEqual(spdhg.tau, min([rho*pi / (si * ni**2) for pi, ni, si in
                                             zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))
            # per-operator convergence criterion  sigma_i * tau * ||A_i||^2 <= p_i
            for si, ni, pi in zip(spdhg.sigma, spdhg._norms, spdhg._prob_weights):
                self.assertLessEqual(si * spdhg.tau * ni**2, pi + 1e-9)

    def test_initial_step_size_scalar_sigma_broadcast(self):
        # a single dual step size is broadcast to a per-operator list
        rule = SPDHGAdaptiveStepSizeBalancing(initial_step_size=[3.0, 0.5])
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        self.assertEqual(spdhg.tau, 3.0)
        self.assertListEqual(spdhg.sigma, [0.5]*self.subsets)

    def test_initial_step_size_tau_from_sigma_list(self):
        # sigma given as a list, tau derived
        rho = 0.99
        rule = SPDHGAdaptiveStepSizeAngle(initial_step_size=[None, [0.7]*self.subsets])
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        self.assertListEqual(spdhg.sigma, [0.7]*self.subsets)
        self.assertEqual(spdhg.tau, min([rho*pi / (si * ni**2) for pi, ni, si in
                                         zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))

    

    def test_balancing_adapts_step_sizes(self):
        # from a deliberately imbalanced ratio, the rule should rescale the step sizes
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[10.0, 0.001], auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        start_ratio = spdhg.tau / spdhg.sigma[0]
        spdhg.run(20)
        end_ratio = spdhg.tau / spdhg.sigma[0]
        self.assertNotAlmostEqual(start_ratio, end_ratio)
        # step sizes remain valid (positive scalar tau, positive sigma list)
        self.assertGreater(spdhg.tau, 0)
        self.assertEqual(len(spdhg.sigma), self.subsets)
        self.assertTrue(all(s > 0 for s in spdhg.sigma))
        # the adaptation strength decayed as the step sizes were rebalanced
        self.assertLess(rule.alpha, 0.95)

    def test_angle_adapts_step_sizes(self):
        # a permissive threshold (c=0) makes every iteration rebalance, exercising the
        # angle-alignment mechanism (the default c=0.999 is deliberately conservative).
        rule = SPDHGAdaptiveStepSizeAngle(
            initial_step_size=[10.0, 0.001], c=0.0, auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        start_ratio = spdhg.tau / spdhg.sigma[0]
        spdhg.run(20)
        end_ratio = spdhg.tau / spdhg.sigma[0]
        self.assertNotAlmostEqual(start_ratio, end_ratio)
        self.assertGreater(spdhg.tau, 0)
        self.assertEqual(len(spdhg.sigma), self.subsets)
        self.assertTrue(all(s > 0 for s in spdhg.sigma))

    def test_products_preserved_under_rescaling(self):
        # rescaling multiplies tau by 1/f and every sigma_i by f, so tau*sigma_i is fixed
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            rule = RuleCls(initial_step_size=[10.0, 0.001], auto_stop=False)
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
            prod0 = [spdhg.tau * si for si in spdhg.sigma]
            spdhg.run(15)
            prod1 = [spdhg.tau * si for si in spdhg.sigma]
            for a, b in zip(prod0, prod1):
                self.assertAlmostEqual(a, b)

    def test_auto_stop_freezes_and_frees(self):
        # force the "no change" branch every iteration by making the tolerance gate
        # impossible to pass; the rule should then stop after auto_stop_patience.
        rule = SPDHGAdaptiveStepSizeBalancing(auto_stop=True, auto_stop_patience=3)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        rule.tolerance = 1e12
        self.assertTrue(rule.adaptive)
        spdhg.run(6)
        self.assertFalse(rule.adaptive)
        # buffers released on auto-stop
        self.assertFalse(hasattr(rule, 'x_prev') and rule.x_prev is not None)
        # continuing to run after auto-stop keeps returning valid step sizes
        tau_frozen = spdhg.tau
        sigma_frozen = list(spdhg.sigma)
        spdhg.run(3)
        self.assertEqual(spdhg.tau, tau_frozen)
        self.assertListEqual(list(spdhg.sigma), sigma_frozen)

    def test_auto_stop_disabled_keeps_adapting(self):
        rule = SPDHGAdaptiveStepSizeAngle(auto_stop=False, auto_stop_patience=2)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        spdhg.run(20)
        self.assertTrue(rule.adaptive)
        self.assertTrue(hasattr(rule, 'x_prev'))



