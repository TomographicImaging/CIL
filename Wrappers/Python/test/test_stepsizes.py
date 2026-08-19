from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA, PDHG, SPDHG
from cil.optimisation.functions import BlockFunction, LeastSquares, IndicatorBox, ZeroFunction, L2NormSquared, L1Norm
from cil.framework import ImageGeometry, VectorGeometry, VectorData
from cil.optimisation.operators import BlockOperator,  IdentityOperator, MatrixOperator, LinearOperator

from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner, ConstantStepSize, ArmijoStepSizeRule, BarzilaiBorweinStepSizeRule, PDHGStronglyConvexUpdate, PDHGConstantStepSize, PDHGAdaptiveStepSize2013, PDHGAdaptiveStepSize2015, PDHGBayesOptimisationStepSize, StepSizeRule
from cil.optimisation.utilities import SPDHGConstantStepSize, SPDHGBayesOptimisationStepSize, SPDHGStepSizesFromRatio, Sampler
from cil.optimisation.utilities import SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle
from cil.optimisation.utilities.StepSizeMethods import _spdhg_tau_from_sigma
import numpy as np

from testclass import CCPiTestClass
from utils import has_skopt
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import Mock
from numbers import Number
from types import SimpleNamespace
import sys
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

  
    NORM_SCALING = 2.

    def setUp(self):
        ig = ImageGeometry(3, 3)
        self.data = ig.allocate('random', seed=3)
        self.A = self.NORM_SCALING * IdentityOperator(ig)
        self.F = L2NormSquared(b=self.data)
        self.G = L2NormSquared()

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
        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], inner_iterations=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        # Force backtracking to report convergence (b <= 1) on the first (and
        # only, hence last) inner iteration.
        rule._calculate_backtracking = MagicMock(return_value=0.5)

        logger = 'cil.optimisation.utilities.StepSizeMethods'
        with self.assertNoLogs(logger, level='WARNING'):
            pdhg.run(3, verbose=0)

    def test_initial_step_size_defaults(self):

        default = 10. / self.A.norm()
        self.assertEqual(default, 5.)
        for initial, expected in ([None, None], (default, default)), \
                                 ([3.2, None], (3.2, default)), \
                                 ([None, 3.2], (default, 3.2)):
            with self.subTest(initial_step_size=initial):
                rule = PDHGAdaptiveStepSize2013(initial_step_size=initial)
                pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
                self.assertEqual(rule.get_initial_step_size(pdhg), expected)

    def test_backtracking_calculation(self):
        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)
        rule.x_resid = self.A.domain.allocate(0)
        rule.y_resid = self.A.range.allocate(0)
        rule.y_old = self.A.range.allocate(0)
        self.assertEqual(rule.gamma, 1)
        self.assertEqual(pdhg.sigma, 1)
        self.assertEqual(pdhg.tau, 1)
        b = rule._calculate_backtracking(pdhg)
        self.assertEqual(rule.x_resid.norm(), 3)
        self.assertEqual(rule.y_resid.norm(), 3)
        # y_tmp = K x_resid, so with ||K|| = 2 it is twice the primal residual
        self.assertNumpyArrayAlmostEqual(
            self.NORM_SCALING * rule.x_resid.as_array(), pdhg.y_tmp.as_array())
        self.assertEqual(rule.y_resid.dot(pdhg.y_tmp), 18)
        # b = |2 sigma tau <dy, K dx>| / (gamma sigma ||dx||^2 + gamma tau ||dy||^2)
        #   = |2*18| / (9 + 9) = 2
        self.assertEqual(b, 2)

    def test_backtracking_no_change_returns_zero(self):
        # When the iterate does not change (x == x_old and y == y_old) the
        # backtracking denominator is zero; ensure we return 0 (accept) rather
        # than 0/0 = nan, which would poison the step sizes.
        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(1)
        pdhg.x = self.A.domain.allocate(1)          # x == x_old  -> no primal change
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)
        rule.x_resid = self.A.domain.allocate(0)
        rule.y_resid = self.A.range.allocate(0)
        rule.y_old = self.A.range.allocate(1)        # y == y_old  -> no dual change

        b = rule._calculate_backtracking(pdhg)
        self.assertTrue(np.isfinite(b))
        self.assertEqual(b, 0.0)
        self.assertLessEqual(b, 1)                     # caller would accept, not backtrack

    def test_backtracking(self):
        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)

        mock_func = Mock(side_effect=[2.0, 1.5, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 3
            rule.d_norm = 3
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

    # Capture logs
        with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='DEBUG') as log:
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
        # The rebalancing step compares the primal and dual residuals against a band
        # [s/delta, s*delta] scaled by s, which defaults to ||K|| 
        # Backtracking and the residuals themselves are mocked, leaving only the
        # branch choice and the resulting rescaling under test.
        delta, alpha, eta = 1.5, 0.95, 0.95
        cases = [
            # p_norm < (s/delta)*d_norm: the dual is behind, so tau is shrunk
            (1., 10., 'p_norm < (s/delta)*d_norm',
             1. * (1 - alpha), 1. / (1 - alpha), alpha * eta),
            # equal residuals are still below the lower edge s/delta = 4/3, so this
            # case only lands in the shrink branch because s = ||K|| = 2 > 1
            (1., 1., 'p_norm < (s/delta)*d_norm',
             1. * (1 - alpha), 1. / (1 - alpha), alpha * eta),
            # inside the band [s/delta, s*delta] = [4/3, 3]: nothing changes, and
            # alpha does not decay. Below the upper edge only because s > 1
            (2., 1., 'No change', 1., 1., alpha),
            # (s*delta)*d_norm < p_norm: the primal is behind, so tau is boosted
            (10., 1., '(s*delta)*d_norm < p_norm',
             1. / (1 - alpha), 1. * (1 - alpha), alpha * eta),
        ]
        for p_norm, d_norm, message, tau, sigma, alpha_after in cases:
            with self.subTest(p_norm=p_norm, d_norm=d_norm):
                rule = PDHGAdaptiveStepSize2013(
                    initial_step_size=[1.0, 1.0], initial_alpha=alpha, gamma=1,
                    delta=delta)
                pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
                pdhg.x_old = self.A.domain.allocate(0)
                pdhg.y_old = self.A.range.allocate(0)
                pdhg.x = self.A.domain.allocate(1)
                pdhg.y = self.A.range.allocate(1)
                pdhg.y_tmp = self.A.range.allocate(0)

                # b <= 1, so backtracking accepts the step on its first try
                mock_func = Mock(side_effect=[0.5, 0.5, 0.5])
                rule._calculate_backtracking = mock_func

                def mock_pnorm_dnorm(algorithm, p_norm=p_norm, d_norm=d_norm):
                    rule.x_resid.fill(1)
                    rule.y_resid.fill(1)
                    rule.p_norm = p_norm
                    rule.d_norm = d_norm
                rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

                with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='DEBUG') as log:
                    rule.get_step_size(pdhg)

                self.assertEqual(rule.s, self.A.norm())   # s defaulted to ||K||
                self.assertIn(message, "\n".join(log.output))
                self.assertAlmostEqual(pdhg.tau, tau)
                self.assertAlmostEqual(pdhg.sigma, sigma)
                self.assertAlmostEqual(rule.alpha, alpha_after)
                self.assertEqual(mock_func.call_count, 1)

    def test_inner_iteration_stopping_criterion(self):

        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0, gamma=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)

        mock_func = Mock(return_value=1.5)

        rule._calculate_backtracking = mock_func

    # Capture logs
        with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='WARNING') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("Backtracking step did not converge", log_output)

        self.assertEqual(mock_func.call_count, 51)

    def test_first_backtracking_retry_restores_initial_dual(self):
        # Regression: y_old used to be allocated as zeros on the first get_step_size
        # call, so a backtracking retry during iteration 0 restored the dual variable
        # to zero rather than to the dual the algorithm was started from.
        y0 = self.A.range.allocate(7.0)
        rule = PDHGAdaptiveStepSize2013(initial_step_size=[50.0, 50.0])
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                    initial=[self.A.domain.allocate(0.3), y0])
        self.assertNumpyArrayAlmostEqual(rule.y_old.as_array(), y0.as_array())

        restored = []
        original_restore = rule._restore_iterate

        def spy(algorithm):
            original_restore(algorithm)
            restored.append(algorithm.y.copy())
        rule._restore_iterate = spy

        pdhg.run(1, verbose=0)
        # the deliberately oversized step sizes force at least one retry
        self.assertGreater(len(restored), 0)
        self.assertNumpyArrayAlmostEqual(restored[0].as_array(), y0.as_array())

    def test_stopping_criterion(self):


        rule = PDHGAdaptiveStepSize2013(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, gamma=1, auto_stop=True)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)

        mock_func = Mock(return_value=0.5)

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 2
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm
        # s = ||K|| = 2 and delta = 1.5, so p_norm = 2, d_norm = 1 sits inside the
        # band [s/delta, s*delta] = [4/3, 3] and the rule reports no change
        rule.delta = 1.5

    # Capture logs
        with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='DEBUG') as log:
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
        # the backtracking restore copy is released along with the other three
        with self.assertRaises(AttributeError):
            rule.x_prev


class TestPDHGAdaptive2015(CCPiTestClass):


    NORM_SCALING = 2.

    def setUp(self):
        ig = ImageGeometry(3, 3)
        self.data = ig.allocate('random', seed=3)
        self.A = self.NORM_SCALING * IdentityOperator(ig)
        self.F = L2NormSquared(b=self.data)
        self.G = L2NormSquared()

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

        default = 10. / self.A.norm()
        self.assertEqual(default, 5.)
        for initial, expected in ([None, None], (default, default)), \
                                 ([3.2, None], (3.2, default)), \
                                 ([None, 3.2], (default, 3.2)):
            with self.subTest(initial_step_size=initial):
                rule = PDHGAdaptiveStepSize2015(initial_step_size=initial)
                pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
                self.assertEqual(rule.get_initial_step_size(pdhg), expected)

    def test_backtracking_calculation(self):
        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0, c=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)
        rule.x_resid = self.A.domain.allocate(0)
        rule.y_resid = self.A.range.allocate(0)
        rule.y_old = self.A.range.allocate(0)
        self.assertEqual(rule.c, 1)
        self.assertEqual(pdhg.sigma, 1)
        self.assertEqual(pdhg.tau, 1)
        b = rule._calculate_backtracking(pdhg)
        self.assertEqual(rule.x_resid.norm(), 3)
        self.assertEqual(rule.y_resid.norm(), 3)
        # y_tmp = K x_resid, so with ||K|| = 2 it is twice the primal residual
        self.assertNumpyArrayAlmostEqual(
            self.NORM_SCALING * rule.x_resid.as_array(), pdhg.y_tmp.as_array())
        self.assertEqual(rule.y_resid.dot(pdhg.y_tmp), 18)
        # b = c sigma ||dx||^2 + c tau ||dy||^2 - |4 sigma tau <dy, K dx>|
        #   = 9 + 9 - |4*18| = -54
        self.assertEqual(b, -54)

    def test_backtracking(self):
        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)

        mock_func = Mock(side_effect=[-2,  -2, 0.5])

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 3
            rule.d_norm = 3
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

    # Capture logs
        with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='DEBUG') as log:
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
        # The 2015 rebalancing step compares the residuals against a fixed factor of
        # two rather than the 2013 rule's s-scaled band, so unlike that rule the
        # branch taken here does not depend on ||K||.
        alpha, eta = 0.95, 0.95
        cases = [
            # 2*p_norm < d_norm: the dual is behind, so tau is shrunk
            (1., 10., '2*p_norm < d_norm',
             1. * (1 - alpha), 1. / (1 - alpha), alpha * eta),
            # neither residual dominates: nothing changes, and alpha does not decay
            (1., 1., 'No change', 1., 1., alpha),
            # 2*d_norm < p_norm: the primal is behind, so tau is boosted
            (10., 1., '2*d_norm < p_norm',
             1. / (1 - alpha), 1. * (1 - alpha), alpha * eta),
        ]
        for p_norm, d_norm, message, tau, sigma, alpha_after in cases:
            with self.subTest(p_norm=p_norm, d_norm=d_norm):
                rule = PDHGAdaptiveStepSize2015(
                    initial_step_size=[1.0, 1.0], initial_alpha=alpha, c=1)
                pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
                pdhg.x_old = self.A.domain.allocate(0)
                pdhg.y_old = self.A.range.allocate(0)
                pdhg.x = self.A.domain.allocate(1)
                pdhg.y = self.A.range.allocate(1)
                pdhg.y_tmp = self.A.range.allocate(0)

                # b >= 0, so backtracking accepts the step on its first try
                mock_func = Mock(side_effect=[0.5, 0.5, 0.5])
                rule._calculate_backtracking = mock_func

                def mock_pnorm_dnorm(algorithm, p_norm=p_norm, d_norm=d_norm):
                    rule.x_resid.fill(1)
                    rule.y_resid.fill(1)
                    rule.p_norm = p_norm
                    rule.d_norm = d_norm
                rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

                with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='DEBUG') as log:
                    rule.get_step_size(pdhg)

                self.assertIn(message, "\n".join(log.output))
                self.assertAlmostEqual(pdhg.tau, tau)
                self.assertAlmostEqual(pdhg.sigma, sigma)
                self.assertAlmostEqual(rule.alpha, alpha_after)
                self.assertEqual(mock_func.call_count, 1)

    def test_inner_iteration_stopping_criterion(self):

        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0, c=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)

        mock_func = Mock(return_value=-1)

        rule._calculate_backtracking = mock_func

    # Capture logs
        with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='WARNING') as log:
            rule.get_step_size(pdhg)

        # Combine log messages
        log_output = "\n".join(log.output)

        # Assertions
        self.assertIn("Backtracking step did not converge", log_output)

        self.assertEqual(mock_func.call_count, 51)

    def test_first_backtracking_retry_restores_initial_dual(self):
        # Regression: y_old used to be allocated as zeros on the first get_step_size
        # call, so a backtracking retry during iteration 0 restored the dual variable
        # to zero rather than to the dual the algorithm was started from.
        y0 = self.A.range.allocate(7.0)
        rule = PDHGAdaptiveStepSize2015(initial_step_size=[50.0, 50.0])
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                    initial=[self.A.domain.allocate(0.3), y0])
        self.assertNumpyArrayAlmostEqual(rule.y_old.as_array(), y0.as_array())

        restored = []
        original_restore = rule._restore_iterate

        def spy(algorithm):
            original_restore(algorithm)
            restored.append(algorithm.y.copy())
        rule._restore_iterate = spy

        pdhg.run(1, verbose=0)
        # the deliberately oversized step sizes force at least one retry
        self.assertGreater(len(restored), 0)
        self.assertNumpyArrayAlmostEqual(restored[0].as_array(), y0.as_array())

    def test_stopping_criterion(self):


        rule = PDHGAdaptiveStepSize2015(
            initial_step_size=[1.0, 1.0], initial_alpha=0.95, c=1, auto_stop=True)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        pdhg.x_old = self.A.domain.allocate(0)
        pdhg.y_old = self.A.range.allocate(0)
        pdhg.x = self.A.domain.allocate(1)
        pdhg.y = self.A.range.allocate(1)
        pdhg.y_tmp = self.A.range.allocate(0)

        mock_func = Mock(return_value=0.5)

        rule._calculate_backtracking = mock_func

        def mock_pnorm_dnorm(algorithm):
            rule.x_resid.fill(1)
            rule.y_resid.fill(1)
            rule.p_norm = 1
            rule.d_norm = 1
        rule._calculate_pnorm_dnorm = mock_pnorm_dnorm

    # Capture logs
        with self.assertLogs('cil.optimisation.utilities.StepSizeMethods', level='DEBUG') as log:
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
        # the backtracking restore copy is released along with the other three
        with self.assertRaises(AttributeError):
            rule.x_prev


class TestPDHGBayesOpt(CCPiTestClass):

    NORM_SCALING = 2.

    def setUp(self):
        ig = ImageGeometry(3, 3)
        self.data = ig.allocate('random', seed=3)
        self.A = self.NORM_SCALING * IdentityOperator(ig)
        self.F = L2NormSquared(b=self.data)
        self.G = L2NormSquared()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def test_init(self):
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[0.1, 2], n_initial_points=5, n_calls=10,
            n_iterations=10, seed=42, plot=False)
        self.assertEqual(rule.gamma_bounds, [0.1, 2])
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 10)
        self.assertEqual(rule.n_iterations, 10)
        self.assertEqual(rule.seed, 42)
        self.assertFalse(rule.plot)

    def test_init_default(self):
        rule = PDHGBayesOptimisationStepSize()
        self.assertIsNone(rule.gamma_bounds)
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 20)
        # unlike the SPDHG rule this one has a fixed default rather than one
        # resolved from the problem at set-up
        self.assertEqual(rule.n_iterations, 10)
        self.assertIsNone(rule.seed)
        self.assertFalse(rule.plot)

    def test_init_invalid_gamma_bounds_length(self):
        with self.assertRaises(ValueError):
            PDHGBayesOptimisationStepSize(gamma_bounds=[(0.1, 10)])
        with self.assertRaises(ValueError):
            PDHGBayesOptimisationStepSize(gamma_bounds=[0.1, 1., 10.])

    def test_init_invalid_gamma_bounds_non_positive(self):
        # every other argument is left valid, so the ValueError can only come
        # from the bound under test
        for bounds in ([0., 1.1], [-1., 1.1], [0.1, -1.1], [0.1, 0.]):
            with self.subTest(gamma_bounds=bounds):
                with self.assertRaises(ValueError):
                    PDHGBayesOptimisationStepSize(gamma_bounds=bounds)

    def test_init_invalid_gamma_bounds_not_increasing(self):
        # reversed or degenerate bounds give an empty log-gamma search space
        for bounds in ([6., 5.], [1., 1.]):
            with self.subTest(gamma_bounds=bounds):
                with self.assertRaises(ValueError):
                    PDHGBayesOptimisationStepSize(gamma_bounds=bounds)

    def test_init_invalid_call_counts(self):
        for kwargs in (dict(n_initial_points=0), dict(n_initial_points=-1),
                       dict(n_initial_points=2.5), dict(n_calls=-5),
                       dict(n_calls=0), dict(n_calls=2.5),
                       # skopt cannot draw more initial points than it has calls
                       dict(n_initial_points=5, n_calls=3)):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    PDHGBayesOptimisationStepSize(**kwargs)

    def test_init_invalid_n_iterations(self):
        # n_iterations=1 leaves the trial with no recorded objective to score,
        # which used to surface as an IndexError from inside the objective
        for n_iterations in (1, 0, -10, 4.5):
            with self.subTest(n_iterations=n_iterations):
                with self.assertRaises(ValueError):
                    PDHGBayesOptimisationStepSize(n_iterations=n_iterations)

    def test_init_accepts_valid_call_counts(self):
        rule = PDHGBayesOptimisationStepSize(
            n_initial_points=1, n_calls=1, n_iterations=2, gamma_bounds=[1e-3, 1e3])
        self.assertEqual(rule.n_calls, 1)
        self.assertEqual(rule.n_iterations, 2)
        self.assertIsNone(PDHGBayesOptimisationStepSize(n_iterations=None).n_iterations)

    def test_default_n_iterations(self):
        rule = PDHGBayesOptimisationStepSize(n_iterations=None)
        self.assertEqual(rule._default_n_iterations(SimpleNamespace()), 10)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_defaults_are_resolved_from_the_problem_at_set_up(self):
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=2, n_calls=3,
            n_iterations=None, seed=42)
        PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)

        self.assertEqual(rule.n_iterations, 10)
        # the bounds are derived from the objective at the zero image and the operator norm
        zero_x = self.A.domain_geometry().allocate(0)
        ratio = np.sqrt(self.F(self.A.direct(zero_x)) +
                        self.G(zero_x)) / self.A.norm()
        bounds = rule.gamma_bounds
        # compared as ratios: the bounds span ten orders of magnitude, so an
        # absolute tolerance would make the lower bound assertion vacuous
        self.assertAlmostEqual(bounds[0] / (1e-5 / ratio), 1.)
        self.assertAlmostEqual(bounds[1] / (1e5 / ratio), 1.)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_default_gamma_bounds_when_f_vanishes_at_zero(self):

        f = L1Norm()
        g = L2NormSquared(b=self.data)
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=2, n_calls=3,
            n_iterations=2, seed=42)
        PDHG(f=f, g=g, operator=self.A, step_size=rule)

        zero_x = self.A.domain_geometry().allocate(0)
        self.assertEqual(f(self.A.direct(zero_x)), 0.)
        ratio = np.sqrt(g(zero_x)) / self.A.norm()
        bounds = rule.gamma_bounds
        self.assertTrue(np.all(np.isfinite(bounds)))
        self.assertAlmostEqual(bounds[0] / (1e-5 / ratio), 1.)
        self.assertAlmostEqual(bounds[1] / (1e5 / ratio), 1.)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_default_gamma_bounds_raises_if_objective_vanishes_at_zero(self):

        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=2, n_calls=3,
            n_iterations=2, seed=42)
        with self.assertRaises(ValueError):
            PDHG(f=L1Norm(), g=L2NormSquared(), operator=self.A, step_size=rule)

    # ------------------------------------------------------------------
    # the gamma -> (tau, sigma) mapping
    # ------------------------------------------------------------------

    def test_step_sizes_from_gamma(self):
        rule = PDHGBayesOptimisationStepSize()
        gamma = 2.5
        tau, sigma = rule._step_sizes_from_gamma(
            SimpleNamespace(operator=self.A), gamma)

        self.assertAlmostEqual(tau, 1. / (gamma * self.NORM_SCALING))
        self.assertAlmostEqual(sigma, gamma / self.NORM_SCALING)
        self.assertAlmostEqual(tau, 0.2)
        self.assertAlmostEqual(sigma, 1.25)
        self.assertAlmostEqual(tau * sigma, 1. / self.NORM_SCALING ** 2)

    # ------------------------------------------------------------------
    # against the real optimiser
    # ------------------------------------------------------------------

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_step_sizes_match_the_gamma_the_optimiser_returned(self):
        # bounds force gamma to be very close to 2.5, so the rule's step sizes are known in advance
        gamma = 2.5
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[gamma * (1 - 1e-6), gamma * (1 + 1e-6)],
            n_initial_points=2, n_calls=3, n_iterations=4, seed=42)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)

        self.assertAlmostEqual(rule.gamma, gamma, places=5)
        self.assertAlmostEqual(np.sqrt(pdhg.sigma / pdhg.tau), gamma, places=5)
        self.assertAlmostEqual(pdhg.tau, 0.2, places=5)
        self.assertAlmostEqual(pdhg.sigma, 1.25, places=5)

        self.assertAlmostEqual(pdhg.tau, rule.tau)
        self.assertAlmostEqual(pdhg.sigma, rule.sigma)
        self.assertEqual(rule.get_step_size(pdhg), (rule.tau, rule.sigma))

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_step_sizes_are_consistent_for_whichever_gamma_is_chosen(self):
        # Test returned gamma is in the right range and the step sizes are consistent with it. The optimiser is free to choose any gamma in the range, so the test cannot assert a particular value.
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)

        gamma = np.sqrt(pdhg.sigma / pdhg.tau)
        self.assertTrue(2. - 1e-9 <= gamma <= 3. + 1e-9,
                        "gamma {} outside the requested bounds".format(gamma))
        # the gamma the rule reports must be the one its step sizes were built from
        self.assertAlmostEqual(rule.gamma, gamma)

        self.assertAlmostEqual(pdhg.tau, 1. / (gamma * self.NORM_SCALING))
        self.assertAlmostEqual(pdhg.sigma, gamma / self.NORM_SCALING)
        self.assertAlmostEqual(pdhg.tau * pdhg.sigma, 1. / self.NORM_SCALING ** 2)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_algorithm_state_is_clean_after_the_search(self):
        # the rule temporarily sets the algorithm's iteration and objective to score the trial, but must restore them to run the algorithm from scratch afterwards
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                    update_objective_interval=4)

        self.assertEqual(pdhg.iteration, -1)
        self.assertEqual(pdhg.objective, [])

        self.assertEqual(pdhg.update_objective_interval, 4)
        self.assertNumpyArrayEqual(
            pdhg.x.as_array(), self.A.domain_geometry().allocate(0).as_array())

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_extreme_bounds_still_give_usable_step_sizes(self):
        #  an absurd interval still installs finite, positive step sizes.
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[1e55, 1e65], n_initial_points=2, n_calls=4,
            n_iterations=4, seed=1)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)

        self.assertTrue(np.isfinite(pdhg.tau) and pdhg.tau > 0.)
        self.assertTrue(np.isfinite(pdhg.sigma) and pdhg.sigma > 0.)
        self.assertAlmostEqual(pdhg.tau * pdhg.sigma, 1. / self.NORM_SCALING ** 2)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_step_sizes_are_constant_during_the_run(self):
        #  gamma is chosen once, up front and not changed
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        pdhg = PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)

        tau, sigma = pdhg.tau, pdhg.sigma
        pdhg.run(5, callbacks=[])
        self.assertEqual(pdhg.tau, tau)
        self.assertEqual(pdhg.sigma, sigma)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_seed_makes_the_search_reproducible(self):
        pdhgs = []
        for _ in range(2):
            rule = PDHGBayesOptimisationStepSize(
                gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
                n_iterations=4, seed=42)
            pdhgs.append(PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule))

        first, second = pdhgs
        self.assertEqual(first.tau, second.tau)
        self.assertEqual(first.sigma, second.sigma)

    def test_missing_skopt_raises_an_informative_importerror(self):
        # the one path that must behave *without* scikit-optimize, so the import
        # is blocked rather than the optimiser replaced
        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        with patch.dict(sys.modules, {'skopt': None}):
            with self.assertRaises(ImportError) as context:
                PDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        self.assertIn('scikit-optimize', str(context.exception))


class TestSPDHGConstantStepSize(CCPiTestClass):

    # Scaled identities, so the operator norms are exactly NORM_SCALINGS. With ten
    # unscaled identities every candidate rho*p_i/(sigma_i ||K_i||^2) is the same
    # number, and the min, the square and the p_i weighting are all invisible.
    NORM_SCALINGS = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

    def setUp(self):
        self.subsets = len(self.NORM_SCALINGS)
        ig = ImageGeometry(4, 4)
        self.data = ig.allocate('random', seed=3)
        self.A = BlockOperator(*[s * IdentityOperator(ig)
                                 for s in self.NORM_SCALINGS])
        self.F = BlockFunction(*[L2NormSquared(b=self.data)
                                 for _ in range(self.subsets)])
        self.G = 0.025 * IndicatorBox(lower=0)

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

    def test_init_and_constant_step_size_invalid_sigma(self):
        # sigma must be a list of positive numbers with one entry per operator
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A,
                  step_size=(None, [1.0]*(self.subsets - 1)))
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A,
                  step_size=(None, [1.0]*(self.subsets - 1) + [-1.0]))

    def test_init_and_constant_step_size_invalid_tau(self):
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A,
                  step_size=(-1.0, [1.0]*self.subsets))
        # tau = 0 must give the same clear error rather than a ZeroDivisionError
        # from computing sigma_i = rho p_i / (tau ||K_i||^2)
        with self.assertRaises(ValueError):
            SPDHG(f=self.F, g=self.G, operator=self.A, step_size=(0.0, None))

    def test_deprecated_parameters(self):
        # sigma and tau are deprecated in favour of step_size, but until they are
        # removed they must still warn and still reach the algorithm
        with self.assertWarns(DeprecationWarning):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A,
                          tau=0.05, sigma=[1.0]*self.subsets)
        self.assertEqual(spdhg.tau, 0.05)
        self.assertListEqual(list(spdhg.sigma), [1.0]*self.subsets)

        with self.assertWarns(DeprecationWarning):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, tau=0.05)
        self.assertEqual(spdhg.tau, 0.05)

        with self.assertWarns(DeprecationWarning):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A,
                          sigma=[1.0]*self.subsets)
        self.assertListEqual(list(spdhg.sigma), [1.0]*self.subsets)

    def test_wrong_shape_rule_raises_valueerror(self):
        # a rule with the right interface but the wrong output shape must be caught
        # by the SPDHG step-size validation, not fail later with an opaque error
        class _BadRule(StepSizeRule):
            def __init__(self, step_size):
                self.step_size = step_size

            def get_initial_step_size(self, algorithm):
                return self.step_size

            def get_step_size(self, algorithm):
                return self.step_size

        bad_step_sizes = [
            (-1.0, [1.0]*self.subsets),                  # non-positive tau
            ([0.1], [1.0]*self.subsets),                 # tau not a scalar
            (0.1, 0.5),                                  # scalar sigma
            (0.1, [1.0]*(self.subsets - 1)),             # wrong-length sigma
            (0.1, [1.0]*(self.subsets - 1) + [0.0]),     # non-positive sigma entry
        ]
        for step_size in bad_step_sizes:
            with self.assertRaises(ValueError):
                SPDHG(f=self.F, g=self.G, operator=self.A,
                      step_size=_BadRule(step_size))

    def test_constant_step_size_unchanged_by_iterations(self):
        # get_step_size returns the same values every
        # iteration
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A,
                      step_size=(0.05, [1.0]*self.subsets),
                      sampler=Sampler.random_with_replacement(self.subsets, seed=42))
        tau, sigma = spdhg.tau, list(spdhg.sigma)
        spdhg.run(5, verbose=0)
        self.assertEqual(spdhg.tau, tau)
        self.assertListEqual(list(spdhg.sigma), sigma)

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
            
    def test_tau_from_sigma_raises_when_no_usable_block(self):
        # every block is skipped -- zero operator norm, or zero probability weight --
        # so no tau can be formed and the helper must say so rather than return inf
        with self.assertRaises(ValueError):
            _spdhg_tau_from_sigma([1., 1.], [0., 0.], [0.5, 0.5], 0.99)
        with self.assertRaises(ValueError):
            _spdhg_tau_from_sigma([1., 1.], [1., 1.], [0., 0.], 0.99)

    def test_tau_from_sigma_skips_unusable_blocks(self):
        # a block with a zero norm or a zero probability weight is skipped, and tau is
        # the min over what is left -- the skipped block must not contribute at all
        self.assertAlmostEqual(
            _spdhg_tau_from_sigma([1., 1.], [0., 2.], [0.5, 0.5], 0.99), 0.99*0.5/4)
        self.assertAlmostEqual(
            _spdhg_tau_from_sigma([1., 1.], [2., 2.], [0., 0.5], 0.99), 0.99*0.5/4)

    def test_tau_is_the_minimising_block(self):
    
        rho = 0.99
        norms = self.NORM_SCALINGS
        probs = [0.19] + [0.09]*(self.subsets - 1)
        sigma = [1.]*self.subsets
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=(None, sigma),
                      sampler=Sampler.random_with_replacement(self.subsets, prob=probs, seed=42))

        candidates = [rho*pi / (si * ni**2) for pi, ni, si
                      in zip(spdhg._prob_weights, norms, sigma)]
        self.assertAlmostEqual(spdhg.tau, min(candidates))

        self.assertEqual(candidates.index(min(candidates)), self.subsets - 1)

    def test_sigma_from_tau_uses_squared_norms(self):

        rho = 0.99
        tau = 100.
        norms = self.NORM_SCALINGS
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=(tau, None))

        self.assertEqual(spdhg.tau, tau)
        for si, ni, pi in zip(spdhg.sigma, norms, spdhg._prob_weights):
            self.assertAlmostEqual(si, rho * pi / (tau * ni**2))

    def test_step_sizes_from_ratio(self):
        gamma = 3.7
        rho = 5.6
        rule = SPDHGStepSizesFromRatio(gamma,rho)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
        self.assertListEqual(
            spdhg.sigma, [gamma * rho / ni for ni in spdhg._norms])
        self.assertEqual(spdhg.tau, min([pi*rho / (si * ni**2) for pi, ni,
                                            si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))

    def test_step_sizes_from_ratio_invalid(self):
        # gamma and rho must both be positive scalars
        for gamma, rho in [(-1., 0.9), (0., 0.9), (1., -0.9), (1., 0.),
                           ([1., 2.], 0.9), (1., [1., 2.])]:
            with self.assertRaises(ValueError):
                SPDHG(f=self.F, g=self.G, operator=self.A,
                      step_size=SPDHGStepSizesFromRatio(gamma, rho))


class TestSPDHGBayesStepSize(CCPiTestClass):


    NORM_SCALINGS = [1., 2., 3., 4.]
    PROB_WEIGHTS = [0.4, 0.3, 0.2, 0.1]
    RHO = 0.99

    def setUp(self):
        self.subsets = 4
        ig = ImageGeometry(4, 4)
        self.data = ig.allocate('random', seed=3)
        # scaled identities, so the operator norms are exactly NORM_SCALINGS
        self.A = BlockOperator(*[s * IdentityOperator(ig)
                                 for s in self.NORM_SCALINGS])
        self.F = BlockFunction(*[L2NormSquared(b=self.data)
                                 for _ in range(self.subsets)])
        self.G = 0.025 * IndicatorBox(lower=0)

    @staticmethod
    def _gamma_from_step_sizes(algorithm):
        r"""Recover the gamma the rule settled on.

        With :math:`\sigma_i=\gamma\rho/\|K_i\|` the primal step size collapses to
        :math:`\tau=\min_i(p_i/(\gamma\|K_i\|))`, so gamma is recovered as
        :math:`\min_i(p_i/(\tau\|K_i\|))` independently of rho.
        """
        return min(pi / (algorithm.tau * ni) for pi, ni
                   in zip(algorithm._prob_weights, algorithm._norms))

    # ------------------------------------------------------------------
    # construction 
    # ------------------------------------------------------------------

    def test_init(self):
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[0.1, 2], n_initial_points=5, n_calls=10,
            n_iterations=10, seed=42, plot=False)
        self.assertEqual(rule.gamma_bounds, [0.1, 2])
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 10)
        self.assertEqual(rule.n_iterations, 10)
        self.assertEqual(rule.seed, 42)
        self.assertFalse(rule.plot)
        # state owned by the SPDHG subclass rather than the shared base class
        self.assertEqual(rule.rho, 0.99)
        self.assertIsNone(rule._pristine_sampler)

    def test_init_default(self):
        rule = SPDHGBayesOptimisationStepSize()
        self.assertIsNone(rule.gamma_bounds)
        self.assertEqual(rule.n_initial_points, 5)
        self.assertEqual(rule.n_calls, 20)
        # resolved from the number of operators at set-up, not at construction
        self.assertIsNone(rule.n_iterations)
        self.assertIsNone(rule.seed)
        self.assertFalse(rule.plot)
        self.assertEqual(rule.rho, 0.99)

    def test_init_invalid_gamma_bounds_length(self):
        with self.assertRaises(ValueError):
            SPDHGBayesOptimisationStepSize(gamma_bounds=[(0.1, 10)])
        with self.assertRaises(ValueError):
            SPDHGBayesOptimisationStepSize(gamma_bounds=[0.1, 1., 10.])

    def test_init_invalid_gamma_bounds_non_positive(self):
        # every other argument is left valid, so the ValueError can only come
        # from the bound under test
        for bounds in ([0., 1.1], [-1., 1.1], [0.1, -1.1], [0.1, 0.]):
            with self.subTest(gamma_bounds=bounds):
                with self.assertRaises(ValueError):
                    SPDHGBayesOptimisationStepSize(gamma_bounds=bounds)

    def test_init_invalid_gamma_bounds_not_increasing(self):
        # reversed or degenerate bounds give an empty log-gamma search space
        for bounds in ([6., 5.], [1., 1.]):
            with self.subTest(gamma_bounds=bounds):
                with self.assertRaises(ValueError):
                    SPDHGBayesOptimisationStepSize(gamma_bounds=bounds)

    def test_init_invalid_call_counts(self):
        for kwargs in (dict(n_initial_points=0), dict(n_initial_points=-1),
                       dict(n_initial_points=2.5), dict(n_calls=-5),
                       dict(n_calls=0), dict(n_calls=2.5),
                       # skopt cannot draw more initial points than it has calls
                       dict(n_initial_points=5, n_calls=3)):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SPDHGBayesOptimisationStepSize(**kwargs)

    def test_init_invalid_n_iterations(self):
        for n_iterations in (1, 0, -10, 4.5):
            with self.subTest(n_iterations=n_iterations):
                with self.assertRaises(ValueError):
                    SPDHGBayesOptimisationStepSize(n_iterations=n_iterations)

    def test_init_accepts_valid_call_counts(self):
        rule = SPDHGBayesOptimisationStepSize(
            n_initial_points=1, n_calls=1, n_iterations=2, gamma_bounds=[1e-3, 1e3])
        self.assertEqual(rule.n_calls, 1)
        self.assertEqual(rule.n_iterations, 2)
        self.assertIsNone(SPDHGBayesOptimisationStepSize(n_iterations=None).n_iterations)

    def test_default_n_iterations_scales_with_number_of_operators(self):
        rule = SPDHGBayesOptimisationStepSize()
        self.assertEqual(
            rule._default_n_iterations(SimpleNamespace(_norms=[1.] * 7)), 70)
        
        
    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_defaults_are_resolved_from_the_problem_at_set_up(self):
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=2, n_calls=3,
            n_iterations=None, seed=42)
        SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
              sampler=Sampler.random_with_replacement(
                  self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        # n_iterations follows the number of operators
        self.assertEqual(rule.n_iterations, 10 * self.subsets)
        # and the bounds are derived from the objective at the zero image and the operator norm
        zero_x = self.A.domain_geometry().allocate(0)
        ratio = np.sqrt(self.F(self.A.direct(zero_x)) +
                        self.G(zero_x)) / self.A.norm()
        bounds = rule.gamma_bounds

        self.assertAlmostEqual(bounds[0] / (1e-5 / ratio), 1.)
        self.assertAlmostEqual(bounds[1] / (1e5 / ratio), 1.)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_default_gamma_bounds_when_f_vanishes_at_zero(self):

        f = BlockFunction(*[L1Norm() for _ in range(self.subsets)])
        g = L2NormSquared(b=self.data)
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=2, n_calls=3,
            n_iterations=2, seed=42)
        SPDHG(f=f, g=g, operator=self.A, step_size=rule,
              sampler=Sampler.random_with_replacement(
                  self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        zero_x = self.A.domain_geometry().allocate(0)
        self.assertEqual(f(self.A.direct(zero_x)), 0.)
        ratio = np.sqrt(g(zero_x)) / self.A.norm()
        bounds = rule.gamma_bounds
        self.assertTrue(np.all(np.isfinite(bounds)))
        self.assertAlmostEqual(bounds[0] / (1e-5 / ratio), 1.)
        self.assertAlmostEqual(bounds[1] / (1e5 / ratio), 1.)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_default_gamma_bounds_raises_if_objective_vanishes_at_zero(self):
        f = BlockFunction(*[L1Norm() for _ in range(self.subsets)])
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=None, n_initial_points=2, n_calls=3,
            n_iterations=2, seed=42)
        with self.assertRaises(ValueError):
            SPDHG(f=f, g=self.G, operator=self.A, step_size=rule,
                  sampler=Sampler.random_with_replacement(
                      self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        
    # ------------------------------------------------------------------
    # the gamma -> (tau, sigma) mapping
    # ------------------------------------------------------------------


    def test_step_sizes_from_gamma(self):
        rule = SPDHGBayesOptimisationStepSize()
        algorithm = SimpleNamespace(_norms=self.NORM_SCALINGS,
                                    _prob_weights=self.PROB_WEIGHTS)
        gamma = 2.5
        tau, sigma = rule._step_sizes_from_gamma(algorithm, gamma)

        expected_sigma = [gamma * self.RHO / ni for ni in self.NORM_SCALINGS]
        self.assertEqual(len(sigma), len(self.NORM_SCALINGS))
        for actual, expected in zip(sigma, expected_sigma):
            self.assertAlmostEqual(actual, expected)

        expected_tau = min(self.RHO * pi / (si * ni ** 2) for pi, ni, si
                           in zip(self.PROB_WEIGHTS, self.NORM_SCALINGS, expected_sigma))
        self.assertAlmostEqual(tau, expected_tau)
        self.assertAlmostEqual(tau, 0.01)

    # ------------------------------------------------------------------
    # against the real optimiser
    # ------------------------------------------------------------------

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_step_sizes_match_the_gamma_the_optimiser_returned(self):
        # bounds force gamma to be very close to 2.5, so the rule's step sizes are known in advance
        gamma = 2.5
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[gamma * (1 - 1e-6), gamma * (1 + 1e-6)],
            n_initial_points=2, n_calls=3, n_iterations=4, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        self.assertAlmostEqual(rule.gamma, gamma, places=5)
        self.assertAlmostEqual(self._gamma_from_step_sizes(spdhg), gamma, places=5)
        self.assertEqual(len(spdhg.sigma), self.subsets)
        for si, ni in zip(spdhg.sigma, self.NORM_SCALINGS):
            self.assertAlmostEqual(si, gamma * self.RHO / ni, places=5)
        self.assertAlmostEqual(spdhg.tau, 0.01, places=5)

        # the rule and the algorithm must agree, and get_step_size keeps returning
        # what was installed
        self.assertAlmostEqual(spdhg.tau, rule.tau)
        tau, sigma = rule.get_step_size(spdhg)
        self.assertAlmostEqual(tau, rule.tau)
        self.assertListEqual(list(sigma), list(rule.sigma))

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_step_sizes_are_consistent_for_whichever_gamma_is_chosen(self):
        # Test returned gamma is in the right range and the step sizes are consistent with it. The optimiser is free to choose any gamma in the range, so the test cannot assert a particular value.    
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        gamma = self._gamma_from_step_sizes(spdhg)
        self.assertTrue(2. - 1e-9 <= gamma <= 3. + 1e-9,
                        "gamma {} outside the requested bounds".format(gamma))
        # the gamma the rule reports must be the one its step sizes were built from
        self.assertAlmostEqual(rule.gamma, gamma)

        products = [si * ni for si, ni in zip(spdhg.sigma, self.NORM_SCALINGS)]
        for product in products:
            self.assertAlmostEqual(product, gamma * self.RHO)

        self.assertAlmostEqual(
            spdhg.tau, min(self.RHO * pi / (si * ni ** 2) for pi, ni, si
                           in zip(self.PROB_WEIGHTS, self.NORM_SCALINGS, spdhg.sigma)))



    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_algorithm_state_is_clean_after_the_search(self):
        # the rule temporarily sets the algorithm's iteration and objective to score the trial, but must restore them to run the algorithm from scratch afterwards
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42),
                      update_objective_interval=4)


        self.assertEqual(spdhg.iteration, -1)
        self.assertEqual(spdhg.objective, [])

        self.assertEqual(spdhg.update_objective_interval, 4)
        self.assertNumpyArrayEqual(
            spdhg.x.as_array(), self.A.domain_geometry().allocate(0).as_array())

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_non_uniform_prob_weights_survive_the_search(self):
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        self.assertListEqual(list(spdhg._prob_weights), self.PROB_WEIGHTS)
        self.assertListEqual(list(spdhg._sampler.prob_weights), self.PROB_WEIGHTS)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_sampler_is_left_at_the_start_of_the_users_sequence(self):
        # each trial is handed a fresh copy of the caller's sampler, so once the
        # search is over the algorithm must be about to draw the same subsets it
        # would have drawn without any Bayesian optimisation at all -- not resume
        # from wherever the last trial left the generator
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        untouched = Sampler.random_with_replacement(
            self.subsets, prob=self.PROB_WEIGHTS, seed=42)
        self.assertListEqual([spdhg._sampler.next() for _ in range(12)],
                             [untouched.next() for _ in range(12)])

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_diverging_trials_are_penalised_instead_of_raising(self):
        # Trials by gp minimise that return NaN or Inf are scored as +Inf, so the optimiser can continue to search for a valid gamma. The algorithm must be left with finite step sizes.
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[1e55, 1e65], n_initial_points=2, n_calls=4,
            n_iterations=4, seed=1)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        self.assertTrue(np.isfinite(spdhg.tau))
        self.assertGreater(spdhg.tau, 0.)
        self.assertTrue(all(np.isfinite(si) and si > 0 for si in spdhg.sigma))

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_step_sizes_are_constant_during_the_run(self):
        #  gamma is chosen once, up front and not changed
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))

        tau, sigma = spdhg.tau, list(spdhg.sigma)
        spdhg.run(5, callbacks=[])
        self.assertEqual(spdhg.tau, tau)
        self.assertListEqual(list(spdhg.sigma), sigma)

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_seed_makes_the_search_reproducible(self):
        spdhgs = []
        for _ in range(2):
            rule = SPDHGBayesOptimisationStepSize(
                gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
                n_iterations=4, seed=42)
            spdhgs.append(SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                                sampler=Sampler.random_with_replacement(
                                    self.subsets, prob=self.PROB_WEIGHTS, seed=42)))

        first, second = spdhgs
        self.assertEqual(first.tau, second.tau)
        self.assertListEqual(list(first.sigma), list(second.sigma))

    def test_missing_skopt_raises_an_informative_importerror(self):
        # the one path that must behave *without* scikit-optimize, so the import
        # is blocked rather than the optimiser replaced
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[2., 3.], n_initial_points=2, n_calls=3,
            n_iterations=4, seed=42)
        with patch.dict(sys.modules, {'skopt': None}):
            with self.assertRaises(ImportError) as context:
                SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        self.assertIn('scikit-optimize', str(context.exception))

class TestSPDHGAdaptiveStepSize(CCPiTestClass):
    """Tests for the adaptive SPDHG step-size rules based on arXiv:2301.02511."""


    NORM_SCALINGS = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    PROB_WEIGHTS = [0.19] + [0.09]*9

    def setUp(self):
        self.subsets = len(self.NORM_SCALINGS)
        ig = ImageGeometry(4, 4)
        self.data = ig.allocate('random', seed=3)
        self.A = BlockOperator(*[s * IdentityOperator(ig)
                                 for s in self.NORM_SCALINGS])
        self.F = BlockFunction(*[L2NormSquared(b=self.data)
                                 for _ in range(self.subsets)])
        self.G = 0.025 * IndicatorBox(lower=0)
 

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
        self.assertAlmostEqual(rule.eta, 0.95)
        self.assertEqual(rule.delta, 1.5)
        self.assertIsNone(rule.s)
        self.assertTrue(rule.auto_stop)
        self.assertEqual(rule.auto_stop_patience, None)
        self.assertAlmostEqual(rule.alpha_tolerance, 1e-3)

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
        self.assertEqual(rule.auto_stop_patience, None)
        self.assertAlmostEqual(rule.alpha_tolerance, 1e-3)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            SPDHGAdaptiveStepSizeBalancing(initial_step_size=[1.0])
        with self.assertRaises(ValueError):
            SPDHGAdaptiveStepSizeAngle(initial_step_size=[1.0, 2.0, 3.0])


    def test_initial_step_size_defaults(self):
        rho = 0.99
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            with self.subTest(rule=RuleCls.__name__):
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

    def test_initial_step_size_sigma_from_tau(self):
        # tau given, sigma derived one entry per operator
        rho = 0.99
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            with self.subTest(rule=RuleCls.__name__):
                rule = RuleCls(initial_step_size=[0.5, None])
                spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
                self.assertEqual(spdhg.tau, 0.5)
                self.assertListEqual(spdhg.sigma, [rho * pi / (0.5 * ni**2) for ni, pi in
                                                   zip(spdhg._norms, spdhg._prob_weights)])

    def test_auto_stop_patience_default(self):
        # left as None until set-up, then ten times the number of operators
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            with self.subTest(rule=RuleCls.__name__):
                rule = RuleCls()
                self.assertIsNone(rule.auto_stop_patience)
                SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule)
                self.assertEqual(rule.auto_stop_patience, 10 * self.subsets)
            
    def test_balancing_default_s_is_operator_norm(self):
        # s=None is filled in lazily with ||A|| on the first rebalance
        rule = SPDHGAdaptiveStepSizeBalancing(initial_step_size=[1.0, 1.0],
                                              auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        self.assertIsNone(rule.s)
        rule.x_prev.fill(0.0)
        rule.v_tmp.fill(1.0)
        rule.y_prev[0].fill(1.0)
        rule._rebalance(spdhg, 0, spdhg._prob_weights[0],
                        spdhg.tau, list(spdhg.sigma))
        self.assertAlmostEqual(rule.s, spdhg.operator.norm())

    def test_balancing_adapts_step_sizes(self):
        # from a deliberately imbalanced ratio, the rule should rescale the step sizes
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[10.0, 0.001], auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        spdhg.run(20, verbose=0)
        # tau started far too large relative to sigma, so the rule must bring it down
        # (and push sigma up) 
        self.assertLess(spdhg.tau, 1.0)
        self.assertGreater(spdhg.sigma[0], 0.001)
        # check step sizes remain valid (positive scalar tau, positive sigma list)
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
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        start_ratio = spdhg.tau / spdhg.sigma[0]
        spdhg.run(20, verbose=0)
        end_ratio = spdhg.tau / spdhg.sigma[0]
        self.assertNotAlmostEqual(start_ratio, end_ratio)
        self.assertLess(rule.alpha, 1.0)
        self.assertGreater(spdhg.tau, 0)
        self.assertEqual(len(spdhg.sigma), self.subsets)
        self.assertTrue(all(s > 0 for s in spdhg.sigma))

    def test_products_preserved_under_rescaling(self):
        # rescaling multiplies tau by 1/f and every sigma_i by f, so tau*sigma_i is fixed
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            with self.subTest(rule=RuleCls.__name__):
                rule = RuleCls(initial_step_size=[10.0, 0.001], auto_stop=False)
                spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                              sampler=Sampler.random_with_replacement(
                                  self.subsets, prob=self.PROB_WEIGHTS, seed=42))
                prod0 = [spdhg.tau * si for si in spdhg.sigma]
                spdhg.run(15, verbose=0)
                prod1 = [spdhg.tau * si for si in spdhg.sigma]
                for a, b in zip(prod0, prod1):
                    self.assertAlmostEqual(a, b)

    def test_convergence_condition_preserved_under_rescaling(self):
        # preserving tau*sigma_i also preserves  sigma_i * tau * ||A_i||^2 <= p_i, the
        # condition the whole rule is only valid under
        for RuleCls in (SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle):
            for seed in (1, 8, 42):
                with self.subTest(rule=RuleCls.__name__, seed=seed):
                    rule = RuleCls(auto_stop=False)
                    spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                                  sampler=Sampler.random_with_replacement(
                                      self.subsets, prob=self.PROB_WEIGHTS, seed=seed))
                    spdhg.run(20, verbose=0)
                    for si, ni, pi in zip(spdhg.sigma, spdhg._norms, spdhg._prob_weights):
                        self.assertLessEqual(si * spdhg.tau * ni**2, pi + 1e-9)

    def test_auto_stop_freezes_and_frees(self):
        # force the "no change" branch every iteration by making the tolerance gate
        # impossible to pass; the rule should then stop after auto_stop_patience.
        rule = SPDHGAdaptiveStepSizeBalancing(auto_stop=True, auto_stop_patience=3)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.tolerance = 1e12
        self.assertTrue(rule.adaptive)
        spdhg.run(6, verbose=0)
        self.assertFalse(rule.adaptive)
        # buffers released on auto-stop
        self.assertFalse(hasattr(rule, 'x_prev'))
        # continuing to run after auto-stop keeps returning valid step sizes
        tau_frozen = spdhg.tau
        sigma_frozen = list(spdhg.sigma)
        spdhg.run(3, verbose=0)
        self.assertEqual(spdhg.tau, tau_frozen)
        self.assertListEqual(list(spdhg.sigma), sigma_frozen)

    def test_auto_stop_on_alpha_tolerance(self):
        # the second stop criterion: alpha decays below alpha_tolerance before the
        # patience counter could ever trip.
        rule = SPDHGAdaptiveStepSizeAngle(
            c=0.0, initial_alpha=1e-3, eta=0.5, alpha_tolerance=1e-3,
            auto_stop=True, auto_stop_patience=10000)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        spdhg.run(3, verbose=0)
        self.assertFalse(rule.adaptive)
        self.assertLess(rule.alpha, 1e-3)
        self.assertFalse(hasattr(rule, 'x_prev'))

    def test_alpha_tolerance_zero_disables_that_criterion(self):
        # alpha_tolerance=0 means only auto_stop_patience can stop the rule
        rule = SPDHGAdaptiveStepSizeAngle(
            c=0.0, initial_alpha=1e-6, eta=0.5, alpha_tolerance=0,
            auto_stop=True, auto_stop_patience=10000)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        spdhg.run(5, verbose=0)
        self.assertTrue(rule.adaptive)

    def test_auto_stop_disabled_keeps_adapting(self):
        rule = SPDHGAdaptiveStepSizeAngle(auto_stop=False, auto_stop_patience=2)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        spdhg.run(20, verbose=0)
        self.assertTrue(rule.adaptive)
        # buffers still allocated 
        self.assertIsNotNone(rule.x_prev)

    def test_balancing_rebalance_increases_tau(self):
        # _rebalance reads the residuals out of the rule's buffers, which the base class
        # would normally have filled from the last SPDHG iteration. Set them by hand:
        #   x_prev = 0     ->  A_0 x_prev = 0, so d depends on y_prev[0] alone
        #   v_tmp  = 1     ->  primal residual   v = ||v_tmp||_1            > 0
        #   y_prev[0] = 1  ->  dual residual     d = ||y_prev[0]||_1 / (p_0 sigma_0) > 0
        # s tiny then makes  v > s*delta*d  true: the primal is behind, so tau is boosted.
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[1.0, 1.0], initial_alpha=0.5, eta=0.9,
            delta=1.5, s=1e-12, auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(0.0)
        rule.v_tmp.fill(1.0)
        rule.y_prev[0].fill(1.0)

        tau, sigma = spdhg.tau, list(spdhg.sigma)
        tau_new, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], tau, sigma)
        self.assertTrue(changed)
        self.assertAlmostEqual(tau_new, tau / 0.5)            # tau /= (1 - alpha)
        for s_new, s_old in zip(sigma_new, sigma):
            self.assertAlmostEqual(s_new, s_old * 0.5)
        self.assertAlmostEqual(rule.alpha, 0.5 * 0.9)         # alpha decayed by eta
        self.assertEqual(rule.count, 0)

    def test_balancing_rebalance_decreases_tau(self):
        # same values as above (v > 0, d > 0), but s huge makes  v < s*d/delta  true:
        # the dual is behind, so tau is shrunk.
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[1.0, 1.0], initial_alpha=0.5, eta=0.9,
            delta=1.5, s=1e12, auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(0.0)
        rule.v_tmp.fill(1.0)
        rule.y_prev[0].fill(1.0)

        tau, sigma = spdhg.tau, list(spdhg.sigma)
        tau_new, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], tau, sigma)
        self.assertTrue(changed)
        self.assertAlmostEqual(tau_new, tau * 0.5)
        for s_new, s_old in zip(sigma_new, sigma):
            self.assertAlmostEqual(s_new, s_old / 0.5)
        self.assertAlmostEqual(rule.alpha, 0.5 * 0.9)

    def test_balancing_rebalance_inside_band_is_noop(self):
        # same values again (v > 0, d > 0), but delta enormous puts v and d inside the
        # band whatever their values, so the step sizes and alpha are left untouched.
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[1.0, 1.0], initial_alpha=0.5, eta=0.9,
            delta=1e12, s=1.0, auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(0.0)
        rule.v_tmp.fill(1.0)
        rule.y_prev[0].fill(1.0)

        tau, sigma = spdhg.tau, list(spdhg.sigma)
        tau_new, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], tau, sigma)
        self.assertFalse(changed)
        self.assertEqual(tau_new, tau)
        self.assertListEqual(list(sigma_new), sigma)
        self.assertAlmostEqual(rule.alpha, 0.5)

    def test_step_size_rule_output_reaches_the_algorithm(self):
        # the (tau, sigma) the rule returns must be what the algorithm goes on to use:
        # SPDHG rebinds self._tau, self._sigma from get_step_size on every iteration.
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[1.0, 1.0], initial_alpha=0.5, delta=1.5, s=1e-12,
            auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        returned = []
        unspied = rule.get_step_size

        def spy(algorithm):
            tau, sigma = unspied(algorithm)
            returned.append((tau, list(sigma)))
            return tau, sigma
        rule.get_step_size = spy

        spdhg.run(5, verbose=0)
        tau_last, sigma_last = returned[-1]
        self.assertEqual(spdhg.tau, tau_last)
        self.assertListEqual(list(spdhg.sigma), sigma_last)
        # and the rule did rebalance, rather than handing back its input untouched
        self.assertNotAlmostEqual(spdhg.tau, 1.0)
        self.assertNotAlmostEqual(spdhg.sigma[0], 1.0)

    def test_balancing_rebalance_does_not_mutate_sigma_argument(self):
       
        rule = SPDHGAdaptiveStepSizeBalancing(
            initial_step_size=[1.0, 1.0], initial_alpha=0.5, delta=1.5, s=1e-12,
            auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(0.0)
        rule.v_tmp.fill(1.0)
        rule.y_prev[0].fill(1.0)

        sigma = list(spdhg.sigma)
        before = list(sigma)
        _, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], spdhg.tau, sigma)
        self.assertTrue(changed)
        # the new values are carried by the return value ...
        self.assertListEqual(list(sigma_new), [s * 0.5 for s in before])
        # ... and the list passed in is untouched
        self.assertIsNot(sigma_new, sigma)
        self.assertListEqual(sigma, before)



    def test_angle_rebalance_aligned_increases_tau(self):
        # w = +1 >= c, the directions agree, so the primal step is increased
        rule = SPDHGAdaptiveStepSizeAngle(initial_step_size=[1.0, 1.0],
                                          initial_alpha=0.5, eta=0.9, c=0.999,
                                          auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(1.0)
        rule.v_tmp.fill(1.0)
        tau, sigma = spdhg.tau, list(spdhg.sigma)
        tau_new, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], tau, sigma)
        self.assertTrue(changed)
        self.assertAlmostEqual(tau_new, tau * 1.5)            # tau *= (1 + alpha)
        for s_new, s_old in zip(sigma_new, sigma):
            self.assertAlmostEqual(s_new, s_old / 1.5)
        self.assertAlmostEqual(rule.alpha, 0.5 * 0.9)

    def test_angle_rebalance_opposed_decreases_tau(self):
        # w = -1 < 0, the directions oppose, so the primal step is decreased
        rule = SPDHGAdaptiveStepSizeAngle(initial_step_size=[1.0, 1.0],
                                          initial_alpha=0.5, eta=0.9, c=0.999,
                                          auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(1.0)
        rule.v_tmp.fill(-1.0)
        tau, sigma = spdhg.tau, list(spdhg.sigma)
        tau_new, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], tau, sigma)
        self.assertTrue(changed)
        self.assertAlmostEqual(tau_new, tau / 1.5)
        for s_new, s_old in zip(sigma_new, sigma):
            self.assertAlmostEqual(s_new, s_old * 1.5)

    def test_angle_rebalance_between_thresholds_is_noop(self):
        # 0 <= w < c leaves the step sizes, and alpha, untouched
        rule = SPDHGAdaptiveStepSizeAngle(initial_step_size=[1.0, 1.0],
                                          initial_alpha=0.5, eta=0.9, c=0.999,
                                          auto_stop=False)
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(
                          self.subsets, prob=self.PROB_WEIGHTS, seed=42))
        rule.x_prev.fill(1.0)
        alternating = np.ones(rule.v_tmp.shape).ravel()
        alternating[1::2] = -1.0
        rule.v_tmp.fill(alternating.reshape(rule.v_tmp.shape))   # w == 0
        tau, sigma = spdhg.tau, list(spdhg.sigma)
        tau_new, sigma_new, changed = rule._rebalance(
            spdhg, 0, spdhg._prob_weights[0], tau, sigma)
        self.assertFalse(changed)
        self.assertEqual(tau_new, tau)
        self.assertAlmostEqual(rule.alpha, 0.5)



