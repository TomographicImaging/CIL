#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import warnings
from utils import has_cvxpy
import unittest
from os import unlink
from tempfile import NamedTemporaryFile

import numpy as np
import logging


from cil.framework import VectorData, ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry, BlockDataContainer, BlockGeometry

from cil.framework.labels import FillType

from cil.optimisation.utilities import ArmijoStepSizeRule, ConstantStepSize, Sampler, callbacks, Sensitivity
from cil.optimisation.algorithms.APGD import NesterovMomentum, ScalarMomentumCoefficient, ConstantMomentum
from cil.optimisation.operators import IdentityOperator
from cil.optimisation.operators import GradientOperator, BlockOperator, MatrixOperator


from cil.optimisation.functions import MixedL21Norm, BlockFunction, L1Norm, KullbackLeibler, IndicatorBox, LeastSquares, ZeroFunction, L2NormSquared, OperatorCompositionFunction, TotalVariation, SGFunction, SVRGFunction, SAGAFunction, SAGFunction, LSVRGFunction, ScaledFunction
from cil.optimisation.algorithms import Algorithm, GD, CGLS, SIRT, FISTA, ISTA, SPDHG, PDHG, LADMM, PD3O, PGD, APGD


from scipy.optimize import minimize, rosen

from cil.utilities import dataexample
from cil.utilities import noise as applynoise
from cil.optimisation.functions import Rosenbrock
from cil.utilities.quality_measures import mae, mse, psnr

import logging
from testclass import CCPiTestClass
from utils import has_astra

from unittest.mock import MagicMock

log = logging.getLogger(__name__)


if has_cvxpy:
    import cvxpy

from unittest.mock import MagicMock, patch
class TestGD(CCPiTestClass):
    def setUp(self):

        x0_1 = 1.1
        x0_2 = 1.1
        # x0_1 = 0.5
        # x0_2 = 0.5
        self.x0 = np.array([x0_1, x0_2])

        self.initial = VectorData(np.array(self.x0))
        method = 'Nelder-Mead'  # or "BFGS"
        # self.scipy_opt_low = minimize(rosen, self.x0, method=method, tol=1e-3, options={"maxiter":50})
        self.scipy_opt_high = minimize(
            rosen, self.x0, method=method, tol=1e-2)  # (1., 1.)
        # fixed (alpha=1, beta=100) same to Scipy, min at (alpha,alpha^2)
        self.f = Rosenbrock(alpha=1, beta=100)

    def test_GD(self):
        ig = ImageGeometry(12, 13, 14)
        initial = ig.allocate()
        b = ig.allocate('random_deprecated')
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)

        step_size = norm2sq.L / 3.

        alg = GD(initial=initial, f=norm2sq, step_size=step_size,
                 atol=1e-9, rtol=1e-6)
        alg.run(1000, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = GD(initial=initial, f=norm2sq, step_size=step_size,
                 atol=1e-9, rtol=1e-6, update_objective_interval=2)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        self.assertAlmostEqual(alg.get_last_objective(), 0)
        self.assertNotAlmostEqual(alg.loss[0], alg.loss[1])

    def test_update_interval_0(self):
        '''
        Checks that an algorithm runs with no problems when
        the update_objective interval is set to 0 and with
        verbose on / off
        '''
        ig = ImageGeometry(12, 13, 14)
        initial = ig.allocate()
        b = ig.allocate('random_deprecated')
        identity = IdentityOperator(ig)
        norm2sq = LeastSquares(identity, b)
        alg = GD(initial=initial,
                 f=norm2sq,
                 update_objective_interval=0,
                 atol=1e-9, rtol=1e-6)
        self.assertTrue(alg.update_objective_interval == 0)
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg.run(20, verbose=False)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

    def test_gd_step_size_init(self):
        gd = GD(initial=self.initial, f=self.f, step_size=0.002)
        self.assertEqual(gd.step_size_rule.step_size, 0.002)
        self.assertEqual(gd.step_size, 0.002)

        gd = GD(initial=self.initial, f=self.f)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e6)
        self.assertEqual(gd.step_size_rule.beta, 0.5)
        self.assertEqual(gd.step_size_rule.max_iterations, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))
        with self.assertRaises(TypeError):
            gd.step_size

        gd = GD(initial=self.initial,
                f=self.f, alpha=1e2, beta=0.25)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e2)
        self.assertEqual(gd.step_size_rule.beta, 0.25)
        self.assertEqual(gd.step_size_rule.max_iterations, np.ceil(
            2 * np.log10(1e2) / np.log10(2)))

        with self.assertRaises(TypeError):
            gd = GD(initial=self.initial, f=self.f,
                    step_size=0.1, step_size_rule=ConstantStepSize(0.5))

    def test_gd_constant_step_size_init(self):
        rule = ConstantStepSize(0.4)
        self.assertEqual(rule.step_size, 0.4)
        gd = GD(initial=self.initial,
                f=self.f, step_size=rule)
        self.assertEqual(gd.step_size_rule.step_size, 0.4)
        self.assertEqual(gd.step_size, 0.4)

    def test_gd_fixed_step_size_rosen(self):

        gd = GD(initial=self.initial, f=self.f, step_size=0.002,
                update_objective_interval=500)
        gd.run(3000, verbose=0)
        np.testing.assert_allclose(
            gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(
            gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)

    def test_armijo_step_size_init(self):

        rule = ArmijoStepSizeRule()
        self.assertEqual(rule.alpha_orig, 1e6)
        self.assertEqual(rule.beta, 0.5)
        self.assertEqual(rule.max_iterations, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))

        gd = GD(initial=self.initial,
                f=self.f, step_size=rule)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e6)
        self.assertEqual(gd.step_size_rule.beta, 0.5)
        self.assertEqual(gd.step_size_rule.max_iterations, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))

        rule = ArmijoStepSizeRule(5e5, 0.2, 5)
        self.assertEqual(rule.alpha_orig, 5e5)
        self.assertEqual(rule.beta, 0.2)
        self.assertEqual(rule.max_iterations, 5)

        gd = GD(initial=self.initial,
                f=self.f, step_size=rule)
        self.assertEqual(gd.step_size_rule.alpha_orig, 5e5)
        self.assertEqual(gd.step_size_rule.beta, 0.2)
        self.assertEqual(gd.step_size_rule.max_iterations, 5)

        with self.assertRaises(TypeError):
            gd.step_size

    def test_GDArmijo(self):
        ig = ImageGeometry(12, 13, 14)
        initial = ig.allocate()
        # b = initial.copy()
        # fill with random numbers
        # b.fill(np.random.random(initial.shape))
        b = ig.allocate('random_deprecated')
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)

        alg = GD(initial=initial, f=norm2sq)
        alg.run(100, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg = GD(initial=initial, f=norm2sq, update_objective_interval=2)
        self.assertTrue(alg.update_objective_interval==2)

        alg = GD(initial=initial, f=norm2sq,
                 update_objective_interval=2)
        self.assertTrue(alg.update_objective_interval == 2)

        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

    def test_gd_armijo_rosen(self):
        armj = ArmijoStepSizeRule(alpha=50, max_iterations=50, warmstart=False)
        gd = GD(initial=self.initial, f=self.f, step_size=armj,
                update_objective_interval=500)
        gd.run(3500, verbose=0)
        np.testing.assert_allclose(
            gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(
            gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)

    def test_gd_run_no_iterations(self):
        gd = GD(initial=self.initial, f=self.f, step_size=0.002)
        with self.assertRaises(ValueError):
            gd.run()

    def test_gd_run_infinite(self):
        gd = GD(initial=self.initial, f=self.f, step_size=0.002)
        with self.assertRaises(ValueError):
            gd.run(np.inf)

        class StopCallback(callbacks.Callback):
            def __init__(self):
                self.count = 0

            def __call__(self, algorithm):
                self.count += 1
                if self.count == 10:
                    raise StopIteration
        with self.assertWarns(UserWarning):
            gd.run(np.inf, callbacks=[StopCallback()])
        self.assertEqual(gd.iteration, 9)

    def test_gd_deprecate_atol_rtol(self):
        with self.assertWarns(DeprecationWarning):
            initial=VectorData(np.array([1.1,1.1]))
            b=VectorData(np.array([1.,1.]))
            A=IdentityOperator(b.geometry)
            gd = GD(initial=initial, f=LeastSquares(A,b ), step_size=1, atol=1, rtol=2)
            
        self.assertEqual(gd.rtol, 2.)
        self.assertEqual(gd.atol, 1.)
        gd.run(10)
        self.assertEqual(gd.iteration, 0)
        
class Test_APGD(CCPiTestClass):
    def setUp(self):
        self.ig = ImageGeometry(11,12,13)
        self.initial = self.ig.allocate(0)
        self.b = self.ig.allocate(None)
        self.b.fill(np.array(range(11*12*13)).reshape(13,12,11))
        self.identity = IdentityOperator(self.ig)

        self.f = OperatorCompositionFunction(L2NormSquared(b=self.b), self.identity)
        self.g= IndicatorBox(lower=0)
        
    def test_init_default(self):
        alg = APGD(initial=self.initial, f=self.f, g=self.g)
        assert isinstance(alg.momentum, NesterovMomentum) 
        assert isinstance(alg.step_size_rule, ConstantStepSize)
        self.assertEqual(alg.step_size_rule.step_size, 1/self.f.L)
        self.assertEqual(alg.update_objective_interval, 1)
        self.assertNumpyArrayAlmostEqual(alg.y.as_array(), self.initial.as_array())

    def test_init_momentum(self):
        alg = APGD(initial=self.initial, f=self.f, g=self.g, momentum=1)
        assert isinstance(alg.momentum, ConstantMomentum)
        self.assertEqual(alg.momentum(alg),1) 
        self.assertEqual(alg.momentum.momentum, 1)
        
        with self.assertRaises(TypeError):
            alg = APGD(initial=self.initial, f=self.f, g=self.g, momentum='banana')
        
    def test_init_step_size(self):
        alg = APGD(initial=self.initial, f=self.f, g=self.g, step_size=1.0)
        assert isinstance(alg.step_size_rule, ConstantStepSize)
        self.assertEqual(alg.step_size_rule.get_step_size(alg), 1.0)
        self.assertEqual(alg.step_size_rule.step_size, 1.0)
        
        with self.assertRaises(TypeError):
            alg = APGD(initial=self.initial, f=self.f, g=self.g, step_size='banana')
            
    def test_update_step(self):
        alg = APGD(initial=self.initial, f=self.f, g=self.g, step_size=0.3, momentum=0.5)
        y_0=self.initial.copy()
        x_0=self.g.proximal(y_0 - 0.3*self.f.gradient(y_0), 0.3)
        y_1=x_0 + 0.5*(x_0-y_0)
        alg.run(1)
        self.assertNumpyArrayAlmostEqual(alg.y.as_array(), y_1.as_array())
        
        x_1=self.g.proximal(y_1 - 0.3*self.f.gradient(y_1), 0.3)
        y_2=x_1 + 0.5*(x_1-x_0)
        alg.run(1)
        self.assertNumpyArrayAlmostEqual(alg.y.as_array(), y_2.as_array())
        
    def test_provable_convergence(self):
        alg = APGD(initial=self.initial, f=self.f, g=self.g)
        self.assertTrue(alg.is_provably_convergent())
        
        alg = APGD(initial=self.initial, f=self.f, g=self.g, momentum=0.5)
        with self.assertRaises(TypeError):
            alg.is_provably_convergent()
            
        alg = APGD(initial=self.initial, f=self.f, g=self.g, preconditioner=Sensitivity(self.identity))
        with self.assertRaises(NotImplementedError):
            alg.is_provably_convergent()
        
        
        
        

class TestFISTA(CCPiTestClass):
    def test_FISTA(self):
        ig = ImageGeometry(127, 139, 149)
        initial = ig.allocate(0)
        b = initial.copy()
        # fill with random numbers
        b.fill(np.random.random(initial.shape))
        initial = ig.allocate(FillType["RANDOM_DEPRECATED"])
        identity = IdentityOperator(ig)

        norm2sq = OperatorCompositionFunction(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt': False}
        log.info("initial objective %s", norm2sq(initial))

        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction())
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction(),
                    update_objective_interval=2)

        self.assertTrue(alg.update_objective_interval == 2)

        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        # Testing g=None
        alg = FISTA(initial=initial, f=norm2sq, g=None,
                    update_objective_interval=2)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        # Testing f=None
        alg = FISTA(initial=initial, f=None, g=L1Norm(b=b),
                    update_objective_interval=2)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        # Testing f and g is None
        with self.assertRaises(ValueError):
            alg = FISTA(initial=initial, f=None, g=None,
                        update_objective_interval=2)

    def test_FISTA_update(self):

        # setup linear system to solve
        np.random.seed(10)
        n = 50
        m = 500

        A = np.random.uniform(0, 1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1 *
             np.random.randn(m)).astype('float32')

        Aop = MatrixOperator(A)
        bop = VectorData(b)

        f = LeastSquares(Aop, b=bop, c=0.5)
        g = ZeroFunction()

        ig = Aop.domain

        initial = ig.allocate()

        # ista run 10 iteration
        tmp_initial = ig.allocate()
        fista = FISTA(initial=tmp_initial, f=f, g=g)
        fista.run(1)

        # fista update method
        t_old = 1

        step_size = 1.0/f.L
        x_old = ig.allocate()
        y_old = ig.allocate()

        for _ in range(1):

            x = g.proximal(y_old - step_size *
                           f.gradient(y_old), tau=step_size)
            t = 0.5*(1 + np.sqrt(1 + 4*(t_old**2)))
            y = x + ((t_old-1)/t) * (x - x_old)

            x_old.fill(x)
            y_old.fill(y)
            t_old = t

        np.testing.assert_allclose(fista.solution.array, x.array, atol=1e-2)

        # check objective
        res1 = fista.objective[-1]
        res2 = f(x) + g(x)
        self.assertTrue(res1 == res2)

        tmp_initial = ig.allocate()
        fista1 = FISTA(initial=tmp_initial, f=f, g=g)
        self.assertTrue(fista1.is_provably_convergent())

        fista1 = FISTA(initial=tmp_initial, f=f, g=g,
                       step_size=30.0)
        self.assertFalse(fista1.is_provably_convergent())
        
            

    def test_FISTA_Norm2Sq(self):
        ig = ImageGeometry(127, 139, 149)
        b = ig.allocate(FillType["RANDOM_DEPRECATED"])
        # fill with random numbers
        initial = ig.allocate(FillType["RANDOM_DEPRECATED"])
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)

        opt = {'tol': 1e-4, 'memopt': False}
        log.info("initial objective %s", norm2sq(initial))
        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction())
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction(),
                    update_objective_interval=3)
        self.assertTrue(alg.update_objective_interval == 3)

        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

    def test_FISTA_catch_Lipschitz(self):
        ig = ImageGeometry(127, 139, 149)
        initial = ImageData(geometry=ig)
        initial = ig.allocate()
        b = initial.copy()
        # fill with random numbers
        b.fill(np.random.random(initial.shape))
        initial = ig.allocate(FillType["RANDOM_DEPRECATED"])
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)
        log.info('Lipschitz %s', norm2sq.L)
        # norm2sq.L = None
        # norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        # norm2sq = OperatorCompositionFunction(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt': False}
        log.info("initial objective %s", norm2sq(initial))
        with self.assertRaises(TypeError):
            alg = FISTA(initial=initial, f=L1Norm(), g=ZeroFunction())




class testISTA(CCPiTestClass):

    def setUp(self):

        np.random.seed(10)
        n = 50
        m = 500

        A = np.random.uniform(0, 1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1 *
             np.random.randn(m)).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b)

        self.f = LeastSquares(self.Aop, b=self.bop, c=0.5)
        self.g = ZeroFunction()
        self.h = L1Norm()

        self.ig = self.Aop.domain

        self.initial = self.ig.allocate()

    def tearDown(self):
        pass

    def test_signature(self):

        # check required arguments (initial, f, g)
        with np.testing.assert_raises(TypeError):
            ista = ISTA(f=self.f, g=self.g)

        with np.testing.assert_raises(TypeError):
            ista = ISTA(initial=self.initial, f=self.f)

        with np.testing.assert_raises(TypeError):
            ista = ISTA(initial=self.initial, g=self.g)

        # ista no step-size
        ista = ISTA(initial=self.initial, f=self.f, g=self.g)
        np.testing.assert_equal(ista.step_size, 0.99*2./self.f.L)

        # ista step-size
        tmp_step_size = 10.
        ista = ISTA(initial=self.initial, f=self.f,
                    g=self.g, step_size=tmp_step_size)
        np.testing.assert_equal(ista.step_size, tmp_step_size)

        # check initialisation
        self.assertTrue(id(ista.x) != id(ista.initial))
        self.assertTrue(id(ista.x_old) != id(ista.initial))

    def test_update(self):

        tmp_initial = self.ig.allocate()
        ista = ISTA(initial=tmp_initial, f=self.f, g=self.g)
        ista.run(1)

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        for _ in range(1):
            x = ista.g.proximal(x_old - (0.99*2/ista.f.L)
                                * ista.f.gradient(x_old), (1./ista.f.L))
            x_old.fill(x)

        np.testing.assert_allclose(ista.solution.array, x.array, atol=1e-2)

        # check objective
        res1 = ista.objective[-1]
        res2 = self.f(x) + self.g(x)
        self.assertTrue(res1 == res2)
        
    def test_update_pgd(self):
        
        tmp_initial = self.ig.allocate()
        pgd = PGD(initial=tmp_initial, f=self.f, g=self.g)
        pgd.run(1)

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        for _ in range(1):
            x = pgd.g.proximal(x_old - (0.99*2/pgd.f.L)
                                * pgd.f.gradient(x_old), (1./pgd.f.L))
            x_old.fill(x)

        np.testing.assert_allclose(pgd.solution.array, x.array, atol=1e-2)

        # check objective
        res1 = pgd.objective[-1]
        res2 = self.f(x) + self.g(x)
        self.assertTrue(res1 == res2)

    def test_update_g_none(self):

        
        tmp_initial = self.ig.allocate()
        ista = ISTA(initial=tmp_initial, f=self.f, g=None)
        ista.run(1)

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        x = ista.g.proximal(x_old - (0.99*2/ista.f.L) *
                            ista.f.gradient(x_old), (1./ista.f.L))
        
        #Check if g is None, the proximal operator is the identity and thus GD = ISTA when g is None
        x2 = x_old - (0.99*2/ista.f.L) * ista.f.gradient(x_old)

        np.testing.assert_allclose(ista.solution.array, x.array, atol=1e-2)
        np.testing.assert_allclose(x.array, x2.array, atol=1e-2)

        # check objective
        res1 = ista.objective[-1]
        res2 = self.f(x) + self.g(x)
        self.assertTrue(res1 == res2)

    def test_update_f_none(self):

        # ista run 1 iteration
        tmp_initial = self.ig.allocate()
        ista = ISTA(initial=tmp_initial, f=None, g=self.h)
        ista.run(1)

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        for _ in range(1):
            x = ista.g.proximal(x_old, ista.step_size)
            x_old.fill(x)

        np.testing.assert_allclose(ista.solution.array, x.array, atol=1e-2)

        # check objective
        res1 = ista.objective[-1]
        res2 = self.h(x)
        self.assertTrue(res1 == res2)

    def test_f_and_g_none(self):
        tmp_initial = self.ig.allocate()
        with self.assertRaises(ValueError):
            ista = ISTA(initial=tmp_initial, f=None, g=None)

    def test_provable_condition(self):

        tmp_initial = self.ig.allocate()
        ista1 = ISTA(initial=tmp_initial, f=self.f, g=self.g)
        self.assertTrue(ista1.is_provably_convergent())

        ista1 = ISTA(initial=tmp_initial, f=self.f, g=self.g,
                     step_size=30.0)
        self.assertFalse(ista1.is_provably_convergent())

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_with_cvxpy(self):

        ista = ISTA(initial=self.initial, f=self.f,
                    g=self.g)
        ista.run(2000, verbose=0)

        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize(
            0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4)

        np.testing.assert_allclose(p.value, ista.objective[-1], atol=1e-3)
        np.testing.assert_allclose(
            u_cvxpy.value, ista.solution.array, atol=1e-3)

 
    def test_ISTA_PGD_alias(self):

        with patch('cil.optimisation.algorithms.ISTA.set_up', MagicMock(return_value=None)) as mock_method:

            alg = PGD(initial=self.initial, f=self.f, g=self.g, step_size=4)
            self.assertEqual(alg._step_size, 4)
            mock_method.assert_called_once_with(initial=self.initial, f=self.f, g=self.g, step_size=4, preconditioner=None)



class TestCGLS(CCPiTestClass):
    def setUp(self):
        self.ig = ImageGeometry(10, 2)
        np.random.seed(2)
        self.initial = self.ig.allocate(1.)
        self.data = self.ig.allocate('random_deprecated')
        self.operator = IdentityOperator(self.ig)
        self.alg = CGLS(initial=self.initial, operator=self.operator, data=self.data,
                        update_objective_interval=2)

    # can be deprecated when tolerance is deprecated in CGLS
    def test_initialization_with_default_tolerance(self):

        self.assertEqual(self.alg.tolerance, 0)

    # can be deprecated when tolerance is deprecated in CGLS
    def test_initialization_with_custom_tolerance(self):
        with self.assertWarns(DeprecationWarning):
            alg = CGLS(initial=self.initial, operator=self.operator,
                       data=self.data, tolerance=0.01)
        self.assertEqual(alg.tolerance, 0.01)

    def test_set_up(self):
        # Test the set_up method

        self.alg.set_up(initial=self.initial,
                        operator=self.operator, data=self.data)

        # Check if internal variables are set up correctly
        self.assertNumpyArrayEqual(
            self.alg.x.as_array(), self.initial.as_array())
        self.assertEqual(self.alg.operator, self.operator)
        self.assertNumpyArrayEqual(
            self.alg.r.as_array(), (self.data-self.initial).as_array())
        self.assertNumpyArrayEqual(
            self.alg.s.as_array(), (self.data-self.initial).as_array())
        self.assertNumpyArrayEqual(
            self.alg.p.as_array(), (self.data-self.initial).as_array())
        self.assertTrue(self.alg.configured)

    def test_update(self):

        self.alg.set_up(initial=self.mock_initial,
                        operator=self.mock_operator, data=self.mock_data)

        self.alg.update()

        self.mock_operator.direct.assert_called_with(
            self.mock_data, out=self.alg.q)
        self.mock_operator.adjoint.assert_called_with(
            self.alg.r, out=self.alg.s)

    def test_convergence(self):

        self.alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            self.alg.x.as_array(), self.data.as_array())

    # can be deprecated when tolerance is deprecated in CGLS
    def test_should_stop_flag_false(self):
        # Mocking norms to ensure tolerance isn't reached
        self.alg.run(2)
        self.alg.norms = 10
        self.alg.norms0 = 99
        self.alg.tolerance = 0.1
        self.alg.normx = 0.1

        self.assertFalse(self.alg.flag())

    # can be deprecated when tolerance is deprecated in CGLS
    def test_should_stop_flag_true(self):
        # Mocking norms to ensure tolerance is reached
        self.alg.run(4)
        self.alg.norms = 1
        self.alg.norms0 = 10
        self.alg.tolerance = 0.1
        self.alg.normx = 10

        self.assertTrue(self.alg.flag())

    # can be deprecated when tolerance is deprecated in CGLS
    def test_tolerance_reached_immediately(self):
        alg = CGLS(initial=self.operator.domain_geometry().allocate(
            0), operator=self.operator, data=self.operator.domain_geometry().allocate(0))
        alg.run(2)

    def test_update_objective(self):
        # Mocking squared_norm to return a finite value

        self.alg.r = MagicMock()
        self.alg.r.squared_norm.return_value = 1.0

        self.alg.update_objective()

        # Ensure the loss list is updated
        self.assertEqual(self.alg.loss, [1.0])

    def test_update_objective_with_nan_raises_stop_iteration(self):
        # Mocking squared_norm to return NaN
        self.alg.r = MagicMock()
        self.alg.r.squared_norm.return_value = np.nan

        with self.assertRaises(StopIteration):
            self.alg.update_objective()

    def test_update(self):
        self.alg.gamma = 4

        self.alg.update()
        norm = (self.data-self.initial).squared_norm()
        alpha = 4/norm
        self.assertNumpyArrayEqual(self.alg.x.as_array(
        ), (self.initial+alpha*(self.data-self.initial)).as_array())
        self.assertNumpyArrayEqual(self.alg.r.as_array(
        ), (self.data - self.initial-alpha*(self.data-self.initial)).as_array())
        beta = ((self.data - self.initial-alpha *
                (self.data-self.initial)).norm()**2)/4
        self.assertNumpyArrayEqual(self.alg.p.as_array(), ((
            self.data - self.initial-alpha*(self.data-self.initial))+beta*(self.data-self.initial)).as_array())


class TestPDHG(CCPiTestClass):

    def test_PDHG_Denoising(self):
        # adapted from demo PDHG_TV_Color_Denoising.py in CIL-Demos repository
        data = dataexample.PEPPERS.get(size=(256, 256))
        ig = data.geometry
        ag = ig

        which_noise = 0
        # Create noisy data.
        noises = ['gaussian', 'poisson', 's&p']
        dnoise = noises[which_noise]

        def setup(data, dnoise):
            if dnoise == 's&p':
                n1 = applynoise.saltnpepper(
                    data, salt_vs_pepper=0.9, amount=0.2, seed=10)
            elif dnoise == 'poisson':
                scale = 5
                n1 = applynoise.poisson(data.as_array()/scale, seed=10)*scale
            elif dnoise == 'gaussian':
                n1 = applynoise.gaussian(data.as_array(), seed=10)
            else:
                raise ValueError('Unsupported Noise ', noise)
            noisy_data = ig.allocate()
            noisy_data.fill(n1)

            # Regularisation Parameter depending on the noise distribution
            if dnoise == 's&p':
                alpha = 0.8
            elif dnoise == 'poisson':
                alpha = 1
            elif dnoise == 'gaussian':
                alpha = .3
                # fidelity
            if dnoise == 's&p':
                g = L1Norm(b=noisy_data)
            elif dnoise == 'poisson':
                g = KullbackLeibler(b=noisy_data)
            elif dnoise == 'gaussian':
                g = 0.5 * L2NormSquared(b=noisy_data)
            return noisy_data, alpha, g

        noisy_data, alpha, g = setup(data, dnoise)
        operator = GradientOperator(
            ig, correlation=GradientOperator.CORRELATION_SPACE, backend='numpy')

        f1 = alpha * MixedL21Norm()

        # Compute operator Norm
        normK = operator.norm()

        # Primal & dual stepsizes
        sigma = 1
        tau = 1/(sigma*normK**2)

        # Setup and run the PDHG algorithm
        pdhg1 = PDHG(f=f1, g=g, operator=operator, tau=tau, sigma=sigma)
        pdhg1.update_objective_interval = 200
        pdhg1.run(1000, verbose=0)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        log.info("RMSE %F", rmse)
        self.assertLess(rmse, 2e-4)

        which_noise = 1
        noise = noises[which_noise]
        noisy_data, alpha, g = setup(data, noise)
        operator = GradientOperator(
            ig, correlation=GradientOperator.CORRELATION_SPACE, backend='numpy')

        f1 = alpha * MixedL21Norm()

        # Compute operator Norm
        normK = operator.norm()

        # Primal & dual stepsizes
        sigma = 1
        tau = 1/(sigma*normK**2)

        # Setup and run the PDHG algorithm
        pdhg1 = PDHG(f=f1, g=g, operator=operator, tau=tau, sigma=sigma,
                     update_objective_interval=200)

        pdhg1.run(1000, verbose=0)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        log.info("RMSE %f", rmse)
        self.assertLess(rmse, 2e-4)

        which_noise = 2
        noise = noises[which_noise]
        noisy_data, alpha, g = setup(data, noise)
        operator = GradientOperator(
            ig, correlation=GradientOperator.CORRELATION_SPACE, backend='numpy')

        f1 = alpha * MixedL21Norm()

        # Compute operator Norm
        normK = operator.norm()

        # Primal & dual stepsizes
        sigma = 1
        tau = 1/(sigma*normK**2)

        # Setup and run the PDHG algorithm
        pdhg1 = PDHG(f=f1, g=g, operator=operator, tau=tau, sigma=sigma)
        pdhg1.update_objective_interval = 200
        pdhg1.run(1000, verbose=0)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        log.info("RMSE %f", rmse)
        self.assertLess(rmse, 2e-4)

    def test_PDHG_step_sizes(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random_deprecated')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = 3*IdentityOperator(ig)

        # check if sigma, tau are None
        pdhg = PDHG(f=f, g=g, operator=operator)
        self.assertAlmostEqual(pdhg.sigma, 1./operator.norm())
        self.assertAlmostEqual(pdhg.tau, 1./operator.norm())

        # check if sigma is negative
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        sigma=-1)

        # check if tau is negative
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, tau=-1)

        # check if tau is None
        sigma = 3.0
        pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma)
        self.assertAlmostEqual(pdhg.sigma, sigma)
        self.assertAlmostEqual(pdhg.tau, 1./(sigma * operator.norm()**2))

        # check if sigma is None
        tau = 3.0
        pdhg = PDHG(f=f, g=g, operator=operator, tau=tau)
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, 1./(tau * operator.norm()**2))

        # check if sigma/tau are not None
        tau = 1.0
        sigma = 1.0
        pdhg = PDHG(f=f, g=g, operator=operator, tau=tau,
                    sigma=sigma)
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, sigma)

        # check sigma/tau as arrays, sigma wrong shape
        ig1 = ImageGeometry(2, 2)
        sigma = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        sigma=sigma)

        # check sigma/tau as arrays, tau wrong shape
        tau = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, tau=tau)

        # check sigma not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        sigma="sigma")

        # check tau not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        tau="tau")

        # check warning message if condition is not satisfied
        sigma = 4/operator.norm()
        tau = 1/3
        with self.assertWarnsRegex(UserWarning, "Convergence criterion"):
            pdhg = PDHG(f=f, g=g, operator=operator, tau=tau,
                        sigma=sigma)

        # check no warning message if check convergence is false
        sigma = 4/operator.norm()
        tau = 1/3
        with warnings.catch_warnings(record=True) as warnings_log:
            pdhg = PDHG(f=f, g=g, operator=operator, tau=tau,
                        sigma=sigma, check_convergence=False)
        self.assertEqual(warnings_log, [])

        # check no warning message if condition is satisfied
        sigma = 1/operator.norm()
        tau = 1/3
        with warnings.catch_warnings(record=True) as warnings_log:
            pdhg = PDHG(f=f, g=g, operator=operator, tau=tau,
                        sigma=sigma)
        self.assertEqual(warnings_log, [])

    def test_PDHG_strongly_convex_gamma_g(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random_deprecated')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0

        pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                    gamma_g=0.5)
        pdhg.run(1, verbose=0)
        self.assertAlmostEqual(
            pdhg.theta, 1.0 / np.sqrt(1 + 2 * pdhg.gamma_g * tau))
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
        data = ig.allocate('random_deprecated')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0

        pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                    gamma_fconj=0.5)
        pdhg.run(1, verbose=0)
        self.assertEqual(pdhg.theta, 1.0 / np.sqrt(1 +
                         2 * pdhg.gamma_fconj * sigma))
        self.assertEqual(pdhg.tau, tau / pdhg.theta)
        self.assertEqual(pdhg.sigma, sigma * pdhg.theta)
        pdhg.run(4, verbose=0)
        self.assertNotEqual(pdhg.sigma, sigma)
        self.assertNotEqual(pdhg.tau, tau)

        # check negative strongly convex constant
        try:
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        gamma_fconj=-0.5)
        except ValueError as ve:
            log.info(str(ve))

        # check strongly convex constant not a number
        try:
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        gamma_fconj="-0.5")
        except ValueError as ve:
            log.info(str(ve))

    def test_PDHG_strongly_convex_both_fconj_and_g(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random_deprecated')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        try:
            pdhg = PDHG(f=f, g=g, operator=operator,
                        gamma_g=0.5, gamma_fconj=0.5)
            pdhg.run(verbose=0)
        except ValueError as err:
            log.info(str(err))

    def test_pdhg_theta(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random_deprecated')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        pdhg = PDHG(f=f, g=g, operator=operator)
        self.assertEqual(pdhg.theta, 1.0)

        pdhg = PDHG(f=f, g=g, operator=operator, theta=0.5)
        self.assertEqual(pdhg.theta, 0.5)

        with self.assertRaises(ValueError):
            PDHG(f=f, g=g, operator=operator, theta=-0.5)

        with self.assertRaises(ValueError):
            PDHG(f=f, g=g, operator=operator, theta=5)


class TestSIRT(CCPiTestClass):

    def setUp(self):
        np.random.seed(10)
        # set up matrix, vectordata
        n, m = 50, 50

        A = np.random.uniform(0, 1, (m, n)).astype('float32')
        b = A.dot(np.random.randn(n))

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b)

        self.ig = self.Aop.domain

        self.initial = self.ig.allocate()

        # set up with linear operator
        self.ig2 = ImageGeometry(3, 4, 5)
        self.initial2 = self.ig2.allocate(0.)
        self.b2 = self.ig2.allocate('random_deprecated')
        self.A2 = IdentityOperator(self.ig2)

    def tearDown(self):
        pass

    def test_set_up(self):

        initial = self.A2.domain_geometry().allocate(0)
        sirt = SIRT(initial=initial, operator=self.A2,
                    data=self.b2, lower=0, upper=1)

        # Test if set_up correctly configures the object
        self.assertTrue(sirt.configured)
        self.assertIsNotNone(sirt.x)
        self.assertIsNotNone(sirt.r)
        self.assertIsNotNone(sirt.constraint)
        self.assertEqual(sirt.constraint.lower, 0)
        self.assertEqual(sirt.constraint.upper, 1)

        constraint = IndicatorBox(lower=0, upper=1)
        sirt = SIRT(initial=None, operator=self.A2,
                    data=self.b2, constraint=constraint)

        # Test if set_up correctly configures the object with constraint
        self.assertTrue(sirt.configured)
        self.assertEqual(sirt.constraint, constraint)

        with self.assertRaises(ValueError) as context:
            sirt = SIRT(initial=None, operator=None, data=self.b2)
            self.assertEqual(str(context.exception),
                             'You must pass an `operator` to the SIRT algorithm')

        with self.assertRaises(ValueError) as context:
            sirt = SIRT(initial=None, operator=self.A2, data=None)
            self.assertEqual(str(context.exception),
                             'You must pass `data` to the SIRT algorithm')
        with self.assertRaises(ValueError) as context:
            sirt = SIRT(initial=None, operator=None, data=None)
            self.assertEqual(str(context.exception),
                             'You must pass an `operator` and `data` to the SIRT algorithm')

        sirt = SIRT(initial=None, operator=self.A2, data=self.b2)
        self.assertTrue(sirt.configured)
        self.assertIsInstance(sirt.x, ImageData)
        self.assertTrue((sirt.x.as_array() == 0).all())

        initial = self.A2.domain_geometry().allocate(1)
        sirt = SIRT(initial=initial, operator=self.A2, data=self.b2)
        self.assertTrue(sirt.configured)
        self.assertIsInstance(sirt.x, ImageData)
        self.assertTrue((sirt.x.as_array() == 1).all())

    def test_update(self):
        # sirt run 5 iterations
        tmp_initial = self.ig.allocate()
        sirt = SIRT(initial=tmp_initial, operator=self.Aop,
                    data=self.bop)
        sirt.run(5)
        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        for _ in range(5):
            x = x_old + sirt.D * \
                (sirt.operator.adjoint(sirt.M*(sirt.data - sirt.operator.direct(x_old))))
            x_old.fill(x)

        np.testing.assert_allclose(sirt.solution.array, x.array, atol=1e-2)

    def test_update_constraints(self):
        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2)
        alg.run(20, verbose=0)
        np.testing.assert_array_almost_equal(alg.x.array, self.b2.array)

        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2,  upper=0.3)
        alg.run(20, verbose=0)
        np.testing.assert_almost_equal(alg.solution.max(), 0.3)

        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2, lower=0.7)
        alg.run(20, verbose=0)
        np.testing.assert_almost_equal(alg.solution.min(), 0.7)

        alg = SIRT(initial=self.initial2, operator=self.A2, data=self.b2,
                   constraint=IndicatorBox(lower=0.1, upper=0.3))
        alg.run(20, verbose=0)
        np.testing.assert_almost_equal(alg.solution.max(), 0.3)
        np.testing.assert_almost_equal(alg.solution.min(), 0.1)

    def test_SIRT_relaxation_parameter(self):
        tmp_initial = self.ig.allocate()
        alg = SIRT(initial=tmp_initial, operator=self.Aop,
                   data=self.bop)

        with self.assertRaises(ValueError):
            alg.set_relaxation_parameter(0)

        with self.assertRaises(ValueError):
            alg.set_relaxation_parameter(2)

        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2)
        alg.set_relaxation_parameter(0.5)

        self.assertEqual(alg.relaxation_parameter, 0.5)

        alg.run(20, verbose=0)
        np.testing.assert_array_almost_equal(alg.x.array, self.b2.array)

        np.testing.assert_almost_equal(0.5 * alg.D.array, alg._Dscaled.array)

    def test_SIRT_nan_inf_values(self):
        Aop_nan_inf = self.Aop
        Aop_nan_inf.A[0:10, :] = 0.
        Aop_nan_inf.A[:, 10:20] = 0.

        tmp_initial = self.ig.allocate()
        sirt = SIRT(initial=tmp_initial, operator=Aop_nan_inf,
                    data=self.bop)

        self.assertFalse(np.any(sirt.M == np.inf))
        self.assertFalse(np.any(sirt.D == np.inf))

    def test_SIRT_remove_nan_or_inf_with_BlockDataContainer(self):
        np.random.seed(10)
        # set up matrix, vectordata
        n, m = 50, 50

        A = np.random.uniform(0, 1, (m, n)).astype('float32')
        b = A.dot(np.random.randn(n))

        A[0:10, :] = 0.
        A[:, 10:20] = 0.
        Aop = BlockOperator(MatrixOperator(A*1), MatrixOperator(A*2))
        bop = BlockDataContainer(VectorData(b*1), VectorData(b*2))

        ig = BlockGeometry(self.ig.copy(), self.ig.copy())
        tmp_initial = ig.allocate()

        sirt = SIRT(initial=tmp_initial, operator=Aop,
                    data=bop)
        for el in sirt.M.containers:
            self.assertFalse(np.any(el == np.inf))

        self.assertFalse(np.any(sirt.D == np.inf))

    def test_SIRT_with_TV(self):
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128, 128))
        ig = data.geometry
        A = IdentityOperator(ig)
        constraint = TotalVariation(warm_start=False)
        initial = ig.allocate('random_deprecated', seed=5)
        sirt = SIRT(initial=initial, operator=A, data=data,
                    constraint=constraint)
        sirt.run(2, verbose=0)
        f = LeastSquares(A, data, c=0.5)
        fista = FISTA(initial=initial, f=f, g=constraint)
        fista.run(100, verbose=0)
        self.assertNumpyArrayAlmostEqual(fista.x.as_array(), sirt.x.as_array())

    def test_SIRT_with_TV_warm_start(self):
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128, 128))
        ig = data.geometry
        A = IdentityOperator(ig)
        constraint = 1e6*TotalVariation(warm_start=True, max_iteration=100)
        initial = ig.allocate('random_deprecated', seed=5)
        sirt = SIRT(initial=initial, operator=A, data=data,
                    constraint=constraint)
        sirt.run(25, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            sirt.x.as_array(), ig.allocate(0.25).as_array(), 3)


class TestSPDHG(CCPiTestClass):
    def setUp(self):
        self.subsets = 10

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get(size=(16, 16))

        partitioned_data = data.partition(self.subsets, 'sequential')
        self.A = BlockOperator(
            *[IdentityOperator(partitioned_data[i].geometry) for i in range(self.subsets)])
        self.A2 = BlockOperator(
            *[IdentityOperator(partitioned_data[i].geometry) for i in range(self.subsets)])

        # block function
        self.F = BlockFunction(*[L2NormSquared(b=partitioned_data[i])
                                for i in range(self.subsets)])
        alpha = 0.025
        self.G = alpha * IndicatorBox(lower=0)



    def test_SPDHG_defaults_and_setters(self):
        # Test SPDHG init with default values
        gamma = 1.
        rho = .99
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A)

        self.assertListEqual(spdhg._norms, [self.A.get_item(i, 0).norm()
                                            for i in range(self.subsets)])
        self.assertListEqual(spdhg._prob_weights, [
                             1/self.subsets] * self.subsets)
        self.assertEqual(spdhg._sampler._type, 'random_with_replacement')
        self.assertListEqual(
            spdhg.sigma, [rho / ni for ni in spdhg._norms])
        self.assertEqual(spdhg.tau, min([rho*pi / (si * ni**2) for pi, ni,
                                         si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))
        self.assertNumpyArrayEqual(
            spdhg.x.as_array(), self.A.domain_geometry().allocate(0).as_array())
        self.assertEqual(spdhg.update_objective_interval, 1)

        # Test SPDHG setters - "from ratio"
        gamma = 3.7
        rho = 5.6
        spdhg.set_step_sizes_from_ratio(gamma, rho)
        self.assertListEqual(
            spdhg.sigma, [gamma * rho / ni for ni in spdhg._norms])
        self.assertEqual(spdhg.tau, min([pi*rho / (si * ni**2) for pi, ni,
                                         si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))

        # Test SPDHG setters - set_step_sizes default values for sigma and tau
        gamma = 1.
        rho = .99
        spdhg.set_step_sizes()
        self.assertListEqual(
            spdhg.sigma, [rho / ni for ni in spdhg._norms])
        self.assertEqual(spdhg.tau, min([rho*pi / (si * ni**2) for pi, ni,
                                         si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))

        # Test SPDHG setters - set_step_sizes with sigma and tau
        spdhg.set_step_sizes(sigma=[1]*self.subsets, tau=100)
        self.assertListEqual(spdhg.sigma, [1]*self.subsets)
        self.assertEqual(spdhg.tau, 100)

        # Test SPDHG setters - set_step_sizes with sigma
        spdhg.set_step_sizes(sigma=[1]*self.subsets, tau=None)
        self.assertListEqual(spdhg.sigma, [1]*self.subsets)
        self.assertEqual(spdhg.tau, min([(rho*pi / (si * ni**2)) for pi, ni,
                                         si in zip(spdhg._prob_weights, spdhg._norms, spdhg.sigma)]))

        # Test SPDHG setters - set_step_sizes with tau
        spdhg.set_step_sizes(sigma=None, tau=100)
        self.assertListEqual(spdhg.sigma, [
                             gamma * rho*pi / (spdhg.tau*ni**2) for ni, pi in zip(spdhg._norms, spdhg._prob_weights)])
        self.assertEqual(spdhg.tau, 100)

    def test_spdhg_non_default_init(self):
        # Test SPDHG init with non-default values
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, sampler=Sampler.random_with_replacement(10, list(np.arange(1, 11)/55.)),
                      initial=self.A.domain_geometry().allocate(1), update_objective_interval=10)

        self.assertListEqual(spdhg._prob_weights,  list(np.arange(1, 11)/55.))
        self.assertNumpyArrayEqual(
            spdhg.x.as_array(), self.A.domain_geometry().allocate(1).as_array())
        self.assertEqual(spdhg.update_objective_interval, 10)

        # Test SPDHG setters - prob_weights and a sampler gives an error
        with self.assertRaises(ValueError):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, prob_weights=[1/(self.subsets-1)]*(
                self.subsets-1)+[0], sampler=Sampler.random_with_replacement(10, list(np.arange(1, 11)/55.)))

        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, prob_weights=[1/(self.subsets-1)]*(
            self.subsets-1)+[0],  initial=self.A.domain_geometry().allocate(1), update_objective_interval=10)

        self.assertListEqual(spdhg._prob_weights,  [
                             1/(self.subsets-1)]*(self.subsets-1)+[0])
        self.assertEqual(spdhg._sampler._type, 'random_with_replacement')

    def test_spdhg_sampler_gives_too_large_index(self):
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, sampler=Sampler.sequential(20),
                      initial=self.A.domain_geometry().allocate(1), update_objective_interval=10)
        with self.assertRaises(IndexError):
            spdhg.run(12)

    def test_spdhg_deprecated_vargs(self):
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, norms=[
                      1]*len(self.A), prob=[1/(self.subsets-1)]*(self.subsets-1)+[0])

        self.assertListEqual(self.A.get_norms_as_list(), [1]*len(self.A))
        self.assertListEqual(spdhg._norms, [1]*len(self.A))
        self.assertListEqual(spdhg._sampler.prob_weights, [
                             1/(self.subsets-1)]*(self.subsets-1)+[0])
        self.assertListEqual(spdhg._prob_weights, [
                             1/(self.subsets-1)]*(self.subsets-1)+[0])

        with self.assertRaises(ValueError):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, prob=[1/(self.subsets-1)]*(
                self.subsets-1)+[0], sampler=Sampler.random_with_replacement(10, list(np.arange(1, 11)/55.)))

        with self.assertRaises(ValueError):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A,  prob=[1/(self.subsets-1)]*(
                self.subsets-1)+[0], prob_weights=[1/(self.subsets-1)]*(self.subsets-1)+[0])

        with self.assertRaises(ValueError):
            spdhg = SPDHG(f=self.F, g=self.G, operator=self.A, sfdsdf=3,
                          sampler=Sampler.random_with_replacement(10, list(np.arange(1, 11)/55.)))

    def test_spdhg_set_norms(self):

        self.A2.set_norms([1]*len(self.A2))
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A2)
        self.assertListEqual(spdhg._norms, [1]*len(self.A2))

    def test_spdhg_check_convergence(self):
        spdhg = SPDHG(f=self.F, g=self.G, operator=self.A)

        self.assertTrue(spdhg.check_convergence())

        gamma = 3.7
        rho = 0.9
        spdhg.set_step_sizes_from_ratio(gamma, rho)
        self.assertTrue(spdhg.check_convergence())

        gamma = 3.7
        rho = 100
        spdhg.set_step_sizes_from_ratio(gamma, rho)
        self.assertFalse(spdhg.check_convergence())

        spdhg.set_step_sizes(sigma=[1]*self.subsets, tau=100)
        self.assertFalse(spdhg.check_convergence())

        spdhg.set_step_sizes(sigma=[1]*self.subsets, tau=None)
        self.assertTrue(spdhg.check_convergence())

        spdhg.set_step_sizes(sigma=None, tau=100)
        self.assertTrue(spdhg.check_convergence())

        
    def test_SPDHG_update_method(self):
        sampler = Sampler.sequential(5)
        np.random.seed(5)
        initial = VectorData(np.random.standard_normal(25))
        b = VectorData(np.array(range(25)))
        functions = []
        operators=[]
        for i in range(5):
            diagonal = np.zeros(25)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(0.5*L2NormSquared(b=A.direct(b)))
            operators.append(A)

        Aop=MatrixOperator(np.diag(np.ones(25)))
         
        g=L1Norm()
        f=BlockFunction(*functions)
        operator=BlockOperator(*operators)
        alg_stochastic = SPDHG(initial=initial, f=BlockFunction(*functions), g=g, operator=operator, sampler=sampler, update_objective_interval=500)
        
        #Test first iteration primal and dual result 
        alg_stochastic.run(1) 
        y_bar_0=operator.range_geometry().allocate(0)
        y_0=operator.range_geometry().allocate(0)
        x_1=g.proximal(initial-alg_stochastic.tau*operator.adjoint(y_bar_0), tau=alg_stochastic.tau) #primal update
        y_1=operator.range_geometry().allocate(0) 
        functions[0].proximal_conjugate(y_0[0]+alg_stochastic.sigma[0]*operators[0].direct(x_1), tau=alg_stochastic.sigma[0], out=y_1[0]) #dual update
        y_bar_1=y_1+alg_stochastic._theta*5*(y_1-y_0) #extrapolation
        
        np.testing.assert_array_almost_equal(alg_stochastic.x.as_array(), x_1.as_array()) #check primal update
        np.testing.assert_array_almost_equal(alg_stochastic._y_old[0].as_array(), y_1[0].as_array()) #check dual update in the first index (should have changed)
        np.testing.assert_array_almost_equal(alg_stochastic._y_old[1].as_array(), y_1[1].as_array())# check dual update in the second index (should not have changed)
        
        #Test second iteration
        alg_stochastic.run(1)

        x_2=g.proximal(x_1-alg_stochastic.tau*operator.adjoint(y_bar_1), tau=alg_stochastic.tau) #primal update
        y_2=y_1.copy()
        functions[1].proximal_conjugate(y_1[1]+alg_stochastic.sigma[1]*operators[1].direct(x_2), tau=alg_stochastic.sigma[1], out=y_2[1]) #dual update
        y_bar_2=y_2+alg_stochastic._theta*5*(y_2-y_1) #extrapolation

        np.testing.assert_array_almost_equal(alg_stochastic.x.as_array(), x_2.as_array()) #check primal update
        np.testing.assert_array_almost_equal(alg_stochastic._y_old[0].as_array(), y_2[0].as_array()) #check dual update in the first index (should not have changed)
        np.testing.assert_array_almost_equal(y_1[0].as_array(), y_2[0].as_array()) #check dual update in the first index (should not have changed)
        np.testing.assert_array_almost_equal(alg_stochastic._y_old[1].as_array(), y_2[1].as_array()) #check dual update in the second index (should have changed)
        
class TestCallbacks(unittest.TestCase):
    class PrintAlgo(Algorithm):
        def __init__(self, update_objective_interval=10, **kwargs):
            super().__init__(update_objective_interval=update_objective_interval, **kwargs)
            self.configured = True

        def update(self):
            self.x = -self.iteration

        def update_objective(self):
            self.loss.append(2 ** getattr(self, 'x', np.nan))

    def test_deprecated_kwargs(self):
        with self.assertWarnsRegex(DeprecationWarning, 'max_iteration'):
            self.PrintAlgo(max_iteration=1000)

        with self.assertWarnsRegex(DeprecationWarning, 'log_file'):
            self.PrintAlgo(log_file="")

    def test_progress(self):
        algo = self.PrintAlgo()

        algo.run(20)
        self.assertListEqual(algo.iterations, [-1, 10, 20])
        algo.run(3, callbacks=[callbacks.TextProgressCallback()])  # upto 23
        self.assertListEqual(algo.iterations, [-1, 10, 20])

        with self.assertWarnsRegex(DeprecationWarning, 'print_interval'):
            algo.run(40, print_interval=2)  # upto 63

        def old_callback(iteration, objective, solution):
            print(f"Called {iteration} {objective} {solution}")

        log = NamedTemporaryFile(delete=False)
        log.close()
        algo.run(20, callbacks=[callbacks.LogfileCallback(
            log.name)], callback=old_callback)
        with open(log.name, 'r') as fd:
            self.assertListEqual(
                ["64/83", "74/83", "83/83", ""],
                [line.lstrip().split(" ", 1)[0] for line in fd.readlines()])
        unlink(log.name)

        its = list(range(10, 90, 10))
        self.assertListEqual([-1] + its, algo.iterations)
        np.testing.assert_array_equal(
            [np.nan] + [2 ** (1-i) for i in its], algo.objective)

    def test_stopiteration(self):
        algo = self.PrintAlgo()
        algo.run(20, callbacks=[])
        self.assertEqual(algo.iteration, 20)

        class EarlyStopping(callbacks.Callback):
            def __call__(self, algorithm: Algorithm):
                if algorithm.iteration >= 15:
                    raise StopIteration

        algo = self.PrintAlgo()
        algo.run(20, callbacks=[EarlyStopping()])
        self.assertEqual(algo.iteration, 15)

    def test_CGLSEarlyStopping(self):
        ig = ImageGeometry(10, 2)
        np.random.seed(2)
        initial = ig.allocate(1.)
        data = ig.allocate('random_deprecated')
        operator = IdentityOperator(ig)
        alg = CGLS(initial=initial, operator=operator, data=data,
                   update_objective_interval=2)

        # Test init
        cb = callbacks.CGLSEarlyStopping(epsilon=0.1, omega=33)
        self.assertEqual(cb.epsilon, 0.1)
        self.assertEqual(cb.omega, 33)

        # Test default values
        cb = callbacks.CGLSEarlyStopping()
        self.assertEqual(cb.epsilon, 1e-6)
        self.assertEqual(cb.omega, 1e6)

        # Tests it doesn't stops iterations if the norm of the current residual is not less than epsilon times the norm of the original residual
        alg.norms = 10
        alg.norms0 = 99
        callbacks.CGLSEarlyStopping(epsilon=0.1)(alg)

        # Test it stops iterations if the norm of the current residual is less than epsilon times the norm of the original residual
        alg.norms = 1
        alg.norms0 = 100
        with self.assertRaises(StopIteration):
            callbacks.CGLSEarlyStopping(epsilon=0.1)(alg)

        # Test it doesn't stop iterations if the norm of x is smaller than omega
        alg.norms = 10
        alg.norms0 = 99
        alg.x = data
        callbacks.CGLSEarlyStopping(epsilon=0.1)(alg)

        # Test it stops iterations if the norm of x is larger than omega
        alg.norms = 10
        alg.norms0 = 99
        alg.x = data
        with self.assertRaises(StopIteration):
            callbacks.CGLSEarlyStopping(epsilon=0.1, omega=0.33)(alg)

    def test_EarlyStoppingObjectiveValue(self):
        ig = ImageGeometry(10, 2)
        np.random.seed(2)
        alg = MagicMock()
        alg.loss = [5]
        callbacks.EarlyStoppingObjectiveValue(0.1)(alg)
        alg.loss = []
        callbacks.EarlyStoppingObjectiveValue(0.1)(alg)
        alg.loss = [5, 10]
        callbacks.EarlyStoppingObjectiveValue(0.1)(alg)
        alg.loss = [5, 5.001]
        with self.assertRaises(StopIteration):
            callbacks.EarlyStoppingObjectiveValue(0.1)(alg)


class TestADMM(unittest.TestCase):
    def setUp(self):
        ig = ImageGeometry(2, 3, 2)
        data = ig.allocate(1, dtype=np.float32)
        noisy_data = data+1

        # TV regularisation parameter
        self.alpha = 1

        self.fidelities = [0.5 * L2NormSquared(b=noisy_data), L1Norm(b=noisy_data),
                           KullbackLeibler(b=noisy_data, backend='numpy')]

        F = self.alpha * MixedL21Norm()
        K = GradientOperator(ig)

        # Compute operator Norm
        normK = K.norm()

        # Primal & dual stepsizes
        self.sigma = 1./normK
        self.tau = 1./normK
        self.F = F
        self.K = K

    def test_ADMM_L2(self):
        self.do_test_with_fidelity(self.fidelities[0])

    def test_ADMM_L1(self):
        self.do_test_with_fidelity(self.fidelities[1])

    def test_ADMM_KL(self):
        self.do_test_with_fidelity(self.fidelities[2])

    def do_test_with_fidelity(self, fidelity):
        alpha = self.alpha
        # F = BlockFunction(alpha * MixedL21Norm(),fidelity)

        G = fidelity
        K = self.K
        F = self.F

        admm = LADMM(f=G, g=F, operator=K, tau=self.tau, sigma=self.sigma,
                     update_objective_interval=10)
        admm.run(1, verbose=0)

        admm_noaxpby = LADMM(f=G, g=F, operator=K, tau=self.tau, sigma=self.sigma,
                             update_objective_interval=10)
        admm_noaxpby.run(1, verbose=0)

        np.testing.assert_array_almost_equal(
            admm.solution.as_array(), admm_noaxpby.solution.as_array())

    def test_compare_with_PDHG(self):
        # Load an image from the CIL gallery.
        data = dataexample.SHAPES.get(size=(64, 64))
        ig = data.geometry
        # Add gaussian noise
        noisy_data = applynoise.gaussian(data, seed=10, var=0.0005)

        # TV regularisation parameter
        alpha = 0.1

        # fidelity = 0.5 * L2NormSquared(b=noisy_data)
        # fidelity = L1Norm(b=noisy_data)
        fidelity = KullbackLeibler(b=noisy_data, backend='numpy')

        # Setup and run the PDHG algorithm
        F = BlockFunction(alpha * MixedL21Norm(), fidelity)
        G = ZeroFunction()
        K = BlockOperator(GradientOperator(ig), IdentityOperator(ig))

        # Compute operator Norm
        normK = K.norm()

        # Primal & dual stepsizes
        sigma = 1./normK
        tau = 1./normK

        pdhg = PDHG(f=F, g=G, operator=K, tau=tau, sigma=sigma,
                    update_objective_interval=10)
        pdhg.run(500, verbose=0)

        sigma = 1
        tau = sigma/normK**2

        admm = LADMM(f=G, g=F, operator=K, tau=tau, sigma=sigma,
                     update_objective_interval=10)
        admm.run(500, verbose=0)
        np.testing.assert_almost_equal(
            admm.solution.array, pdhg.solution.array,  decimal=3)


class Test_PD3O(CCPiTestClass):

    def setUp(self):

        # default test data
        self.data = dataexample.CAMERA.get(size=(32, 32))

    def test_init_and_set_up(self):
        F1 = 0.5 * L2NormSquared(b=self.data)
        H1 = 0.1 * MixedL21Norm()
        operator = GradientOperator(self.data.geometry)
        G1 = IndicatorBox(lower=0)

        algo_pd3o = PD3O(f=F1, g=G1, h=H1, operator=operator)
        self.assertEqual(algo_pd3o.gamma, 0.99*2.0/F1.L) 
        self.assertEqual(algo_pd3o.delta, F1.L/(2.0*operator.norm()**2))
        np.testing.assert_array_equal(
            algo_pd3o.x.array, self.data.geometry.allocate(0).array)
        self.assertTrue(algo_pd3o.configured)

        algo_pd3o = PD3O(f=F1, g=G1, h=H1, operator=operator, gamma=3,
                         delta=43.1, initial=self.data.geometry.allocate('random_deprecated', seed=3))
        self.assertEqual(algo_pd3o.gamma, 3)
        self.assertEqual(algo_pd3o.delta, 43.1)
        np.testing.assert_array_equal(
            algo_pd3o.x.array, self.data.geometry.allocate('random_deprecated', seed=3).array)
        self.assertTrue(algo_pd3o.configured)

        with self.assertWarnsRegex(UserWarning, "Please use PDHG instead."):
            algo_pd3o = PD3O(f=ZeroFunction(), g=G1, h=H1,
                             operator=operator, delta=43.1)
            self.assertEqual(algo_pd3o.gamma, 1.0/operator.norm())

    def test_PD3O_PDHG_denoising_1_iteration(self):

        # compare the TV denoising problem using
        # PDHG, PD3O for 1 iteration

        # regularisation parameter
        alpha = 0.1

        # setup PDHG denoising
        F = alpha * MixedL21Norm()
        operator = GradientOperator(self.data.geometry)
        norm_op = operator.norm()
        G = 0.5 * L2NormSquared(b=self.data)
        sigma = 1./norm_op
        tau = 1./norm_op
        pdhg = PDHG(f=F, g=G, operator=operator, tau=tau,
                    sigma=sigma, update_objective_interval=100)
        pdhg.run(1)

        # setup PD3O denoising  (F=ZeroFunction)
        H = alpha * MixedL21Norm()

        F = ZeroFunction()
        gamma = 1./norm_op
        delta = 1./norm_op

        pd3O = PD3O(f=F, g=G, h=H, operator=operator, gamma=gamma, delta=delta,
                    update_objective_interval=100)
        pd3O.run(1)

        # PD3O vs pdhg
        np.testing.assert_allclose(
            pdhg.solution.array, pd3O.solution.array, atol=1e-2)

        # objective values
        np.testing.assert_allclose(
            pdhg.objective[-1], pd3O.objective[-1], atol=1e-2)

 
    def test_PD3O_stochastic_f(self): 
        sampler=Sampler.sequential(3)
        initial = VectorData(np.zeros(21))
        b =  VectorData(np.array(range(1,22)))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(21)
            diagonal[7*i:7*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))
        
        H1 = 0.1 * L1Norm()
        operator = IdentityOperator(initial.geometry)
        G1 = IndicatorBox(lower=0)


        f=SVRGFunction(functions, sampler, snapshot_update_interval=3)
        algo_pd3o = PD3O(f=f, g=G1, h=H1, operator=operator)
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 1)
        self.assertEqual(f.data_passes_indices, [[0,1,2]])
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 4/3)
        self.assertEqual(f.data_passes_indices, [[0,1,2], [0]])
        algo_pd3o.run(1)
        self.assertAlmostEqual(f.data_passes[-1], 5/3)
        self.assertEqual(f.data_passes_indices, [[0,1,2], [0], [1]])
        algo_pd3o.run(1)
        self.assertAlmostEqual(f.data_passes[-1], 8/3)
        self.assertEqual(f.data_passes_indices, [[0,1,2], [0], [1], [0,1,2]])
        
        sampler=Sampler.sequential(3)
        f=SVRGFunction(functions, sampler, snapshot_update_interval=5, store_gradients=True)
        algo_pd3o = PD3O(f= 3*(2*f), g=G1, h=H1, operator=operator)
        self.assertTrue(isinstance(3*2*f,ScaledFunction) )
        self.assertEqual(algo_pd3o.f.scalar, 6)
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 1)
        self.assertEqual(f.data_passes_indices, [[0,1,2]])
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[2].array, f.functions[2].gradient(algo_pd3o.solution).array )
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 4/3)
        self.assertEqual(f.data_passes_indices, [[0,1,2], [0]])
        algo_pd3o.run(1)
        self.assertAlmostEqual(f.data_passes[-1], 5/3)
        self.assertEqual(f.data_passes_indices, [[0,1,2], [0], [1]])
        algo_pd3o.run(1)
        self.assertAlmostEqual(f.data_passes[-1], 6/3)
        self.assertEqual(f.data_passes_indices, [[0,1,2], [0], [1], [2]])
        
        
        
        sampler=Sampler.sequential(3)
        f=LSVRGFunction(functions, sampler)
        algo_pd3o = PD3O(f=f, g=G1, h=H1, operator=operator)
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 1)
        self.assertEqual(f.data_passes_indices, [[0,1,2]])
            
        sampler=Sampler.sequential(3)
        f=SGFunction(functions, sampler)
        algo_pd3o = PD3O(f=f, g=G1, h=H1, operator=operator)
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 1/3)
        self.assertEqual(f.data_passes_indices, [[0]])
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 2/3)
        self.assertEqual(f.data_passes_indices, [[0], [1]])
    
        sampler=Sampler.sequential(3)
        f=SAGFunction(functions, sampler)
        algo_pd3o = PD3O(f=f, g=G1, h=H1, operator=operator)
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 1/3)
        self.assertEqual(f.data_passes_indices, [[0]])
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 2/3)
        self.assertEqual(f.data_passes_indices, [[0], [1]])
        
        sampler=Sampler.sequential(3)
        f=SAGAFunction(functions, sampler)
        algo_pd3o = PD3O(f=f, g=G1, h=H1, operator=operator)
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 1/3)
        self.assertEqual(f.data_passes_indices, [[0]])
        algo_pd3o.run(1)
        self.assertEqual(f.data_passes[-1], 2/3)
        self.assertEqual(f.data_passes_indices, [[0], [1]])
        
