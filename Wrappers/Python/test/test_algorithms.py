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

from utils import has_cvxpy
import unittest
from os import unlink
from tempfile import NamedTemporaryFile

import numpy as np
import logging

from cil.framework import VectorData
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.framework import BlockDataContainer
from cil.framework import BlockGeometry

from cil.optimisation.utilities import ArmijoStepSize, ConstantStepSize
from cil.optimisation.operators import IdentityOperator
from cil.optimisation.operators import GradientOperator, BlockOperator, MatrixOperator

from cil.optimisation.functions import LeastSquares, ZeroFunction, \
    L2NormSquared, OperatorCompositionFunction
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L1Norm, KullbackLeibler
from cil.optimisation.functions import IndicatorBox

from scipy.optimize import minimize, rosen

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.algorithms import GD
from cil.optimisation.algorithms import CGLS
from cil.optimisation.algorithms import SIRT
from cil.optimisation.algorithms import FISTA
from cil.optimisation.algorithms import ISTA
from cil.optimisation.algorithms import SPDHG
from cil.optimisation.algorithms import PDHG
from cil.optimisation.algorithms import LADMM

from cil.utilities import dataexample
from cil.utilities import noise as applynoise
from cil.optimisation.functions import Rosenbrock
from cil.optimisation.utilities import callbacks
from cil.framework import VectorData, VectorGeometry
from cil.utilities.quality_measures import mae, mse, psnr

# Fast Gradient Projection algorithm for Total Variation(TV)
from cil.optimisation.functions import TotalVariation
from testclass import CCPiTestClass
from utils import has_astra, initialise_tests

log = logging.getLogger(__name__)
initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator


if has_cvxpy:
    import cvxpy


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
        # b = initial.copy()
        # fill with random numbers
        # b.fill(np.random.random(initial.shape))
        b = ig.allocate('random')
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)
        step_size = norm2sq.L / 3.

        alg = GD(initial=initial, objective_function=norm2sq, step_size=step_size,
                 atol=1e-9, rtol=1e-6)
        alg.max_iteration = 1000
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg = GD(initial=initial, objective_function=norm2sq, step_size=step_size,
                 atol=1e-9, rtol=1e-6, max_iteration=20, update_objective_interval=2)
        alg.max_iteration = 20
        self.assertTrue(alg.max_iteration == 20)
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
        b = ig.allocate('random')
        identity = IdentityOperator(ig)
        norm2sq = LeastSquares(identity, b)
        alg = GD(initial=initial,
                 objective_function=norm2sq,
                 max_iteration=20,
                 update_objective_interval=0,
                 atol=1e-9, rtol=1e-6)
        self.assertTrue(alg.update_objective_interval == 0)
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg.run(20, verbose=False)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

    def test_gd_step_size_init(self):
        gd = GD(initial=self.initial, objective_function=self.f, step_size=0.002)
        self.assertEqual(gd.step_size_rule.step_size, 0.002)
        self.assertEqual(gd.step_size, 0.002)

        gd = GD(initial=self.initial, objective_function=self.f)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e6)
        self.assertEqual(gd.step_size_rule.beta, 0.5)
        self.assertEqual(gd.step_size_rule.kmax, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))
        with self.assertRaises(TypeError):
            gd.step_size

        gd = GD(initial=self.initial,
                objective_function=self.f, alpha=1e2, beta=0.25)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e2)
        self.assertEqual(gd.step_size_rule.beta, 0.25)
        self.assertEqual(gd.step_size_rule.kmax, np.ceil(
            2 * np.log10(1e2) / np.log10(2)))

        with self.assertRaises(TypeError):
            gd = GD(initial=self.initial, objective_function=self.f,
                    step_size=0.1, step_size_rule=ConstantStepSize(0.5))

    def test_gd_constant_step_size_init(self):
        rule = ConstantStepSize(0.4)
        self.assertEqual(rule.step_size, 0.4)
        gd = GD(initial=self.initial,
                objective_function=self.f, step_size_rule=rule)
        self.assertEqual(gd.step_size_rule.step_size, 0.4)
        self.assertEqual(gd.step_size, 0.4)

    def test_gd_fixed_step_size_rosen(self):

        gd = GD(initial=self.initial, objective_function=self.f, step_size=0.002,
                max_iteration=3000,
                update_objective_interval=500)
        gd.run(verbose=0)
        np.testing.assert_allclose(
            gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(
            gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)

    def test_armijo_step_size_init(self):

        rule = ArmijoStepSize()
        self.assertEqual(rule.alpha_orig, 1e6)
        self.assertEqual(rule.beta, 0.5)
        self.assertEqual(rule.kmax, np.ceil(2 * np.log10(1e6) / np.log10(2)))

        gd = GD(initial=self.initial,
                objective_function=self.f, step_size_rule=rule)
        self.assertEqual(gd.step_size_rule.alpha_orig, 1e6)
        self.assertEqual(gd.step_size_rule.beta, 0.5)
        self.assertEqual(gd.step_size_rule.kmax, np.ceil(
            2 * np.log10(1e6) / np.log10(2)))

        rule = ArmijoStepSize(5e5, 0.2, 5)
        self.assertEqual(rule.alpha_orig, 5e5)
        self.assertEqual(rule.beta, 0.2)
        self.assertEqual(rule.kmax, 5)

        gd = GD(initial=self.initial,
                objective_function=self.f, step_size_rule=rule)
        self.assertEqual(gd.step_size_rule.alpha_orig, 5e5)
        self.assertEqual(gd.step_size_rule.beta, 0.2)
        self.assertEqual(gd.step_size_rule.kmax, 5)

        with self.assertRaises(TypeError):
            gd.step_size

    def test_GDArmijo(self):
        ig = ImageGeometry(12, 13, 14)
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
        # alg.max_iteration = 20
        self.assertTrue(alg.max_iteration == 20)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

    def test_gd_armijo_rosen(self):
        armj = ArmijoStepSize(alpha=50, kmax=150)
        gd = GD(initial=self.initial, objective_function=self.f, step_size_rule=armj,
                max_iteration=2500,
                update_objective_interval=500)
        gd.run(verbose=0)
        np.testing.assert_allclose(
            gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(
            gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)


class TestFISTA(CCPiTestClass):
    def test_FISTA(self):
        ig = ImageGeometry(127, 139, 149)
        initial = ig.allocate()
        b = initial.copy()
        # fill with random numbers
        b.fill(np.random.random(initial.shape))
        initial = ig.allocate(ImageGeometry.RANDOM)
        identity = IdentityOperator(ig)

        norm2sq = OperatorCompositionFunction(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt': False}
        log.info("initial objective %s", norm2sq(initial))

        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction())
        alg.max_iteration = 2
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction(),
                    max_iteration=2, update_objective_interval=2)

        self.assertTrue(alg.max_iteration == 2)
        self.assertTrue(alg.update_objective_interval == 2)

        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        # Testing g=None
        alg = FISTA(initial=initial, f=norm2sq, g=None,
                    max_iteration=2, update_objective_interval=2)
        self.assertTrue(alg.max_iteration == 2)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        # Testing f=None
        alg = FISTA(initial=initial, f=None, g=L1Norm(b=b),
                    max_iteration=2, update_objective_interval=2)
        self.assertTrue(alg.max_iteration == 2)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        # Testing f and g is None
        with self.assertRaises(ValueError):
            alg = FISTA(initial=initial, f=None, g=None,
                        max_iteration=2, update_objective_interval=2)

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
        fista = FISTA(initial=tmp_initial, f=f, g=g, max_iteration=1)
        fista.run()

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
        fista1 = FISTA(initial=tmp_initial, f=f, g=g, max_iteration=1)
        self.assertTrue(fista1.is_provably_convergent())

        fista1 = FISTA(initial=tmp_initial, f=f, g=g,
                       max_iteration=1, step_size=30.0)
        self.assertFalse(fista1.is_provably_convergent())

    def test_FISTA_Norm2Sq(self):
        ig = ImageGeometry(127, 139, 149)
        b = ig.allocate(ImageGeometry.RANDOM)
        # fill with random numbers
        initial = ig.allocate(ImageGeometry.RANDOM)
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)

        opt = {'tol': 1e-4, 'memopt': False}
        log.info("initial objective %s", norm2sq(initial))
        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction())
        alg.max_iteration = 2
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = FISTA(initial=initial, f=norm2sq, g=ZeroFunction(),
                    max_iteration=2, update_objective_interval=3)
        self.assertTrue(alg.max_iteration == 2)
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
        initial = ig.allocate(ImageGeometry.RANDOM)
        identity = IdentityOperator(ig)

        norm2sq = LeastSquares(identity, b)
        log.info('Lipschitz %s', norm2sq.L)
        # norm2sq.L = None
        # norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        # norm2sq = OperatorCompositionFunction(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt': False}
        log.info("initial objective %s", norm2sq(initial))
        with self.assertRaises(ValueError):
            alg = FISTA(initial=initial, f=L1Norm(), g=ZeroFunction())

    def test_FISTA_Denoising(self):
        # adapted from demo FISTA_Tikhonov_Poisson_Denoising.py in CIL-Demos repository
        data = dataexample.SHAPES.get()
        ig = data.geometry
        ag = ig
        N = 300
        # Create Noisy data with Poisson noise
        scale = 5
        noisy_data = applynoise.poisson(data/scale, seed=10) * scale

        # Regularisation Parameter
        alpha = 10

        # Setup and run the FISTA algorithm
        operator = GradientOperator(ig)
        fid = KullbackLeibler(b=noisy_data)
        reg = OperatorCompositionFunction(alpha * L2NormSquared(), operator)

        initial = ig.allocate()
        fista = FISTA(initial=initial, f=reg, g=fid)
        fista.max_iteration = 3000
        fista.update_objective_interval = 500
        fista.run(verbose=0)
        rmse = (fista.get_output() - data).norm() / data.as_array().size
        log.info("RMSE %f", rmse)
        self.assertLess(rmse, 4.2e-4)


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

        # ista run 10 iteration
        tmp_initial = self.ig.allocate()
        ista = ISTA(initial=tmp_initial, f=self.f, g=self.g, max_iteration=1)
        ista.run()

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

    def test_update_g_none(self):

        # ista run 10 iteration
        tmp_initial = self.ig.allocate()
        ista = ISTA(initial=tmp_initial, f=self.f, g=None,  max_iteration=1)
        ista.run()

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        x = ista.g.proximal(x_old - (0.99*2/ista.f.L) *
                            ista.f.gradient(x_old), (1./ista.f.L))
        x_old.fill(x)

        np.testing.assert_allclose(ista.solution.array, x.array, atol=1e-2)

        # check objective
        res1 = ista.objective[-1]
        res2 = self.f(x) + self.g(x)
        self.assertTrue(res1 == res2)

    def test_update_f_none(self):

        # ista run 1 iteration
        tmp_initial = self.ig.allocate()
        ista = ISTA(initial=tmp_initial, f=None, g=self.h,  max_iteration=1)
        ista.run()

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
            ista = ISTA(initial=tmp_initial, f=None, g=None,  max_iteration=1)

    def test_provable_condition(self):

        tmp_initial = self.ig.allocate()
        ista1 = ISTA(initial=tmp_initial, f=self.f, g=self.g, max_iteration=10)
        self.assertTrue(ista1.is_provably_convergent())

        ista1 = ISTA(initial=tmp_initial, f=self.f, g=self.g,
                     max_iteration=10, step_size=30.0)
        self.assertFalse(ista1.is_provably_convergent())

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_with_cvxpy(self):

        ista = ISTA(initial=self.initial, f=self.f,
                    g=self.g, max_iteration=2000)
        ista.run(verbose=0)

        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize(
            0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4)

        np.testing.assert_allclose(p.value, ista.objective[-1], atol=1e-3)
        np.testing.assert_allclose(
            u_cvxpy.value, ista.solution.array, atol=1e-3)


class TestCGLS(CCPiTestClass):
    def test_CGLS(self):
        ig = ImageGeometry(10, 2)
        np.random.seed(2)
        initial = ig.allocate(1.)
        b = ig.allocate('random')
        identity = IdentityOperator(ig)

        alg = CGLS(initial=initial, operator=identity, data=b)

        np.testing.assert_array_equal(
            initial.as_array(), alg.solution.as_array())

        alg.max_iteration = 200
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = CGLS(initial=initial, operator=identity, data=b,
                   max_iteration=200, update_objective_interval=2)
        self.assertTrue(alg.max_iteration == 200)
        self.assertTrue(alg.update_objective_interval == 2)
        alg.run(20, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())


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
        pdhg1.max_iteration = 2000
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
                     max_iteration=2000, update_objective_interval=200)

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
        pdhg1.max_iteration = 2000
        pdhg1.update_objective_interval = 200
        pdhg1.run(1000, verbose=0)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        log.info("RMSE %f", rmse)
        self.assertLess(rmse, 2e-4)

    def test_PDHG_step_sizes(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = 3*IdentityOperator(ig)

        # check if sigma, tau are None
        pdhg = PDHG(f=f, g=g, operator=operator, max_iteration=10)
        self.assertAlmostEqual(pdhg.sigma, 1./operator.norm())
        self.assertAlmostEqual(pdhg.tau, 1./operator.norm())

        # check if sigma is negative
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        max_iteration=10, sigma=-1)

        # check if tau is negative
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, max_iteration=10, tau=-1)

        # check if tau is None
        sigma = 3.0
        pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, max_iteration=10)
        self.assertAlmostEqual(pdhg.sigma, sigma)
        self.assertAlmostEqual(pdhg.tau, 1./(sigma * operator.norm()**2))

        # check if sigma is None
        tau = 3.0
        pdhg = PDHG(f=f, g=g, operator=operator, tau=tau, max_iteration=10)
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, 1./(tau * operator.norm()**2))

        # check if sigma/tau are not None
        tau = 1.0
        sigma = 1.0
        pdhg = PDHG(f=f, g=g, operator=operator, tau=tau,
                    sigma=sigma, max_iteration=10)
        self.assertAlmostEqual(pdhg.tau, tau)
        self.assertAlmostEqual(pdhg.sigma, sigma)

        # check sigma/tau as arrays, sigma wrong shape
        ig1 = ImageGeometry(2, 2)
        sigma = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        sigma=sigma, max_iteration=10)

        # check sigma/tau as arrays, tau wrong shape
        tau = ig1.allocate()
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, tau=tau, max_iteration=10)

        # check sigma not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        sigma="sigma", max_iteration=10)

        # check tau not Number or object with correct shape
        with self.assertRaises(AttributeError):
            pdhg = PDHG(f=f, g=g, operator=operator,
                        tau="tau", max_iteration=10)

        # check warning message if condition is not satisfied
        sigma = 4
        tau = 1/3
        with self.assertWarnsRegex(UserWarning, "Convergence criterion"):
            pdhg = PDHG(f=f, g=g, operator=operator, tau=tau,
                        sigma=sigma, max_iteration=10)

    def test_PDHG_strongly_convex_gamma_g(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0

        pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                    max_iteration=5, gamma_g=0.5)
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
                        max_iteration=5, gamma_g=-0.5)

        # check strongly convex constant not a number
        with self.assertRaises(ValueError):
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        max_iteration=5, gamma_g="-0.5")

    def test_PDHG_strongly_convex_gamma_fcong(self):
        ig = ImageGeometry(3, 3)
        data = ig.allocate('random')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        # sigma, tau
        sigma = 1.0
        tau = 1.0

        pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                    max_iteration=5, gamma_fconj=0.5)
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
                        max_iteration=5, gamma_fconj=-0.5)
        except ValueError as ve:
            log.info(str(ve))

        # check strongly convex constant not a number
        try:
            pdhg = PDHG(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                        max_iteration=5, gamma_fconj="-0.5")
        except ValueError as ve:
            log.info(str(ve))

    def test_PDHG_strongly_convex_both_fconj_and_g(self):

        ig = ImageGeometry(3, 3)
        data = ig.allocate('random')

        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        try:
            pdhg = PDHG(f=f, g=g, operator=operator, max_iteration=10,
                        gamma_g=0.5, gamma_fconj=0.5)
            pdhg.run(verbose=0)
        except ValueError as err:
            log.info(str(err))


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
        self.b2 = self.ig2.allocate('random')
        self.A2 = IdentityOperator(self.ig2)

    def tearDown(self):
        pass

    def test_update(self):
        # sirt run 5 iterations
        tmp_initial = self.ig.allocate()
        sirt = SIRT(initial=tmp_initial, operator=self.Aop,
                    data=self.bop, max_iteration=5)
        sirt.run()

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        for _ in range(5):
            x = x_old + sirt.D * \
                (sirt.operator.adjoint(sirt.M*(sirt.data - sirt.operator.direct(x_old))))
            x_old.fill(x)

        np.testing.assert_allclose(sirt.solution.array, x.array, atol=1e-2)

    def test_update_constraints(self):
        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2, max_iteration=20)
        alg.run(verbose=0)
        np.testing.assert_array_almost_equal(alg.x.array, self.b2.array)

        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2, max_iteration=20, upper=0.3)
        alg.run(verbose=0)
        np.testing.assert_almost_equal(alg.solution.max(), 0.3)

        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2, max_iteration=20, lower=0.7)
        alg.run(verbose=0)
        np.testing.assert_almost_equal(alg.solution.min(), 0.7)

        alg = SIRT(initial=self.initial2, operator=self.A2, data=self.b2,
                   max_iteration=20, constraint=IndicatorBox(lower=0.1, upper=0.3))
        alg.run(verbose=0)
        np.testing.assert_almost_equal(alg.solution.max(), 0.3)
        np.testing.assert_almost_equal(alg.solution.min(), 0.1)

    def test_SIRT_relaxation_parameter(self):
        tmp_initial = self.ig.allocate()
        alg = SIRT(initial=tmp_initial, operator=self.Aop,
                   data=self.bop, max_iteration=5)

        with self.assertRaises(ValueError):
            alg.set_relaxation_parameter(0)

        with self.assertRaises(ValueError):
            alg.set_relaxation_parameter(2)

        alg = SIRT(initial=self.initial2, operator=self.A2,
                   data=self.b2, max_iteration=20)
        alg.set_relaxation_parameter(0.5)

        self.assertEqual(alg.relaxation_parameter, 0.5)

        alg.run(verbose=0)
        np.testing.assert_array_almost_equal(alg.x.array, self.b2.array)

        np.testing.assert_almost_equal(0.5 * alg.D.array, alg._Dscaled.array)

    def test_SIRT_nan_inf_values(self):
        Aop_nan_inf = self.Aop
        Aop_nan_inf.A[0:10, :] = 0.
        Aop_nan_inf.A[:, 10:20] = 0.

        tmp_initial = self.ig.allocate()
        sirt = SIRT(initial=tmp_initial, operator=Aop_nan_inf,
                    data=self.bop, max_iteration=5)

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
                    data=bop, max_iteration=5)
        for el in sirt.M.containers:
            self.assertFalse(np.any(el == np.inf))

        self.assertFalse(np.any(sirt.D == np.inf))

    def test_SIRT_with_TV(self):
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128, 128))
        ig = data.geometry
        A = IdentityOperator(ig)
        constraint = TotalVariation(warm_start=False, max_iteration=100)
        initial = ig.allocate('random', seed=5)
        sirt = SIRT(initial=initial, operator=A, data=data,
                    max_iteration=2, constraint=constraint)
        sirt.run(2, verbose=0)
        f = LeastSquares(A, data, c=0.5)
        fista = FISTA(initial=initial, f=f, g=constraint, max_iteration=1000)
        fista.run(100, verbose=0)
        self.assertNumpyArrayAlmostEqual(fista.x.as_array(), sirt.x.as_array())

    def test_SIRT_with_TV_warm_start(self):
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128, 128))
        ig = data.geometry
        A = IdentityOperator(ig)
        constraint = 1e6*TotalVariation(warm_start=True, max_iteration=100)
        initial = ig.allocate('random', seed=5)
        sirt = SIRT(initial=initial, operator=A, data=data,
                    max_iteration=150, constraint=constraint)
        sirt.run(25, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            sirt.x.as_array(), ig.allocate(0.25).as_array(), 3)


class TestSPDHG(unittest.TestCase):

    @unittest.skipUnless(has_astra, "cil-astra not available")
    def test_SPDHG_vs_PDHG_implicit(self):
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128, 128))

        ig = data.geometry
        ig.voxel_size_x = 0.1
        ig.voxel_size_y = 0.1

        detectors = ig.shape[0]
        angles = np.linspace(0, np.pi, 90)
        ag = AcquisitionGeometry.create_Parallel2D().set_angles(
            angles, angle_unit='radian').set_panel(detectors, 0.1)
        # Select device
        dev = 'cpu'

        Aop = ProjectionOperator(ig, ag, dev)

        sin = Aop.direct(data)
        # Create noisy data. Apply Gaussian noise
        noises = ['gaussian', 'poisson']
        noise = noises[1]
        noisy_data = ag.allocate()
        if noise == 'poisson':
            np.random.seed(10)
            scale = 20
            eta = 0
            noisy_data.fill(np.random.poisson(
                scale * (eta + sin.as_array()))/scale)
        elif noise == 'gaussian':
            np.random.seed(10)
            n1 = np.random.normal(0, 0.1, size=ag.shape)
            noisy_data.fill(n1 + sin.as_array())
        else:
            raise ValueError('Unsupported Noise ', noise)

        # Create BlockOperator
        operator = Aop
        f = KullbackLeibler(b=noisy_data)
        alpha = 0.005
        g = alpha * TotalVariation(50, 1e-4, lower=0, warm_start=True)
        normK = operator.norm()

        # % 'implicit' PDHG, preconditioned step-sizes
        tau_tmp = 1.
        sigma_tmp = 1.
        tau = sigma_tmp / \
            operator.adjoint(tau_tmp * operator.range_geometry().allocate(1.))
        sigma = tau_tmp / \
            operator.direct(
                sigma_tmp * operator.domain_geometry().allocate(1.))

        # Setup and run the PDHG algorithm
        pdhg = PDHG(f=f, g=g, operator=operator, tau=tau, sigma=sigma,
                    max_iteration=1000,
                    update_objective_interval=500)
        pdhg.run(verbose=0)

        subsets = 10
        size_of_subsets = int(len(angles)/subsets)
        # take angles and create uniform subsets in uniform+sequential setting
        list_angles = [angles[i:i+size_of_subsets]
                       for i in range(0, len(angles), size_of_subsets)]
        # create acquisitioin geometries for each the interval of splitting angles
        list_geoms = [AcquisitionGeometry.create_Parallel2D().set_angles(list_angles[i], angle_unit='radian').set_panel(detectors, 0.1)
                      for i in range(len(list_angles))]
        # create with operators as many as the subsets
        A = BlockOperator(*[ProjectionOperator(ig, list_geoms[i], dev)
                          for i in range(subsets)])
        # number of subsets
        # (sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
        #
        # acquisisiton data
        AD_list = []
        for sub_num in range(subsets):
            for i in range(0, len(angles), size_of_subsets):
                arr = noisy_data.as_array()[i:i+size_of_subsets, :]
                AD_list.append(AcquisitionData(
                    arr, geometry=list_geoms[sub_num]))

        g = BlockDataContainer(*AD_list)

        # block function
        F = BlockFunction(*[KullbackLeibler(b=g[i]) for i in range(subsets)])
        G = alpha * TotalVariation(50, 1e-4, lower=0, warm_start=True)

        prob = [1/len(A)]*len(A)
        spdhg = SPDHG(f=F, g=G, operator=A,
                      max_iteration=1000,
                      update_objective_interval=200, prob=prob)
        spdhg.run(1000, verbose=0)
        qm = (mae(spdhg.get_output(), pdhg.get_output()),
              mse(spdhg.get_output(), pdhg.get_output()),
              psnr(spdhg.get_output(), pdhg.get_output())
              )
        log.info("Quality measures %r", qm)

        np.testing.assert_almost_equal(mae(spdhg.get_output(), pdhg.get_output()),
                                       0.000335, decimal=3)
        np.testing.assert_almost_equal(mse(spdhg.get_output(), pdhg.get_output()),
                                       5.51141e-06, decimal=3)

    @unittest.skipUnless(has_astra, "ccpi-astra not available")
    def test_SPDHG_vs_PDHG_explicit(self):
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128, 128))

        ig = data.geometry
        ig.voxel_size_x = 0.1
        ig.voxel_size_y = 0.1

        detectors = ig.shape[0]
        angles = np.linspace(0, np.pi, 180)
        ag = AcquisitionGeometry.create_Parallel2D().set_angles(
            angles, angle_unit='radian').set_panel(detectors, 0.1)
        # Select device
        dev = 'cpu'

        Aop = ProjectionOperator(ig, ag, dev)

        sin = Aop.direct(data)
        # Create noisy data. Apply Gaussian noise
        noises = ['gaussian', 'poisson']
        noise = noises[1]
        if noise == 'poisson':
            scale = 5
            noisy_data = scale * applynoise.poisson(sin/scale, seed=10)
            # np.random.seed(10)
            # scale = 5
            # eta = 0
            # noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
        elif noise == 'gaussian':
            noisy_data = noise.gaussian(sin, var=0.1, seed=10)
        else:
            raise ValueError('Unsupported Noise ', noise)

        # %% 'explicit' SPDHG, scalar step-sizes
        subsets = 10
        size_of_subsets = int(len(angles)/subsets)
        # create Gradient operator
        op1 = GradientOperator(ig)
        # take angles and create uniform subsets in uniform+sequential setting
        list_angles = [angles[i:i+size_of_subsets]
                       for i in range(0, len(angles), size_of_subsets)]
        # create acquisitioin geometries for each the interval of splitting angles
        list_geoms = [AcquisitionGeometry.create_Parallel2D().set_angles(list_angles[i], angle_unit='radian').set_panel(detectors, 0.1)
                      for i in range(len(list_angles))]
        # create with operators as many as the subsets
        A = BlockOperator(*[ProjectionOperator(ig, list_geoms[i], dev)
                          for i in range(subsets)] + [op1])
        # number of subsets
        # (sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
        #
        # acquisisiton data
        AD_list = []
        for sub_num in range(subsets):
            for i in range(0, len(angles), size_of_subsets):
                arr = noisy_data.as_array()[i:i+size_of_subsets, :]
                AD_list.append(AcquisitionData(
                    arr, geometry=list_geoms[sub_num]))

        g = BlockDataContainer(*AD_list)
        alpha = 0.5
        # block function
        F = BlockFunction(*[*[KullbackLeibler(b=g[i])
                          for i in range(subsets)] + [alpha * MixedL21Norm()]])
        G = IndicatorBox(lower=0)

        prob = [1/(2*subsets)]*(len(A)-1) + [1/2]
        spdhg = SPDHG(f=F, g=G, operator=A,
                      max_iteration=1000,
                      update_objective_interval=200, prob=prob)
        spdhg.run(1000, verbose=0)

        # %% 'explicit' PDHG, scalar step-sizes
        op1 = GradientOperator(ig)
        op2 = Aop
        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2, 1))
        f2 = KullbackLeibler(b=noisy_data)
        g = IndicatorBox(lower=0)
        normK = operator.norm()
        sigma = 1/normK
        tau = 1/normK

        f1 = alpha * MixedL21Norm()
        f = BlockFunction(f1, f2)
        # Setup and run the PDHG algorithm
        pdhg = PDHG(f=f, g=g, operator=operator, tau=tau, sigma=sigma)
        pdhg.max_iteration = 1000
        pdhg.update_objective_interval = 200
        pdhg.run(1000, verbose=0)

        # %% show diff between PDHG and SPDHG
        # plt.imshow(spdhg.get_output().as_array() -pdhg.get_output().as_array())
        # plt.colorbar()
        # plt.show()

        qm = (mae(spdhg.get_output(), pdhg.get_output()),
              mse(spdhg.get_output(), pdhg.get_output()),
              psnr(spdhg.get_output(), pdhg.get_output())
              )
        log.info("Quality measures: %r", qm)
        np.testing.assert_almost_equal(mae(spdhg.get_output(), pdhg.get_output()),
                                       0.00150, decimal=3)
        np.testing.assert_almost_equal(mse(spdhg.get_output(), pdhg.get_output()),
                                       1.68590e-05, decimal=3)


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
                     max_iteration=100, update_objective_interval=10)
        admm.run(1, verbose=0)

        admm_noaxpby = LADMM(f=G, g=F, operator=K, tau=self.tau, sigma=self.sigma,
                             max_iteration=100, update_objective_interval=10)
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
                    max_iteration=500, update_objective_interval=10)
        pdhg.run(verbose=0)

        sigma = 1
        tau = sigma/normK**2

        admm = LADMM(f=G, g=F, operator=K, tau=tau, sigma=sigma,
                     max_iteration=500, update_objective_interval=10)
        admm.run(verbose=0)
        np.testing.assert_almost_equal(
            admm.solution.array, pdhg.solution.array,  decimal=3)
