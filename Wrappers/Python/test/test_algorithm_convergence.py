
from cil.optimisation.algorithms import SPDHG, PDHG, LSQR, FISTA, APGD, GD, PD3O
from cil.optimisation.functions import L2NormSquared, IndicatorBox, BlockFunction, ZeroFunction, KullbackLeibler, OperatorCompositionFunction, LeastSquares, TotalVariation, MixedL21Norm, L1Norm
from cil.optimisation.operators import BlockOperator, IdentityOperator, MatrixOperator, GradientOperator
from cil.optimisation.utilities import Sampler, BarzilaiBorweinStepSizeRule, ArmijoStepSizeRule, PDHGAdaptiveStepSize2013, PDHGAdaptiveStepSize2015, PDHGBayesOptimisationStepSize, SPDHGBayesOptimisationStepSize, SPDHGStepSizesFromRatio, SPDHGAdaptiveStepSizeBalancing, SPDHGAdaptiveStepSizeAngle
from cil.framework import AcquisitionGeometry, BlockDataContainer, BlockGeometry, VectorData, ImageGeometry
from cil.utilities import dataexample
from cil.utilities import noise as applynoise

import numpy as np
import unittest
from testclass import CCPiTestClass
from utils import has_skopt

from scipy.optimize import minimize, rosen
from cil.optimisation.functions import Rosenbrock

try:
    import cvxpy
    has_cvxpy = True
except ImportError:
    has_cvxpy = False

try:
    import astra
    has_astra = True
    from cil.plugins.astra import ProjectionOperator
except ImportError:
    has_astra = False

import logging 
log = logging.getLogger(__name__)


class TestSPDHG(CCPiTestClass):
    @unittest.skipUnless(has_astra, "cil-astra not available")
    def test_SPDHG_num_subsets_1_astra(self):
        
        data = dataexample.SIMPLE_PHANTOM_2D.get(size=(10, 10))

        subsets = 1

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
        partitioned_data = sin.partition(subsets, 'sequential')
        A = BlockOperator(
            *[IdentityOperator(partitioned_data[i].geometry) for i in range(subsets)])

        # block function
        F = BlockFunction(*[L2NormSquared(b=partitioned_data[i])
                            for i in range(subsets)])

        F_phdhg = L2NormSquared(b=partitioned_data[0])
        A_pdhg = IdentityOperator(partitioned_data[0].geometry)

        alpha = 0.025
        G = alpha * IndicatorBox(lower=0)

        spdhg = SPDHG(f=F, g=G, operator=A,  update_objective_interval=10)

        spdhg.run(7)

        pdhg = PDHG(f=F_phdhg, g=G, operator=A_pdhg,
                    update_objective_interval=10)

        pdhg.run(7)

        self.assertNumpyArrayAlmostEqual(
            pdhg.solution.as_array(), spdhg.solution.as_array(), decimal=3)


    @unittest.skipUnless(has_cvxpy, "cvxpy not available")
    def test_SPDHG_toy_example(self):
        sampler = Sampler.random_with_replacement(5, seed=10)
        np.random.seed(10)
        initial = VectorData(np.random.standard_normal(25))
        b = VectorData(np.arange(25))
        functions = []
        operators=[]
        for i in range(5):
            diagonal = np.zeros(25)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(0.5*L2NormSquared(b=A.direct(b)))
            operators.append(A)

        Aop=MatrixOperator(np.diag(np.ones(25)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4)

        g=ZeroFunction()

        alg_stochastic = SPDHG(f=BlockFunction(*functions), g=g, operator=BlockOperator(*operators), sampler=sampler, initial=initial, update_objective_interval=500)
        alg_stochastic.run(200, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), decimal=6)


        
class TestLSQR(CCPiTestClass):

    def setUp(self):
        # Mock the operator and data containers
        
        np.random.seed(10)
        self.n = 50
        self.m = 70

        A = np.random.uniform(0, 1, (self.m, self.n)).astype('float32')
        x = np.arange(self.n, dtype=np.float32) / self.n
        b = A.dot(x)

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b)
        self.x = VectorData(x)
        
        self.ig = self.Aop.domain

        self.initial = self.ig.allocate(None)
        self.initial.fill(np.ones(self.n)/self.n)
        

    def test_convergence(self):
        lsqr = LSQR(initial=self.initial, operator=self.Aop, data=self.bop, alpha=0)
        lsqr.run(200)
        self.assertNumpyArrayAlmostEqual(lsqr.solution.as_array(), self.x.as_array(), 3)
        self.assertAlmostEqual(lsqr.objective[-1], (self.Aop.direct(self.x)-self.bop).norm()**2, 1)

 

class TestFISTA(CCPiTestClass):
    def test_FISTA_Denoising(self):
        # adapted from demo FISTA_Tikhonov_Poisson_Denoising.py in CIL-Demos repository
        data = dataexample.SHAPES.get()
        ig = data.geometry
        ag = ig
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
        fista.update_objective_interval = 500
        fista.run(3000, verbose=0)
        rmse = (fista.get_output() - data).norm() / data.as_array().size
        self.assertLess(rmse, 4.2e-4)

    def test_APGD(self):
        ig = ImageGeometry(41, 43, 47)
        initial = ig.allocate(0)
        b = ig.allocate("random")**2
        identity = IdentityOperator(ig)

        f = OperatorCompositionFunction(L2NormSquared(b=b), identity)
        g= IndicatorBox(lower=0)

        apgd = APGD(f=f, g=g, initial=initial, update_objective_interval=100, momentum=0.5)
        apgd.run(500, verbose=0)
        self.assertNumpyArrayAlmostEqual(apgd.solution.as_array(), b.as_array(), decimal=3)


    @unittest.skipUnless(has_cvxpy, "cvxpy not available")
    def test_APGD_dossal_chambolle(self):

        np.random.seed(10)
        n = 100
        m = 50
        A = np.random.normal(0,1, (m, n)).astype('float32')
        b = np.random.normal(0,1, m).astype('float32')
        reg = 0.5

        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig = Aop.domain

        # cvxpy solutions
        u_cvxpy = cvxpy.Variable(ig.shape[0])
        objective = cvxpy.Minimize( 0.5 * cvxpy.sum_squares(Aop.A @ u_cvxpy - bop.array) + reg/2 * cvxpy.sum_squares(u_cvxpy))
        p = cvxpy.Problem(objective)
        p.solve(verbose=False, solver=cvxpy.SCS, eps=1e-4)

        # default fista
        f = LeastSquares(A=Aop, b=bop, c=0.5)
        g = reg/2*L2NormSquared()
        fista = FISTA(initial=ig.allocate(), f=f, g=g, update_objective_interval=1)
        fista.run(500)
        np.testing.assert_allclose(fista.objective[-1], p.value, atol=1e-3)
        np.testing.assert_allclose(fista.solution.array, u_cvxpy.value, atol=1e-3)

        # fista Dossal Chambolle "On the convergence of the iterates of ”FISTA”
        from cil.optimisation.algorithms.APGD import ScalarMomentumCoefficient
        class DossalChambolle(ScalarMomentumCoefficient):
            def __call__(self, algo=None):
                return (algo.iteration-1)/(algo.iteration+50)
        momentum = DossalChambolle()
        fista_dc = APGD(initial=ig.allocate(), f=f, g=g, update_objective_interval=1, momentum=momentum)
        fista_dc.run(500)
        np.testing.assert_allclose(fista_dc.solution.array, u_cvxpy.value, atol=1e-3)
        np.testing.assert_allclose(fista_dc.solution.array, u_cvxpy.value, atol=1e-3)

class TestPD3O(CCPiTestClass):
    def test_pd3o_convergence(self):
        data = dataexample.CAMERA.get(size=(32, 32))
        # pd30 convergence test using TV denoising

        # regularisation parameter
        alpha = 0.11

        # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
        TV = TotalVariation(max_iteration=40)
        tv_cil = TV.proximal(data, tau=alpha)

        F = alpha * MixedL21Norm()
        operator = GradientOperator(data.geometry)
        norm_op = operator.norm()

        # setup PD3O denoising  (H proximalble and G,F = 1/4 * L2NormSquared)
        H = alpha * MixedL21Norm()
        G = 0.25 * L2NormSquared(b=data)
        F = 0.25 * L2NormSquared(b=data)
        gamma = 2./F.L
        delta = 1./(gamma*norm_op**2)

        pd3O_with_f = PD3O(f=F, g=G, h=H, operator=operator, gamma=gamma, delta=delta,
                           update_objective_interval=100)
        pd3O_with_f.run(800)

        # pd30 vs fista
        np.testing.assert_allclose(
            tv_cil.array, pd3O_with_f.solution.array, atol=1e-2)


class TestGD(CCPiTestClass):
    
    def setUp(self):
        x0_1 = 1.1
        x0_2 = 1.1
        x0 = np.array([x0_1, x0_2])

        self.initial = VectorData(np.array(x0))
        method = 'Nelder-Mead' 
        self.scipy_opt_high = minimize(
            rosen, x0, method=method, tol=1e-2)  # (1., 1.)
        self.f = Rosenbrock(alpha=1, beta=100)
        
    def test_gd_fixed_step_size_rosen(self):

        gd = GD(initial=self.initial, f=self.f, step_size=0.002,
                update_objective_interval=500)
        gd.run(3000, verbose=0)
        np.testing.assert_allclose(
            gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(
            gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)
        
    def test_gd_armijo_rosen(self):
        
        
        armj = ArmijoStepSizeRule(alpha=50, max_iterations=50, warmstart=False)
        gd = GD(initial=self.initial, f=self.f, step_size=armj,
                update_objective_interval=500)
        gd.run(4000, verbose=0)
        self.assertAlmostEqual(gd.solution.array[0], self.scipy_opt_high.x[0], places=3)
        self.assertAlmostEqual(gd.solution.array[1], self.scipy_opt_high.x[1], places=3)

    def test_bb_step_size_gd_converge(self):
        np.random.seed(2)
        n = 10
        m = 10
        A = np.arange(1, n*m+1, dtype=np.float32).reshape(n,m)
        A = np.diag(1/(np.transpose(A)@np.ones(m)))*A
        x = (np.arange(n, dtype=np.float32)-n/2)/n
        b=A@x


        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig=Aop.domain

        initial = VectorData((np.arange(n, dtype=np.float32)-n/2)/(n+1))
        f = LeastSquares(Aop, b=bop, c=2)



        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'short')
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        alg.run(300, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), x, decimal=4)

        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'long')
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        alg.run(300, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), x, decimal=4)

        ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'alternate')
        alg = GD(initial=initial, f=f, step_size=ss_rule)
        alg.run(300, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), x, decimal=4)
        
class TestPDHGConvergence(CCPiTestClass):
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
        pdhg1 = PDHG(f=f1, g=g, operator=operator, step_size=[tau,sigma])
        self.assertEqual( pdhg1.tau, tau)
        self.assertEqual(pdhg1.sigma, sigma)
        
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
        pdhg1 = PDHG(f=f1, g=g, operator=operator, step_size=(tau,sigma),
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
        pdhg1 = PDHG(f=f1, g=g, operator=operator, step_size=(tau,sigma) )
        pdhg1.update_objective_interval = 200
        pdhg1.run(1000, verbose=0)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        log.info("RMSE %f", rmse)
        self.assertLess(rmse, 2e-4)  
        
    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_PDHG_bayes_convergence(self):
        # min_x ||x - b||^2 + ||x||^2, whose minimiser is b/2. The claim being
        # tested is not just that PDHG converges, but that the gamma the search picks beats both ends of the search interval
        ig = ImageGeometry(3, 3)
        data = ig.allocate(0)
        data.fill(np.diag([1, 2, 3]))
        ideal = ig.allocate(0)
        ideal.fill(np.diag([0.5, 1, 1.5]))
        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGBayesOptimisationStepSize(
            gamma_bounds=[1e-2, 1e2], n_initial_points=5, n_calls=10,
            n_iterations=10, seed=42)
        bayes = PDHG(f=f, g=g, operator=operator, step_size=rule,
                     update_objective_interval=1)
        bayes.run(20, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            bayes.x.as_array(), ideal.as_array(), decimal=4)
        # check the objective has decreased from its initial value (the first entry is the initial value)
        self.assertLess(bayes.objective[-1], bayes.objective[0])

        # over the same budget the chosen gamma must beat both ends of the
        # interval it was searched over
        norm = operator.norm()
        for gamma in (1e-2, 1e2):
            with self.subTest(gamma=gamma):
                fixed = PDHG(f=f, g=g, operator=operator,
                             step_size=(1. / (gamma * norm), gamma / norm),
                             update_objective_interval=1)
                fixed.run(20, verbose=0)
                self.assertLess(bayes.objective[-1], fixed.objective[-1])
                self.assertLess((bayes.x - ideal).norm(), (fixed.x - ideal).norm())

  
    
    def test_PDHG_adaptive_step_size_2013(self):
        # The rule starts from tau = sigma = 10/||K||, which violates the PDHG
        # condition tau*sigma*||K||^2 <= 1 by a factor of 100: held fixed there the
        # algorithm diverges. What is tested is that the rule pulls the step sizes
        # back into the convergent region and reaches the minimiser b/2 -- not just
        # that PDHG converges, which it would here with no rule at all.
        ig = ImageGeometry(3, 3)
        data = ig.allocate(0)
        data.fill(np.diag([1, 2, 3]))
        ideal = ig.allocate(0)
        ideal.fill(np.diag([0.5, 1, 1.5]))
        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2013()
        adaptive = PDHG(f=f, g=g, operator=operator, step_size=rule,
                        update_objective_interval=1)
        tau0, sigma0 = adaptive.tau, adaptive.sigma
        adaptive.run(20, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            adaptive.x.as_array(), ideal.as_array(), decimal=4)
        # check that the objective decreases
        self.assertLess(adaptive.objective[-1], adaptive.objective[0])
        # check the step sizes are now in the convergent region
        self.assertLess(adaptive.tau * adaptive.sigma * operator.norm() ** 2, 1.)

        # check that the result is better than with the initial, bad step sizes held fixed, so the assertions above are properties of the rule rather than of an easy problem
        diverging = PDHG(f=f, g=g, operator=operator, step_size=(tau0, sigma0),
                         update_objective_interval=1)
        diverging.run(20, verbose=0)
        self.assertLess(adaptive.objective[-1], diverging.objective[-1])
    
    def test_PDHG_adaptive_step_size_2015(self):
        # Companion to test_PDHG_adaptive_step_size_2013 for the 2015 rule, from the
        # same divergent start of tau = sigma = 10/||K||. Note this rule is *slower*
        # here than the default constant step size, so the comparison is against the
        # start it was given rather than against a well-tuned run.
        ig = ImageGeometry(3, 3)
        data = ig.allocate(0)
        data.fill(np.diag([1, 2, 3]))
        ideal = ig.allocate(0)
        ideal.fill(np.diag([0.5, 1, 1.5]))
        f = L2NormSquared(b=data)
        g = L2NormSquared()
        operator = IdentityOperator(ig)

        rule = PDHGAdaptiveStepSize2015()
        adaptive = PDHG(f=f, g=g, operator=operator, step_size=rule,
                        update_objective_interval=1)
        tau0, sigma0 = adaptive.tau, adaptive.sigma
        adaptive.run(30, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            adaptive.x.as_array(), ideal.as_array(), decimal=4)
        # check that the objective decreases
        self.assertLess(adaptive.objective[-1], adaptive.objective[0])
        # check the step sizes are now in the convergent region
        self.assertLess(adaptive.tau * adaptive.sigma * operator.norm() ** 2, 1.)

        # check that the result is better than with the initial, bad step sizes held fixed, so the assertions above are properties of the rule rather than of an easy problem
        diverging = PDHG(f=f, g=g, operator=operator, step_size=(tau0, sigma0),
                         update_objective_interval=1)
        diverging.run(30, verbose=0)
        self.assertLess(adaptive.objective[-1], diverging.objective[-1])
        
        
class TestSPDHGConvergence(CCPiTestClass):
    """Convergence tests for the SPDHG step-size rules.

    All of them solve min_x 2||x - b||^2 + ||x||^2 split over two subsets, whose
    minimiser is 2b/3. SPDHG samples subsets at random, so every run is given a
    seeded sampler.
    """

    def setUp(self):
        self.subsets = 2
        ig = ImageGeometry(3, 3)
        data_ind = ig.allocate(0)
        data_ind.fill(np.diag([1, 2, 3]))
        self.ideal = ig.allocate(0)
        self.ideal.fill(np.diag([2/3, 4/3, 2]))

        self.A = BlockOperator(*[IdentityOperator(ig) for i in range(self.subsets)])
        self.F = BlockFunction(*[L2NormSquared(b=data_ind)
                                 for i in range(self.subsets)])
        self.G = L2NormSquared()

    @unittest.skipUnless(has_skopt, "scikit-optimize (skopt) not installed")
    def test_SPDHG_bayes_convergence(self):
        # min_x 2||x - b||^2 + ||x||^2 split over two subsets, whose minimiser is
        # 2b/3. The claim being tested is not just that SPDHG converges, but that
        # the gamma the search picks beats the ends of the search interval 
        # SPDHG samples subsets at random, so every run here is seeded.
        seed = 7
        rule = SPDHGBayesOptimisationStepSize(
            gamma_bounds=[1e-2, 1e2], n_initial_points=5, n_calls=10,
            n_iterations=10, seed=42)
        bayes = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                      sampler=Sampler.random_with_replacement(self.subsets, seed=seed),
                      update_objective_interval=1)
        bayes.run(50, verbose=0)

        self.assertNumpyArrayAlmostEqual(
            bayes.x.as_array(), self.ideal.as_array(), decimal=4)
        # objective has decreased
        self.assertLess(bayes.objective[-1], bayes.objective[0])

        # over the same budget, and on the same sequence of subsets, the chosen
        # gamma must beat both ends of the interval it was searched over
        for gamma in (1e-2, 1e2):
            with self.subTest(gamma=gamma):
                fixed = SPDHG(f=self.F, g=self.G, operator=self.A,
                              step_size=SPDHGStepSizesFromRatio(gamma, 0.99),
                              sampler=Sampler.random_with_replacement(self.subsets, seed=seed),
                              update_objective_interval=1)
                fixed.run(50, verbose=0)
                self.assertLess(bayes.objective[-1], fixed.objective[-1])
                self.assertLess((bayes.x - self.ideal).norm(),
                                (fixed.x - self.ideal).norm())
        
    def test_convergence_balancing(self):
        # From a poorly balanced start (tau far too large, sigma far too small) the
        # residual-balancing rule (rule (a) of Chambolle et al. 2023) should bringthe
        # step sizes back, reduce the objective and reach a value comparable to a
        # well-tuned constant rule, converging to the same minimiser.
        # the adaptive run recovers from the bad start but, having spent early iterations
                # rebalancing, is a little behind the well-tuned constant rule at 50 iterations.
        # SPDHG samples subsets at random, so both runs are seeded for reproducibility.
        seed = 7
        constant = SPDHG(f=self.F, g=self.G, operator=self.A,
                         sampler=Sampler.random_with_replacement(self.subsets, seed=seed))
        constant.run(50, verbose=0)

        rule = SPDHGAdaptiveStepSizeBalancing(initial_step_size=[10.0, 0.001])
        adaptive = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                         sampler=Sampler.random_with_replacement(self.subsets, seed=seed))
        adaptive.run(50, verbose=0)
        # check the objective has decreased from its initial value (the first entry is the initial value)
        self.assertLess(adaptive.objective[-1], adaptive.objective[0])
    
        # Check both get close tot he ideal 
        self.assertNumpyArrayAlmostEqual(constant.x.as_array() , self.ideal.as_array(), decimal=4)
        self.assertNumpyArrayAlmostEqual(adaptive.x.as_array() , self.ideal.as_array(), decimal=2)

    def test_convergence_angles(self):
        # Companion to test_convergence_balancing for the angle/alignment rule (rule (b)
        # of Chambolle et al. 2023). This rule *increases* the primal step when successive
        # primal directions stay aligned, so the imbalanced start is chosen the other way
        # round: tau far too small (sigma filled in from the operator norms). From there it
        # should grow tau, reduce the objective and reach a value comparable to a well-tuned
        # constant rule, converging to the same minimiser.
        # the adaptive run recovers from the bad start but, having spent early iterations
        # rebalancing, is a little behind the well-tuned constant rule at 50 iterations.
        # SPDHG samples subsets at random, so both runs are seeded for reproducibility.
        seed = 7
        constant = SPDHG(f=self.F, g=self.G, operator=self.A,
                         sampler=Sampler.random_with_replacement(self.subsets, seed=seed))
        constant.run(50, verbose=0)

        rule = SPDHGAdaptiveStepSizeAngle(initial_step_size=[0.001, None])
        adaptive = SPDHG(f=self.F, g=self.G, operator=self.A, step_size=rule,
                         sampler=Sampler.random_with_replacement(self.subsets, seed=seed))
        adaptive.run(50, verbose=0)
        # check the objective has decreased from its initial value (the first entry is the initial value)
        self.assertLess(adaptive.objective[-1], adaptive.objective[0])
        # Check both get close tot he ideal 
        self.assertNumpyArrayAlmostEqual(constant.x.as_array() , self.ideal.as_array(), decimal=4)
        self.assertNumpyArrayAlmostEqual(adaptive.x.as_array() , self.ideal.as_array(), decimal=2)