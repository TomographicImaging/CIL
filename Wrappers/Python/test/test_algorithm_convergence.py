from cil.optimisation.algorithms import SPDHG, PDHG, GD, PD3O
from cil.optimisation.functions import L2NormSquared, IndicatorBox, BlockFunction, ZeroFunction, LeastSquares, TotalVariation, MixedL21Norm
from cil.optimisation.operators import BlockOperator, IdentityOperator, MatrixOperator, GradientOperator
from cil.optimisation.utilities import Sampler, BarzilaiBorweinStepSizeRule

from cil.framework import AcquisitionGeometry, BlockDataContainer, BlockGeometry, VectorData

from cil.utilities import dataexample

import numpy as np
import unittest
from testclass import CCPiTestClass

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
    
    
    
class TestAlgorithmConvergence(CCPiTestClass):    
    @unittest.skipUnless(has_astra, "cil-astra not available")
    def test_SPDHG_num_subsets_1(self):
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
        

    def test_pd3o_convergence(self):
        data = dataexample.CAMERA.get(size=(32, 32))
        # pd30 convergence test using TV denoising

        # regularisation parameter
        alpha = 0.11

        # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
        TV = TotalVariation(max_iteration=200)
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
        pd3O_with_f.run(1000)

        # pd30 vs fista
        np.testing.assert_allclose(
            tv_cil.array, pd3O_with_f.solution.array, atol=1e-2)
        
        
    def test_bb_step_size_gd_converge(self):
        np.random.seed(2)
        n = 10
        m = 10
        A = np.array(range(1,n*m+1)).reshape(n,m).astype('float32')
        A = np.diag(1/(np.transpose(A)@np.ones(m)))*A
        x = (np.array(range(n)).astype('float32')-n/2)/n
        b=A@x


        Aop = MatrixOperator(A)
        bop = VectorData(b)
        ig=Aop.domain
        
        initial = VectorData((np.array(range(n)).astype('float32')-n/2)/(n+1))
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
