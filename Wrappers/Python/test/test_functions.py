# -*- coding: utf-8 -*-
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

import unittest

from cil.optimisation.functions.Function import ScaledFunction
import numpy as np

from cil.framework import DataContainer, ImageGeometry, \
    VectorGeometry, VectorData, BlockDataContainer
from cil.optimisation.operators import IdentityOperator, MatrixOperator, CompositionOperator, DiagonalOperator, BlockOperator
from cil.optimisation.functions import Function, KullbackLeibler, ConstantFunction, TranslateFunction
from cil.optimisation.operators import GradientOperator

from cil.optimisation.functions import Function, KullbackLeibler, WeightedL2NormSquared, L2NormSquared,\
                                         L1Norm, MixedL21Norm, LeastSquares, \
                                         SmoothMixedL21Norm, OperatorCompositionFunction,\
                                         Rosenbrock, IndicatorBox, TotalVariation
from cil.optimisation.functions import BlockFunction

import numpy
import scipy.special

from cil.framework import ImageGeometry, BlockGeometry
from cil.optimisation.functions import TranslateFunction
from timeit import default_timer as timer

import numpy as np
from cil.utilities import dataexample
from cil.utilities import noise
from testclass import CCPiTestClass
from cil.utilities.quality_measures import mae
import cil.utilities.multiprocessing as cilmp

from utils import has_ccpi_regularisation, has_tomophantom, has_numba, initialise_tests
import numba

initialise_tests()

if has_ccpi_regularisation:
    from cil.plugins.ccpi_regularisation.functions import FGP_TV

if has_tomophantom:
    from cil.plugins import TomoPhantom

if has_numba:
    from cil.optimisation.functions.MixedL21Norm import _proximal_step_numba, _proximal_step_numpy


class TestFunction(CCPiTestClass):

    def test_Function(self):
        numpy.random.seed(10)
        N = 3
        ig = ImageGeometry(N, N)
        ag = ig
        op1 = GradientOperator(ig)
        op2 = IdentityOperator(ig, ag)

        # Form Composite Operator
        operator = BlockOperator(op1, op2, shape=(2, 1))

        # Create functions
        noisy_data = ag.allocate(ImageGeometry.RANDOM, dtype=numpy.float64)

        d = ag.allocate(ImageGeometry.RANDOM, dtype=numpy.float64)
        alpha = 0.5

        # scaled function
        g = alpha * L2NormSquared(b=noisy_data)

        # Compare call of g
        a2 = alpha * (d - noisy_data).power(2).sum()

        self.assertEqual(a2, g(d))

        # Compare convex conjugate of g
        a3 = 0.5 * d.squared_norm() + d.dot(noisy_data)
        self.assertAlmostEqual(a3, g.convex_conjugate(d), places=7)

    def test_L2NormSquared(self):
        # TESTS for L2 and scalar * L2
        numpy.random.seed(1)
        M, N, K = 2, 3, 5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N, voxel_num_z=K)
        u = ig.allocate(ImageGeometry.RANDOM)
        b = ig.allocate(ImageGeometry.RANDOM)

        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = 2 * u
        numpy.testing.assert_array_almost_equal(a1.as_array(),
                                                a2.as_array(),
                                                decimal=4)
        numpy.testing.assert_equal(f(u), u.squared_norm())

        # check grad/call with data
        f1 = L2NormSquared(b=b)
        b1 = f1.gradient(u)
        b2 = 2 * (u - b)

        numpy.testing.assert_array_almost_equal(b1.as_array(),
                                                b2.as_array(),
                                                decimal=4)
        numpy.testing.assert_equal(f1(u), (u - b).squared_norm())

        #check convex conjuagate no data
        c1 = f.convex_conjugate(u)
        c2 = 1 / 4. * u.squared_norm()
        numpy.testing.assert_equal(c1, c2)

        #check convex conjugate with data
        d1 = f1.convex_conjugate(u)
        d2 = (1. / 4.) * u.squared_norm() + (u * b).sum()
        numpy.testing.assert_almost_equal(d1, d2, decimal=6)

        # check proximal no data
        tau = 5
        e1 = f.proximal(u, tau)
        e2 = u / (1 + 2 * tau)
        numpy.testing.assert_array_almost_equal(e1.as_array(),
                                                e2.as_array(),
                                                decimal=4)

        # check proximal with data
        tau = 5
        h1 = f1.proximal(u, tau)
        h2 = (u - b) / (1 + 2 * tau) + b
        numpy.testing.assert_array_almost_equal(h1.as_array(),
                                                h2.as_array(),
                                                decimal=4)

        # check proximal conjugate no data
        tau = 0.2
        k1 = f.proximal_conjugate(u, tau)
        k2 = u / (1 + tau / 2)
        numpy.testing.assert_array_almost_equal(k1.as_array(),
                                                k2.as_array(),
                                                decimal=4)

        # check proximal conjugate with data
        l1 = f1.proximal_conjugate(u, tau)
        l2 = (u - tau * b) / (1 + tau / 2)
        numpy.testing.assert_array_almost_equal(l1.as_array(),
                                                l2.as_array(),
                                                decimal=4)

        # check scaled function properties

        # scalar
        scalar = 100
        f_scaled_no_data = scalar * L2NormSquared()
        f_scaled_data = scalar * L2NormSquared(b=b)

        # call
        numpy.testing.assert_equal(f_scaled_no_data(u), scalar * f(u))
        numpy.testing.assert_equal(f_scaled_data(u), scalar * f1(u))

        # grad
        numpy.testing.assert_array_almost_equal(
            f_scaled_no_data.gradient(u).as_array(),
            scalar * f.gradient(u).as_array(),
            decimal=4)
        numpy.testing.assert_array_almost_equal(
            f_scaled_data.gradient(u).as_array(),
            scalar * f1.gradient(u).as_array(),
            decimal=4)

        # conj
        numpy.testing.assert_almost_equal(f_scaled_no_data.convex_conjugate(u), \
                                f.convex_conjugate(u/scalar) * scalar, decimal=4)

        numpy.testing.assert_almost_equal(f_scaled_data.convex_conjugate(u), \
                                scalar * f1.convex_conjugate(u/scalar), decimal=4)

        # proximal
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal(u, tau).as_array(), \
                                                f.proximal(u, tau*scalar).as_array())


        numpy.testing.assert_array_almost_equal(f_scaled_data.proximal(u, tau).as_array(), \
                                                f1.proximal(u, tau*scalar).as_array())

        # proximal conjugate
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal_conjugate(u, tau).as_array(), \
                                                (u/(1 + tau/(2*scalar) )).as_array(), decimal=4)

        numpy.testing.assert_array_almost_equal(f_scaled_data.proximal_conjugate(u, tau).as_array(), \
                                                ((u - tau * b)/(1 + tau/(2*scalar) )).as_array(), decimal=4)

    def test_L2NormSquaredOut(self):
        # TESTS for L2 and scalar * L2

        M, N, K = 2, 3, 5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N, voxel_num_z=K)
        u = ig.allocate(ImageGeometry.RANDOM, seed=1)
        b = ig.allocate(ImageGeometry.RANDOM, seed=2)

        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = a1 * 0.
        f.gradient(u, out=a2)
        numpy.testing.assert_array_almost_equal(a1.as_array(),
                                                a2.as_array(),
                                                decimal=4)
        #numpy.testing.assert_equal(f(u), u.squared_norm())

        # check grad/call with data
        f1 = L2NormSquared(b=b)
        b1 = f1.gradient(u)
        b2 = b1 * 0.
        f1.gradient(u, out=b2)

        numpy.testing.assert_array_almost_equal(b1.as_array(),
                                                b2.as_array(),
                                                decimal=4)
        #numpy.testing.assert_equal(f1(u), (u-b).squared_norm())

        # check proximal no data
        tau = 5
        e1 = f.proximal(u, tau)
        e2 = e1 * 0.
        f.proximal(u, tau, out=e2)
        numpy.testing.assert_array_almost_equal(e1.as_array(),
                                                e2.as_array(),
                                                decimal=4)

        # check proximal with data
        tau = 5
        h1 = f1.proximal(u, tau)
        h2 = h1 * 0.
        f1.proximal(u, tau, out=h2)
        numpy.testing.assert_array_almost_equal(h1.as_array(),
                                                h2.as_array(),
                                                decimal=4)

        # check proximal conjugate no data
        tau = 0.2
        k1 = f.proximal_conjugate(u, tau)
        k2 = k1 * 0.
        f.proximal_conjugate(u, tau, out=k2)

        numpy.testing.assert_array_almost_equal(k1.as_array(),
                                                k2.as_array(),
                                                decimal=4)

        # check proximal conjugate with data
        l1 = f1.proximal_conjugate(u, tau)
        l2 = l1 * 0.
        f1.proximal_conjugate(u, tau, out=l2)
        numpy.testing.assert_array_almost_equal(l1.as_array(),
                                                l2.as_array(),
                                                decimal=4)

        # check scaled function properties

        # scalar
        scalar = 100
        f_scaled_no_data = scalar * L2NormSquared()
        f_scaled_data = scalar * L2NormSquared(b=b)

        # grad
        w = f_scaled_no_data.gradient(u)
        ww = w * 0
        f_scaled_no_data.gradient(u, out=ww)

        numpy.testing.assert_array_almost_equal(w.as_array(),
                                                ww.as_array(),
                                                decimal=4)

        # numpy.testing.assert_array_almost_equal(f_scaled_data.gradient(u).as_array(), scalar*f1.gradient(u).as_array(), decimal=4)

        # # conj
        # numpy.testing.assert_almost_equal(f_scaled_no_data.convex_conjugate(u), \
        #                         f.convex_conjugate(u/scalar) * scalar, decimal=4)

        # numpy.testing.assert_almost_equal(f_scaled_data.convex_conjugate(u), \
        #                         scalar * f1.convex_conjugate(u/scalar), decimal=4)

        # # proximal
        w = f_scaled_no_data.proximal(u, tau)
        ww = w * 0
        f_scaled_no_data.proximal(u, tau, out=ww)
        numpy.testing.assert_array_almost_equal(w.as_array(), \
                                                ww.as_array())

        # numpy.testing.assert_array_almost_equal(f_scaled_data.proximal(u, tau).as_array(), \
        #                                         f1.proximal(u, tau*scalar).as_array())

        # proximal conjugate
        w = f_scaled_no_data.proximal_conjugate(u, tau)
        ww = w * 0
        f_scaled_no_data.proximal_conjugate(u, tau, out=ww)
        numpy.testing.assert_array_almost_equal(w.as_array(), \
                                                ww.as_array(), decimal=4)

        # numpy.testing.assert_array_almost_equal(f_scaled_data.proximal_conjugate(u, tau).as_array(), \
        #                                         ((u - tau * b)/(1 + tau/(2*scalar) )).as_array(), decimal=4)

    def test_Norm2sq_as_OperatorCompositionFunction(self):
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N)
        #numpy.random.seed(1)
        b = ig.allocate('random', seed=1)

        operator = 3 * IdentityOperator(ig)

        u = ig.allocate('random', seed=50)
        f = 0.5 * L2NormSquared(b=b)
        func1 = OperatorCompositionFunction(f, operator)
        func2 = LeastSquares(operator, b, 0.5)

        numpy.testing.assert_almost_equal(func1(u), func2(u))

        tmp1 = ig.allocate()
        tmp2 = ig.allocate()
        res_gradient1 = func1.gradient(u)
        res_gradient2 = func2.gradient(u)
        func1.gradient(u, out=tmp1)
        func2.gradient(u, out=tmp2)

        self.assertNumpyArrayAlmostEqual(res_gradient1.as_array(),
                                         res_gradient2.as_array())
        self.assertNumpyArrayAlmostEqual(tmp1.as_array(), tmp2.as_array())

        mat = np.random.randn(M, N)
        operator = MatrixOperator(mat)
        vg = VectorGeometry(N)
        b = vg.allocate('random')
        u = vg.allocate('random')

        func1 = OperatorCompositionFunction(0.5 * L2NormSquared(b=b), operator)
        func2 = LeastSquares(operator, b, 0.5)

        self.assertNumpyArrayAlmostEqual(func1(u), func2(u))
        numpy.testing.assert_almost_equal(func1.L, func2.L)

    def test_MixedL21Norm(self):
        numpy.random.seed(1)
        M, N, K = 2, 3, 5
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N)
        u1 = ig.allocate('random')
        u2 = ig.allocate('random')

        U = BlockDataContainer(u1, u2, shape=(2, 1))

        # Define no scale and scaled
        f_no_scaled = MixedL21Norm()
        f_scaled = 1 * MixedL21Norm()

        # call

        a1 = f_no_scaled(U)
        a2 = f_scaled(U)
        self.assertNumpyArrayAlmostEqual(a1, a2)

        tmp = [el**2 for el in U.containers]
        self.assertBlockDataContainerEqual(BlockDataContainer(*tmp),
                                           U.power(2))

        z1 = f_no_scaled.proximal_conjugate(U, 1)
        u3 = ig.allocate('random')
        u4 = ig.allocate('random')

        z3 = BlockDataContainer(u3, u4, shape=(2, 1))

        f_no_scaled.proximal_conjugate(U, 1, out=z3)
        self.assertBlockDataContainerAlmostEqual(z3, z1, decimal=5)

    @unittest.skipUnless(has_numba, 'Skipping as numba is not installed')
    def test_MixedL21Norm_step(self):
        data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        g = GradientOperator(data.geometry)

        y = g.direct(data)
        gamma = 2.

        # test proximax step with numba
        absgamma = np.abs(gamma, dtype=np.float32)
        # test numba
        tmp = y.pnorm(2)
        tmp = np.asarray(tmp.as_array(), order='C', dtype=np.float32)

        res1 = _proximal_step_numba(tmp, absgamma)

        # test proximax step with numpy
        tmp = y.pnorm(2)
        res2 = _proximal_step_numpy(tmp, gamma)

        # check they are the same
        np.testing.assert_allclose(res1, res2.as_array(), atol=1e-5, rtol=1e-6)

    def test_MixedL21Norm_proximal_step_numpy_float(self):
        from cil.optimisation.functions.MixedL21Norm import _proximal_step_numpy
        from cil.framework import ImageGeometry

        tau = 1.1
        
        ig = ImageGeometry(2,3,4)
        tmp = ig.allocate(1)
        a = _proximal_step_numpy(tmp, tau)

        b = _proximal_step_numpy(tmp, -tau)

        np.testing.assert_allclose(a.as_array(), b.as_array())

    def test_MixedL21Norm_proximal_step_numpy_dc(self):
        from cil.optimisation.functions.MixedL21Norm import _proximal_step_numpy
        from cil.framework import ImageGeometry

        
        ig = ImageGeometry(2,3,4)
        tmp = ig.allocate(1)
        tau = ig.allocate(2)
        a = _proximal_step_numpy(tmp, tau)
        
        tau *= -1
        b = _proximal_step_numpy(tmp, tau)

        np.testing.assert_allclose(a.as_array(), b.as_array())
    
    def test_MixedL21Norm_proximal_step_numpy_ndarray(self):
        from cil.optimisation.functions.MixedL21Norm import _proximal_step_numpy
        from cil.framework import ImageGeometry

        
        ig = ImageGeometry(2,3,4)
        tmp = ig.allocate(1)
        tau = ig.allocate(2)
        tauarr = tau.as_array()
        a = _proximal_step_numpy(tmp, tauarr)
        
        tauarr *= -1
        b = _proximal_step_numpy(tmp, tauarr)

        np.testing.assert_allclose(a.as_array(), b.as_array())
    
    
    def test_smoothL21Norm(self):
        ig = ImageGeometry(4, 5)
        bg = BlockGeometry(ig, ig)

        epsilon = 0.5

        f1 = SmoothMixedL21Norm(epsilon)
        x = bg.allocate('random', seed=10)

        # check call
        res1 = f1(x)
        res2 = (x.pnorm(2)**2 + epsilon**2).sqrt().sum()

        # alternative
        tmp1 = x.copy()
        tmp1.containers += (epsilon, )
        res3 = tmp1.pnorm(2).sum()

        np.testing.assert_almost_equal(res1, res2, decimal=5)
        np.testing.assert_almost_equal(res1, res3, decimal=5)

        res1 = f1.gradient(x)
        res2 = x.divide((x.pnorm(2)**2 + epsilon**2).sqrt())
        np.testing.assert_array_almost_equal(
            res1.get_item(0).as_array(),
            res2.get_item(0).as_array())

        np.testing.assert_array_almost_equal(
            res1.get_item(1).as_array(),
            res2.get_item(1).as_array())

        # check with MixedL21Norm, when epsilon close to 0

        f1 = SmoothMixedL21Norm(1e-12)
        f2 = MixedL21Norm()

        res1 = f1(x)
        res2 = f2(x)
        np.testing.assert_almost_equal(f1(x), f2(x))

    def test_KullbackLeibler(self):
        #numpy.random.seed(1)
        M, N, K = 2, 3, 4
        ig = ImageGeometry(N, M, K)

        u1 = ig.allocate('random', seed=500)
        g1 = ig.allocate('random', seed=100)
        b1 = ig.allocate('random', seed=1000)

        # with no data
        with self.assertRaises(TypeError):
            f = KullbackLeibler()

        with self.assertRaises(ValueError):
            f = KullbackLeibler(b=-1 * g1)

        f = KullbackLeibler(b=g1)
        self.assertNumpyArrayAlmostEqual(0.0, f(g1))

        res_gradient = f.gradient(u1)
        res_gradient_out = u1.geometry.allocate()
        f.gradient(u1, out=res_gradient_out)
        self.assertNumpyArrayAlmostEqual(res_gradient.as_array(), \
                                                res_gradient_out.as_array(),decimal = 4)

        tau = 400.4
        res_proximal = f.proximal(u1, tau)
        res_proximal_out = u1.geometry.allocate()
        f.proximal(u1, tau, out=res_proximal_out)
        self.assertNumpyArrayAlmostEqual(res_proximal.as_array(), \
                                                res_proximal_out.as_array(), decimal =5)

        u2 = u1 * 0 + 2.
        self.assertNumpyArrayAlmostEqual(0.0, f.convex_conjugate(u2))
        eta = b1

        f1 = KullbackLeibler(b=g1, eta=b1)

        tmp_sum = (u1 + eta).as_array()
        ind = tmp_sum >= 0
        tmp = scipy.special.kl_div(f1.b.as_array()[ind], tmp_sum[ind])
        self.assertNumpyArrayAlmostEqual(f1(u1), numpy.sum(tmp))

        res_proximal_conj_out = u1.geometry.allocate()
        proxc = f.proximal_conjugate(u1, tau)
        f.proximal_conjugate(u1, tau, out=res_proximal_conj_out)
        numpy.testing.assert_array_almost_equal(
            proxc.as_array(), res_proximal_conj_out.as_array())

    def test_Rosenbrock(self):
        f = Rosenbrock(alpha=1, beta=100)
        x = VectorData(numpy.asarray([1, 1]))
        assert f(x) == 0.
        numpy.testing.assert_array_almost_equal(
            f.gradient(x).as_array(),
            numpy.zeros(shape=(2, ), dtype=numpy.float32))

    def tests_for_L2NormSq_and_weighted(self):
        numpy.random.seed(1)
        M, N, K = 2, 3, 1
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N, voxel_num_z=K)
        u = ig.allocate('random')
        b = ig.allocate('random')

        # check grad/call no data
        f = L2NormSquared()
        a1 = f.gradient(u)
        a2 = 2 * u
        numpy.testing.assert_array_almost_equal(a1.as_array(),
                                                a2.as_array(),
                                                decimal=4)
        numpy.testing.assert_equal(f(u), u.squared_norm())

        # check grad/call with data

        #        igggg = ImageGeometry(4,4)
        f1 = L2NormSquared(b=b)
        b1 = f1.gradient(u)
        b2 = 2 * (u - b)

        numpy.testing.assert_array_almost_equal(b1.as_array(),
                                                b2.as_array(),
                                                decimal=4)
        numpy.testing.assert_equal(f1(u), ((u - b)).squared_norm())

        #check convex conjuagate no data
        c1 = f.convex_conjugate(u)
        c2 = 1 / 4 * u.squared_norm()
        numpy.testing.assert_equal(c1, c2)

        #check convex conjuagate with data
        d1 = f1.convex_conjugate(u)
        d2 = (1 / 4) * u.squared_norm() + u.dot(b)
        numpy.testing.assert_equal(d1, d2)

        # check proximal no data
        tau = 5
        e1 = f.proximal(u, tau)
        e2 = u / (1 + 2 * tau)
        numpy.testing.assert_array_almost_equal(e1.as_array(),
                                                e2.as_array(),
                                                decimal=4)

        # check proximal with data
        tau = 5
        h1 = f1.proximal(u, tau)
        h2 = (u - b) / (1 + 2 * tau) + b
        numpy.testing.assert_array_almost_equal(h1.as_array(),
                                                h2.as_array(),
                                                decimal=4)

        # check proximal conjugate no data
        tau = 0.2
        k1 = f.proximal_conjugate(u, tau)
        k2 = u / (1 + tau / 2)
        numpy.testing.assert_array_almost_equal(k1.as_array(),
                                                k2.as_array(),
                                                decimal=4)

        # check proximal conjugate with data
        l1 = f1.proximal_conjugate(u, tau)
        l2 = (u - tau * b) / (1 + tau / 2)
        numpy.testing.assert_array_almost_equal(l1.as_array(),
                                                l2.as_array(),
                                                decimal=4)

        # check scaled function properties
        # scalar
        scalar = 100
        f_scaled_no_data = scalar * L2NormSquared()
        f_scaled_data = scalar * L2NormSquared(b=b)

        # call
        numpy.testing.assert_equal(f_scaled_no_data(u), scalar * f(u))
        numpy.testing.assert_equal(f_scaled_data(u), scalar * f1(u))

        # grad
        numpy.testing.assert_array_almost_equal(
            f_scaled_no_data.gradient(u).as_array(),
            scalar * f.gradient(u).as_array(),
            decimal=4)
        numpy.testing.assert_array_almost_equal(
            f_scaled_data.gradient(u).as_array(),
            scalar * f1.gradient(u).as_array(),
            decimal=4)

        # conj
        numpy.testing.assert_almost_equal(f_scaled_no_data.convex_conjugate(u), \
                                   f.convex_conjugate(u/scalar) * scalar, decimal=4)

        numpy.testing.assert_almost_equal(f_scaled_data.convex_conjugate(u), \
                                   scalar * f1.convex_conjugate(u/scalar), decimal=4)

        # proximal
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal(u, tau).as_array(), \
                                                f.proximal(u, tau*scalar).as_array())


        numpy.testing.assert_array_almost_equal(f_scaled_data.proximal(u, tau).as_array(), \
                                                f1.proximal(u, tau*scalar).as_array())

        # proximal conjugate
        numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal_conjugate(u, tau).as_array(), \
                                                (u/(1 + tau/(2*scalar) )).as_array(), decimal=4)

        numpy.testing.assert_array_almost_equal(f_scaled_data.proximal_conjugate(u, tau).as_array(), \
                                                ((u - tau * b)/(1 + tau/(2*scalar) )).as_array(), decimal=4)

        u_out_no_out = ig.allocate('random_int')
        res_no_out = f_scaled_data.proximal_conjugate(u_out_no_out, 0.5)

        res_out = ig.allocate()
        f_scaled_data.proximal_conjugate(u_out_no_out, 0.5, out=res_out)

        numpy.testing.assert_array_almost_equal(res_no_out.as_array(), \
                                                res_out.as_array(), decimal=4)

        ig1 = ImageGeometry(2, 3)

        tau = 0.1

        u = ig1.allocate('random')
        b = ig1.allocate('random')

        scalar = 0.5
        f_scaled = scalar * L2NormSquared(b=b)
        f_noscaled = L2NormSquared(b=b)

        res1 = f_scaled.proximal(u, tau)
        res2 = f_noscaled.proximal(u, tau * scalar)

        # res2 = (u + tau*b)/(1+tau)

        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)

        # Tests for weighted L2NormSquared
        ig = ImageGeometry(voxel_num_x=3, voxel_num_y=3)
        weight = ig.allocate('random')

        f = WeightedL2NormSquared(weight=weight)
        x = ig.allocate(0.4)

        res1 = f(x)
        res2 = (weight * (x**2)).sum()
        numpy.testing.assert_almost_equal(res1, res2, decimal=4)

        # gradient for weighted L2NormSquared
        res1 = f.gradient(x)
        res2 = 2 * weight * x
        out = ig.allocate()
        f.gradient(x, out=out)
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                out.as_array(), decimal=4)
        numpy.testing.assert_array_almost_equal(res2.as_array(), \
                                                out.as_array(), decimal=4)

        # convex conjugate for weighted L2NormSquared
        res1 = f.convex_conjugate(x)
        res2 = 1 / 4 * (x / weight.sqrt()).squared_norm()
        numpy.testing.assert_array_almost_equal(res1, \
                                                res2, decimal=4)

        # proximal for weighted L2NormSquared
        tau = 0.3
        out = ig.allocate()
        res1 = f.proximal(x, tau)
        f.proximal(x, tau, out=out)
        res2 = x / (1 + 2 * tau * weight)
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)
        tau = 0.3
        out = ig.allocate()
        res1 = f.proximal_conjugate(x, tau)
        res2 = x / (1 + tau / (2 * weight))
        numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                                res2.as_array(), decimal=4)

        b = ig.allocate('random')
        f1 = TranslateFunction(WeightedL2NormSquared(weight=weight), b)
        f2 = WeightedL2NormSquared(weight=weight, b=b)
        res1 = f1(x)
        res2 = f2(x)
        numpy.testing.assert_almost_equal(res1, res2, decimal=4)

        f1 = WeightedL2NormSquared(b=b)
        f2 = L2NormSquared(b=b)

        numpy.testing.assert_almost_equal(f1.L, f2.L, decimal=4)
        numpy.testing.assert_almost_equal(f1.L, 2, decimal=4)
        numpy.testing.assert_almost_equal(f2.L, 2, decimal=4)

    def tests_for_LS_weightedLS(self):
        ig = ImageGeometry(40, 30)

        numpy.random.seed(1)

        A = IdentityOperator(ig)
        b = ig.allocate('random')
        x = ig.allocate('random')
        c = numpy.float64(0.3)

        weight = ig.allocate('random')

        D = DiagonalOperator(weight)
        norm_weight = numpy.float64(D.norm())

        f1 = LeastSquares(A, b, c, weight)
        f2 = LeastSquares(A, b, c)

        # check Lipshitz
        numpy.testing.assert_almost_equal(f2.L, 2 * c * (A.norm()**2))
        numpy.testing.assert_almost_equal(
            f1.L,
            numpy.float64(2.) * c * norm_weight * (A.norm()**2))

        # check call with weight
        res1 = c * (A.direct(x) - b).dot(weight * (A.direct(x) - b))
        res2 = f1(x)
        numpy.testing.assert_almost_equal(res1, res2)

        # check call without weight
        #res1 = c * (A.direct(x)-b).dot((A.direct(x) - b))
        res1 = c * (A.direct(x) - b).squared_norm()
        res2 = f2(x)
        numpy.testing.assert_almost_equal(res1, res2)

        # check gradient with weight
        out = ig.allocate(None)
        res1 = f1.gradient(x)
        #out = f1.gradient(x)
        f1.gradient(x, out=out)
        res2 = 2 * c * A.adjoint(weight * (A.direct(x) - b))
        numpy.testing.assert_array_almost_equal(res1.as_array(),
                                                res2.as_array())
        numpy.testing.assert_array_almost_equal(out.as_array(),
                                                res2.as_array())

        # check gradient without weight
        out = ig.allocate()
        res1 = f2.gradient(x)
        f2.gradient(x, out=out)
        res2 = 2 * c * A.adjoint(A.direct(x) - b)
        numpy.testing.assert_array_almost_equal(res1.as_array(),
                                                res2.as_array())
        numpy.testing.assert_array_almost_equal(out.as_array(),
                                                res2.as_array())

        ig2 = ImageGeometry(100, 100, 100)
        A = IdentityOperator(ig2)
        b = ig2.allocate('random')
        x = ig2.allocate('random')
        c = 0.3

        weight = ig2.allocate('random')

        weight_operator = DiagonalOperator(weight.sqrt())
        tmp_A = CompositionOperator(weight_operator, A)
        tmp_b = weight_operator.direct(b)

        f1 = LeastSquares(tmp_A, tmp_b, c)
        f2 = LeastSquares(A, b, c, weight)

        t0 = timer()
        res1 = f1(x)
        t1 = timer()

        t2 = timer()
        res2 = f2(x)
        t3 = timer()

        numpy.testing.assert_almost_equal(res1, res2, decimal=2)

    def test_Lipschitz(self):
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N)
        b = ig.allocate('random', seed=1)
        operator = 3 * IdentityOperator(ig)

        u = ig.allocate('random_int', seed=50)
        func2 = LeastSquares(operator, b, 0.5)
        assert func2.L != 2
        func2.L = 2
        assert func2.L == 2

    def test_Lipschitz2(self):
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N)
        b = ig.allocate('random', seed=1)
        operator = 3 * IdentityOperator(ig)

        u = ig.allocate('random_int', seed=50)
        func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        f3 = func1 + func2
        assert f3.L != 2
        func2.L = 2
        assert func2.L == 2

    def test_Lipschitz3(self):
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N)
        b = ig.allocate('random', seed=1)
        operator = 3 * IdentityOperator(ig)

        u = ig.allocate('random_int', seed=50)
        # func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        f3 = TranslateFunction(func1, 3)
        assert f3.L != 2
        f3.L = 2
        assert f3.L == 2

    def test_Lipschitz4(self):
        M, N = 50, 50
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N)
        b = ig.allocate('random', seed=1)
        operator = 3 * IdentityOperator(ig)

        u = ig.allocate('random_int', seed=50)
        # func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        f3 = func1 + 3
        assert f3.L == 0
        f3.L = 2
        assert f3.L == 2
        assert func1.L == 0
        with self.assertRaises(AttributeError):
            func1.L = 2

        f2 = LeastSquares(operator, b, 0.5)
        f4 = 2 * f2
        assert f4.L == 2 * f2.L

        f4.L = 10
        assert f4.L != 2 * f2.L

        f4 = -2 * f2
        assert f4.L == 2 * f2.L

    def test_proximal_conjugate(self):
        from cil.framework import AcquisitionGeometry, BlockGeometry
        ag = AcquisitionGeometry.create_Parallel2D()
        angles = np.linspace(0, 360, 10, dtype=np.float32)

        #default
        ag.set_angles(angles)
        ag.set_panel(5)

        ig = ag.get_ImageGeometry()
        bg = BlockGeometry(ig, ig)

        b = ag.allocate('random', seed=2)

        func_geom_test_list = [
            (IndicatorBox(), ag),
            (KullbackLeibler(b=b, backend='numba'), ag),
            (KullbackLeibler(b=b, backend='numpy'), ag),
            (L1Norm(), ag),
            (L2NormSquared(), ag),
            (MixedL21Norm(), bg),
            (TotalVariation(backend='c', warm_start=False, max_iteration=100), ig),
            (TotalVariation(backend='numpy', warm_start=False, max_iteration=100), ig),
        ]
        
        for func, geom in func_geom_test_list:
            self.proximal_conjugate_test(func, geom)

    def proximal_conjugate_test(self, function, geom):
        x = geom.allocate('random', seed=1)
        tau = 1.0
        f = Function()
        f.proximal = function.proximal

        a = function.proximal_conjugate(x, tau)
        b = f.proximal_conjugate(x, tau)

        # handle the case of MixedL21Norm
        if isinstance(a, BlockDataContainer):
            for xa,xb in zip(a,b):
                np.testing.assert_allclose(xa.as_array(), xb.as_array(),
                                    rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(a.as_array(), b.as_array(),
                                    rtol=1e-5, atol=1e-5)


class TestTotalVariation(unittest.TestCase):

    def setUp(self) -> None:
        self.tv = TotalVariation()
        self.alpha = 0.15
        self.tv_scaled = self.alpha * TotalVariation()
        self.tv_iso = TotalVariation()
        self.tv_aniso = TotalVariation(isotropic=False)
        self.ig_real = ImageGeometry(3, 4)
        self.grad = GradientOperator(self.ig_real)
        self.alpha_arr = self.ig_real.allocate(0.15)

    def test_configure_tv_defaults(self):
        self.assertEquals(self.tv.warm_start, True)
        self.assertEquals(self.tv.iterations, 10)
        self.assertEquals(self.tv.correlation, "Space")
        self.assertEquals(self.tv.backend, "c")
        self.assertEquals(self.tv.lower, -np.inf)
        self.assertEquals(self.tv.upper, np.inf)
        self.assertEquals(self.tv.isotropic, True)
        self.assertEquals(self.tv.split, False)
        self.assertEquals(self.tv.info, False)
        self.assertEquals(self.tv.strong_convexity_constant, 0)
        self.assertEquals(self.tv.tolerance, None)

    def test_configure_tv_not_defaults(self):
        tv=TotalVariation( max_iteration=100, 
                 tolerance = 1e-5, 
                 correlation = "SpaceChannels",
                 backend = "numpy",
                 lower = 0.,
                 upper = 1.,
                 isotropic = False,
                 split = True,
                 info = True, 
                 strong_convexity_constant = 1.,
                 warm_start=False)
        self.assertEquals(tv.warm_start, False) 
        self.assertEquals(tv.iterations, 100)
        self.assertEquals(tv.correlation, "SpaceChannels")
        self.assertEquals(tv.backend, "numpy")
        self.assertEquals(tv.lower, 0.)
        self.assertEquals(tv.upper, 1.)
        self.assertEquals(tv.isotropic, False)
        self.assertEquals(tv.split, True)
        self.assertEquals(tv.info, True)
        self.assertEquals(tv.strong_convexity_constant, 1.)
        self.assertEquals(tv.tolerance, 1e-5)



    def test_scaled_regularisation_parameter(self):
        np.testing.assert_almost_equal(self.tv_scaled.regularisation_parameter,
                                       self.alpha, err_msg='Multiplying the TV functional did not change the regularisation parameter')

    def test_rmul(self):
        assert isinstance(self.tv_scaled, TotalVariation)

    def test_regularisation_parameter3(self):
        with self.assertRaises(TypeError):
            self.tv.regularisation_parameter = 'string'

    def test_rmul2(self):
        alpha = 'string'
        with self.assertRaises(TypeError):
            tv = alpha * TotalVariation()

    def test_rmul3(self):
        self.assertEqual(self.alpha, self.tv_scaled.regularisation_parameter, msg='Failed to scale the TV regularisation parameter')
        alpha = 2.
        tv2 = alpha * self.tv_scaled
        self.assertEqual(self.alpha * alpha, tv2.regularisation_parameter, msg="Failed at scaling regularisation parameter of already scaled TotalVariation")

    def test_call_real_isotropic(self):
        x_real = self.ig_real.allocate('random', seed=4)

        res1 = self.tv_iso(x_real)
        res2 = self.grad.direct(x_real).pnorm(2).sum()
        np.testing.assert_equal(res1, res2, err_msg="Error with isotropic TV calculation")

    def test_call_real_anisotropic(self):
        x_real = self.ig_real.allocate('random', seed=4)

        res1 = self.tv_aniso(x_real)
        res2 = self.grad.direct(x_real).pnorm(1).sum()
        np.testing.assert_almost_equal(res1, res2, decimal=3, err_msg="Error with anisotropic TV calculation")

    def test_strongly_convex_CIL_TV(self):

        TV_no_strongly_convex = self.alpha * TotalVariation()
        self.assertEqual(TV_no_strongly_convex.strong_convexity_constant, 0, msg="Error strong convexity constant not set to zero by default")

        # TV as strongly convex, with "small" strongly convex constant
        TV_strongly_convex = self.alpha * TotalVariation(
            strong_convexity_constant=1e-4)

        # check call
        x_real = self.ig_real.allocate('random', seed=4)
        res1 = TV_strongly_convex(x_real)
        res2 = TV_no_strongly_convex(
            x_real) + (TV_strongly_convex.strong_convexity_constant /
                       2) * x_real.squared_norm()
        np.testing.assert_allclose(res1, res2, atol=1e-3, err_msg='TV calculation with and without strong convexity not equal')

        # check proximal
        x_real = self.ig_real.allocate('random', seed=4)
        res1 = TV_no_strongly_convex.proximal(x_real, tau=1.0)

        tmp_x_real = x_real.copy()
        res2 = TV_strongly_convex.proximal(x_real, tau=1.0)
        # check input remain the same after proximal
        np.testing.assert_array_equal(tmp_x_real.array, x_real.array, err_msg='Input not the same after calling proximal')

        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3, err_msg="TV proximal calculation not the same with and without strong convexity")

    @unittest.skipUnless(has_ccpi_regularisation,
                         "Regularisation Toolkit not present")
    def test_strongly_convex_FGP_TV(self):

        FGP_TV_no_strongly_convex = self.alpha * FGP_TV()
        self.assertEqual(FGP_TV_no_strongly_convex.strong_convexity_constant,
                         0, msg="Default strong-convexity constant not set to zero")

        # TV as strongly convex, with "small" strongly convex constant
        FGP_TV_strongly_convex = self.alpha * FGP_TV(
            strong_convexity_constant=1e-3)

        # check call
        x_real = self.ig_real.allocate('random', seed=4)
        res1 = FGP_TV_strongly_convex(x_real)

        res2 = FGP_TV_no_strongly_convex(
            x_real) + (FGP_TV_strongly_convex.strong_convexity_constant /
                       2) * x_real.squared_norm()
        np.testing.assert_allclose(res1, res2, atol=1e-3, err_msg='Failed at comparing FGP_TV call with and without strong convexity')

        # check proximal
        x_real = self.ig_real.allocate('random', seed=4)
        res1 = FGP_TV_no_strongly_convex.proximal(x_real, tau=1.0)

        tmp_x_real = x_real.copy()
        res2 = FGP_TV_strongly_convex.proximal(x_real, tau=1.0)
        # check input remain the same after proximal
        np.testing.assert_array_equal(tmp_x_real.array, x_real.array, err_msg='Failed at checking input remains the same after calling proximal')

        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3, err_msg="For FGP_TV comparing prox with and without strong convexity")

    @unittest.skipUnless(has_ccpi_regularisation,
                         "Regularisation Toolkit not present")
    def test_compare_regularisation_toolkit(self):
        data = dataexample.SHAPES.get(size=(16, 16))
        ig = data.geometry
        ag = ig

        np.random.seed(0)
        # Create noisy data.
        n1 = np.random.normal(0, 0.0005, size=ig.shape)
        noisy_data = ig.allocate()
        noisy_data.fill(n1 + data.as_array())

        alpha = 0.1
        iters = 500

        # CIL_FGP_TV no tolerance
        g_CIL = alpha * TotalVariation(
            iters, tolerance=None, lower=0, info=True, warm_start=False)
        t0 = timer()
        res1 = g_CIL.proximal(noisy_data, 1.)
        t1 = timer()
        # print(t1-t0)

        r_alpha = alpha
        r_iterations = iters
        r_tolerance = 1e-9
        r_iso = True
        r_nonneg = True
        g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations,
                                            tolerance=r_tolerance,
                                            isotropic=r_iso,
                                            nonnegativity=r_nonneg,
                                            device='cpu')

        t2 = timer()
        res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
        t3 = timer()

        np.testing.assert_array_almost_equal(res1.as_array(),
                                             res2.as_array(),
                                             decimal=4, err_msg='Comparing the CCPi proximal against the CIL TV proximal, no tolerance')

        # print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with iterations.")
        iters = 408
        # CIL_FGP_TV with tolerance
        g_CIL = alpha * TotalVariation(iters, tolerance=1e-9, lower=0., warm_start=False)
        t0 = timer()
        res1 = g_CIL.proximal(noisy_data, 1.)
        t1 = timer()
        # print(t1-t0)

        r_alpha = alpha
        r_iterations = iters
        r_tolerance = 1e-9
        r_iso = True
        r_nonneg = True
        g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations,
                                            tolerance=r_tolerance,
                                            isotropic=r_iso,
                                            nonnegativity=r_nonneg,
                                            device='cpu')

        t2 = timer()
        res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
        t3 = timer()
        # print(t3-t2)
        np.testing.assert_array_almost_equal(res1.as_array(),
                                             res2.as_array(),
                                             decimal=3, err_msg='Comparing the CCPi proximal against the CIL TV proximal, with tolerance')

        # CIL_FGP_TV with warm_start
        iters=10
        g_CIL = alpha * TotalVariation(iters, lower=0., warm_start=True)
        for i in range(6):
            res1 = g_CIL.proximal(noisy_data, 1.)
        np.testing.assert_array_almost_equal(res1.as_array(),
                                             res2.as_array(),
                                             decimal=3, err_msg='Comparing the CCPi proximal against the CIL TV proximal, with warm_start')

    @unittest.skipUnless(has_tomophantom and has_ccpi_regularisation,
                         "Missing Tomophantom or Regularisation-Toolkit")
    def test_compare_regularisation_toolkit_tomophantom(self):
        # print ("Building 3D phantom using TomoPhantom software")
        model = 13  # select a model number from the library
        N_size =    16  # Define phantom dimensions using a scalar value (cubic phantom)
        #This will generate a N_size x N_size x N_size phantom (3D)

        ig = ImageGeometry(N_size, N_size, N_size)
        data = TomoPhantom.get_ImageData(num_model=model, geometry=ig)

        noisy_data = noise.gaussian(data, seed=10)

        alpha = 0.1
        iters = 100

        # print("Use tau as an array of ones")
        # CIL_TotalVariation no tolerance
        g_CIL = alpha * TotalVariation(iters, tolerance=None, info=True, warm_start=False)
        # res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
        t0 = timer()
        res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
        t1 = timer()
        # print(t1-t0)

        # CCPi Regularisation toolkit high tolerance

        r_alpha = alpha
        r_iterations = iters
        r_tolerance = 1e-9
        r_iso = True
        r_nonneg = True
        g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations,
                                            tolerance=r_tolerance,
                                            isotropic=r_iso,
                                            nonnegativity=r_nonneg,
                                            device='cpu')

        t2 = timer()
        res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
        t3 = timer()
        # print (t3-t2)

        np.testing.assert_allclose(res1.as_array(),
                                   res2.as_array(),
                                   atol=5e-2, err_msg='Comparing the CCPi proximal against the CIL TV proximal, without tolerance without warm_start')

        #with warm start 
        iters=10
        g_CIL = alpha * TotalVariation(iters, tolerance=None, info=True, warm_start=True)
        # res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
        t0 = timer()
        for i in range(4):
            res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
        t1 = timer()
        np.testing.assert_allclose(res1.as_array(),
                                   res2.as_array(),
                                   atol=4e-2, err_msg='Comparing the CCPi proximal against the CIL TV proximal, without tolerance with warm_start')


    def test_non_scalar_tau_cil_tv(self):

        x_real = self.ig_real.allocate('random', seed=4)

        # tau is an array filled with alpha = 0.15
        res1 = self.tv_iso.proximal(x_real, tau=self.alpha_arr)

        # use the alpha * TV
        res2 = self.tv_scaled.proximal(x_real, tau=1.0)

        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3, err_msg="Testing non scalar tau in prox calculation")

    def test_get_p2_with_warm_start(self):
        data = dataexample.SHAPES.get(size=(16, 16))
        tv=TotalVariation(warm_start=True, max_iteration=10)
        self.assertEquals(tv._p2, None, msg="tv._p2 not initialised to None")
        tv(data)
        checkp2=tv.gradient.range_geometry().allocate(0)
        for i, x in enumerate(tv._get_p2()):
                np.testing.assert_allclose(x.as_array(), checkp2[i].as_array(), rtol=1e-8, atol=1e-8, err_msg="P2 not initially set to zero")
        test=tv.proximal(data, 1.)
        print(test)
        a=np.sum(np.linalg.norm(test))
        print(np.linalg.norm(test))
        for i, x in enumerate(tv._get_p2()):
                np.testing.assert_equal(np.any(np.not_equal(x.as_array(), checkp2[i].as_array())), True, err_msg="The stored value of p2 doesn't change after calling proximal")
        np.testing.assert_almost_equal(np.sum(np.linalg.norm(test)),126.3372581, err_msg="Incorrect value of the proximal")
 
                
    def test_get_p2_without_warm_start(self):
        data = dataexample.SHAPES.get(size=(16, 16))
        tv=TotalVariation(warm_start=False)
        self.assertEquals(tv._p2, None, msg="tv._p2 not initialised to None")
        tv(data)
        checkp2=tv.gradient.range_geometry().allocate(0)
        for i, x in enumerate(tv._get_p2()):
                np.testing.assert_allclose(x.as_array(), checkp2[i].as_array(), rtol=1e-8, atol=1e-8, err_msg="P2 not initially set to zero")
        tv.proximal(data, 1.)
        for i, x in enumerate(tv._get_p2()):
                np.testing.assert_allclose(x.as_array(), checkp2[i].as_array(), rtol=1e-8, atol=1e-8, err_msg="P2 not reset to zero after a call to proximal")
 


class TestLeastSquares(unittest.TestCase):

    def setUp(self) -> None:
        ig = ImageGeometry(10, 2)
        A = IdentityOperator(ig)
        self.A = A
        self.ig = ig
        return super().setUp()

    def test_rmul(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        assert constant * ls.c == twicels.c

    def test_rmul_with_call(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        np.testing.assert_almost_equal(constant * ls(x), twicels(x))

    def test_rmul_with_Lipschitz(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        np.testing.assert_almost_equal(constant * ls.L, twicels.L)

    def test_rmul_with_gradient(self):
        ig = self.ig
        A = self.A
        b = ig.allocate(1)
        x = ig.allocate(3)
        c = 1.
        constant = 2.
        ls = LeastSquares(A, b, c=c)
        twicels = constant * ls

        y1 = ls.gradient(x)
        y2 = twicels.gradient(x)
        np.testing.assert_array_almost_equal(constant * y1.as_array(),
                                             y2.as_array())

        ls.gradient(x, out=y2)
        twicels.gradient(x, out=y1)
        np.testing.assert_array_almost_equal(constant * y2.as_array(),
                                             y1.as_array())


# tests for OperatorCompositionFunction
class TestOperatorCompositionFunctionWithWrongInterfaceFunction(
        unittest.TestCase):

    def setUp(self):
        ig = ImageGeometry(2, 2)
        I = IdentityOperator(ig)

        x = ig.allocate(0)

        class NotAFunction(object):
            pass

        ocf = OperatorCompositionFunction(NotAFunction(), I)
        self.pars = (ig, I, x, ocf)

    def tearDown(self):
        pass

    def test_call(self):
        ig, I, x, ocf = self.pars

        with self.assertRaises(TypeError):
            ocf(x)

    def test_L(self):
        ig, I, x, ocf = self.pars
        with self.assertRaises(AttributeError):
            ocf.L

    def test_gradient(self):
        ig, I, x, ocf = self.pars
        with self.assertRaises(AttributeError):
            ocf.gradient(x)

    def test_proximal(self):
        ig, I, x, ocf = self.pars
        with self.assertRaises(NotImplementedError):
            ocf.proximal(x, tau=1)

    def test_proximal_conjugate(self):
        ig, I, x, ocf = self.pars
        with self.assertRaises(NotImplementedError):
            ocf.proximal_conjugate(x, tau=1)

    def test_convex_conjugate(self):
        ig, I, x, ocf = self.pars
        with self.assertRaises(NotImplementedError):
            ocf.convex_conjugate(x)

    def test_proximal_conjugate(self):
        ig, I, x, ocf = self.pars
        with self.assertRaises(NotImplementedError):
            ocf.proximal_conjugate(x, tau=1)


class TestOperatorCompositionFunctionWithWrongInterfaceFunctionAddScalar(
        TestOperatorCompositionFunctionWithWrongInterfaceFunction):

    def setUp(self):
        ig = ImageGeometry(2, 2)
        I = IdentityOperator(ig)

        x = ig.allocate(0)

        class NotAFunction(object):
            pass

        ocf = OperatorCompositionFunction(NotAFunction(), I) + 1
        self.pars = (ig, I, x, ocf)


class TestOperatorCompositionFunctionWithWrongInterfaceFunctionMultiplyScalar(
        TestOperatorCompositionFunctionWithWrongInterfaceFunction):

    def setUp(self):
        ig = ImageGeometry(2, 2)
        I = IdentityOperator(ig)

        x = ig.allocate(0)

        class NotAFunction(object):
            pass

        ocf = OperatorCompositionFunction(NotAFunction(), I) * 2.
        self.pars = (ig, I, x, ocf)


class TestOperatorCompositionFunctionWithWrongInterfaceFunctionAddFunction(
        TestOperatorCompositionFunctionWithWrongInterfaceFunction):

    def setUp(self):
        ig = ImageGeometry(2, 2)
        I = IdentityOperator(ig)

        x = ig.allocate(0)

        class NotAFunction(object):
            pass

        ocf = OperatorCompositionFunction(NotAFunction(), I) + IndicatorBox()
        self.pars = (ig, I, x, ocf)


class TestOperatorCompositionFunctionWithWrongInterfaceOperator(
        TestOperatorCompositionFunctionWithWrongInterfaceFunction):

    def setUp(self):
        ig = ImageGeometry(2, 2)
        F = IndicatorBox()

        x = ig.allocate(0)

        class NotAnOperator(object):
            pass

        nao = NotAnOperator()
        ocf = OperatorCompositionFunction(F, nao)
        self.pars = (ig, nao, x, ocf)

    def tearDown(self):
        pass

    def test_call(self):
        ig, I, x, ocf = self.pars

        with self.assertRaises(AttributeError):
            ocf(x)


class TestOperatorCompositionFunctionWithWrongInterfaceOperatorScaled(
        TestOperatorCompositionFunctionWithWrongInterfaceOperator):

    def setUp(self):
        ig = ImageGeometry(2, 2)
        F = IndicatorBox()

        x = ig.allocate(0)

        class NotAnOperator(object):

            def __mul__(self, value):
                return self

        nao = NotAnOperator() * 2
        ocf = OperatorCompositionFunction(F, nao)
        self.pars = (ig, nao, x, ocf)


class TestBlockFunction(unittest.TestCase):

    def setUp(self):
        # M, N = 50, 50
        # ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
        # b = ig.allocate('random', seed=1)

        # print('Check call with IdentityOperator operator... OK\n')
        # operator = 3 * IdentityOperator(ig)

        # u = ig.allocate('random_int', seed = 50)
        # func2 = LeastSquares(operator, b, 0.5)
        func1 = ConstantFunction(0.3)
        func2 = ConstantFunction(-1.0)
        self.funcs = [func1, func2]
        # self.ig = ig

    def tearDown(self) -> None:
        return super().tearDown()

    def test_iterator(self):
        bf = BlockFunction(*self.funcs)
        for el in bf:
            assert isinstance(el, ConstantFunction)

    def test_rmul_with_scalar_return(self):
        bf = BlockFunction(*self.funcs)
        bf2 = 2 * bf
        assert isinstance(bf2, BlockFunction)

    def test_getitem(self):
        bf = BlockFunction(*self.funcs)
        assert isinstance(bf[0], ConstantFunction)
        assert isinstance(bf[1], ConstantFunction)

    def test_rmul_with_scalar1(self):
        bf0 = BlockFunction(*self.funcs)
        bf = 2 * bf0

        for i in range(2):
            assert bf[i].constant == 2 * bf0[i].constant

    def test_rmul_with_scalar2(self):
        bf0 = BlockFunction(L1Norm())
        bf = 2 * bf0
        assert isinstance(bf[0], ScaledFunction)


class TestIndicatorBox(unittest.TestCase):

    def _test_IndicatorBox(self, accelerated):
        ig = ImageGeometry(10, 10)
        im = ig.allocate(-1)
        ib = IndicatorBox(lower=0, accelerated=accelerated)
        a = ib(im)
        numpy.testing.assert_equal(a, numpy.inf)
        ib = IndicatorBox(lower=-2, accelerated=accelerated)
        a = ib(im)
        numpy.testing.assert_array_equal(0, a)
        ib = IndicatorBox(lower=-5, upper=-2, accelerated=accelerated)
        a = ib(im)
        numpy.testing.assert_equal(a, numpy.inf)

    def test_basic(self):
        self._test_IndicatorBox('numpy')
        self._test_IndicatorBox('numba')

    def create_circular_mask(self, ig):
        mask = ig.allocate(1)
        # create circular mask
        pix_x, pix_y = ig.shape
        center = [int(pix_x / 2.), int(pix_y / 2.)]

        Y, X = np.ogrid[:pix_y, :pix_x]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

        radius = 4
        mask_arr = dist_from_center <= radius

        np.multiply(mask.array, mask_arr, out=mask.array)
        return mask

    def test_pixelwise_call(self):
        self._test_IndicatorBox_pixelwise_call('numpy')
        self._test_IndicatorBox_pixelwise_call('numba')

    def _test_IndicatorBox_pixelwise_call(self, accelerated):
        ig = ImageGeometry(10, 10)
        mask = self.create_circular_mask(ig)

        im = ig.allocate(2)
        ib = IndicatorBox(lower=-2 * mask, accelerated=accelerated)
        for val, res in zip([2, -3], [0, np.inf]):
            print("test1", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        im = ig.allocate(2)
        ib = IndicatorBox(lower=-2 * mask, upper=None, accelerated=accelerated)
        for val, res in zip([2, -3], [0, np.inf]):
            print("test1", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        im = ig.allocate(2)
        ib = IndicatorBox(upper=2 * mask, accelerated=accelerated)
        for val, res in zip([-1, 3], [0, np.inf]):
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)
        ib = IndicatorBox(upper=2 * mask, lower=None, accelerated=accelerated)
        for val, res in zip([-1, 3], [0, np.inf]):
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        ib = IndicatorBox(upper=2 * mask,
                          lower=-2 * mask,
                          accelerated=accelerated)
        for val, res in zip([-1, 1, 3], [np.inf, np.inf, np.inf]):
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        im.fill(1)
        im.array *= mask.as_array()
        np.testing.assert_equal(ib(im), 0)

    def test_pixelwise_call_suppress(self):
        self._test_IndicatorBox_pixelwise_call_suppress('numpy')
        self._test_IndicatorBox_pixelwise_call_suppress('numba')

    def _test_IndicatorBox_pixelwise_call_suppress(self, accelerated):
        ig = ImageGeometry(10, 10)
        mask = self.create_circular_mask(ig)

        im = ig.allocate(2)
        ib = IndicatorBox(lower=-2 * mask, accelerated=accelerated)
        ib.set_suppress_evaluation(True)
        for val, res in zip([2, -3], [0, 0]):
            print("test1", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        im = ig.allocate(2)
        ib = IndicatorBox(upper=2 * mask, accelerated=accelerated)
        ib.set_suppress_evaluation(True)
        for val, res in zip([-1, 3], [0, 0]):
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        ib = IndicatorBox(lower=-2 * mask, upper=None, accelerated=accelerated)
        ib.set_suppress_evaluation(True)
        for val, res in zip([2, -3], [0, 0]):
            print("test1", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        im = ig.allocate(2)
        ib = IndicatorBox(upper=2 * mask, lower=None, accelerated=accelerated)
        ib.set_suppress_evaluation(True)
        for val, res in zip([-1, 3], [0, 0]):
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        ib = IndicatorBox(upper=2 * mask,
                          lower=-2 * mask,
                          accelerated=accelerated)
        ib.set_suppress_evaluation(True)
        for val, res in zip([-1, 1, 3], [0, 0, 0]):
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_equal(ib(im), res)

        im.fill(1)
        im.array *= mask.as_array()
        np.testing.assert_equal(ib(im), 0)

    def test_pixelwise_proximal(self):
        self._test_IndicatorBox_pixelwise_proximal('numpy')
        self._test_IndicatorBox_pixelwise_proximal('numba')

    def _test_IndicatorBox_pixelwise_proximal(self, accelerated):
        ig = ImageGeometry(10, 10)
        mask = self.create_circular_mask(ig)

        im = ig.allocate(2)
        ib = IndicatorBox(lower=-2 * mask, accelerated=accelerated)
        for val, res in zip([2, -3], [ig.allocate(2), -2 * mask]):
            # logging.info("test1", val, res)
            im.fill(val)
            np.testing.assert_allclose(
                ib.proximal(im, 1).as_array(), res.as_array())

        im = ig.allocate(2)
        ib = IndicatorBox(upper=2 * mask, accelerated=accelerated)
        for val, res in zip([-1, 3], [ig.allocate(-1), 2 * mask]):
            # logging.info("test1", val, res)
            print("test2", val, res)
            im.fill(val)
            np.testing.assert_allclose(
                ib.proximal(im, 1).as_array(), res.as_array())

        im = ig.allocate(2)
        ib = IndicatorBox(upper=2 * mask,
                          lower=-2 * mask,
                          accelerated=accelerated)
        for val, res in zip([-1, -3, 1], [-1 * mask, -2 * mask, 1 * mask]):
            print("test3", val, res)
            im.fill(val)
            np.testing.assert_allclose(
                ib.proximal(im, 1).as_array(), res.as_array())
            # check against old implementation of proximal:
            out = im.maximum(ib.lower)
            # numpy clip
            out.minimum(ib.upper, out=out)
            np.testing.assert_allclose(
                ib.proximal(im, 1).as_array(), out.as_array())

    def test_input0(self):
        self._test_IndicatorBox_input0('numpy')
        self._test_IndicatorBox_input0('numba')

    def _test_IndicatorBox_input0(self, accelerated):
        if not accelerated:
            return

        exc = numba.core.errors.TypingError

        self.input_IndicatorBox(lower='string',
                                upper=[1, 1],
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(upper='string',
                                lower=[1, 1],
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(lower=[0, 0],
                                upper=[1, 1],
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(lower=[0, 0],
                                upper=1,
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(lower=[0, 0],
                                upper=None,
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(lower=[0, 0],
                                upper=VectorData(numpy.asarray([0.5, 0.5])),
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(upper=[0, 0],
                                lower=[1, 1],
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(upper=[0, 0],
                                lower=1,
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(upper=[0, 0],
                                lower=None,
                                exception=exc,
                                accelerated=accelerated)
        self.input_IndicatorBox(upper=[0, 0],
                                lower=VectorData(numpy.asarray([0.5, 0.5])),
                                exception=exc,
                                accelerated=accelerated)

    def test_input1(self):
        self._test_IndicatorBox_input1('numpy')
        self._test_IndicatorBox_input1('numba')

    def _test_IndicatorBox_input1(self, accelerated):
        if accelerated:
            exc = ValueError
            self.input_IndicatorBox(upper=VectorData(numpy.asarray([
                0.5,
            ])),
                                    lower=VectorData(numpy.asarray([
                                        0.5,
                                    ])),
                                    exception=exc,
                                    accelerated=accelerated)
            self.input_IndicatorBox(upper=VectorData(numpy.asarray([0.5,
                                                                    0.5])),
                                    lower=VectorData(
                                        numpy.asarray([0.5, 0.5, 0.5])),
                                    exception=exc,
                                    accelerated=accelerated)

        else:
            pass

    def input_IndicatorBox(self, lower, upper, exception, accelerated):
        ib = IndicatorBox(lower=lower, upper=upper, accelerated=accelerated)
        x = VectorData(numpy.asarray([0.5, 0.5]))
        with self.assertRaises(exception):
            ib(x)

    def test_convex_conjugate(self):
        self._test_IndicatorBox_convex_conjugate('numpy')
        self._test_IndicatorBox_convex_conjugate('numba')

    def _test_IndicatorBox_convex_conjugate(self, accelerated):
        ig = ImageGeometry(10, 10)
        mask = self.create_circular_mask(ig)

        im = ig.allocate(-2) * mask
        ib = IndicatorBox(lower=-2 * mask, accelerated=accelerated)

        ib.convex_conjugate(im)
        np.testing.assert_equal(ib.convex_conjugate(im), im.maximum(0).sum())

        im = ig.allocate(2) * mask
        np.testing.assert_equal(ib.convex_conjugate(im), im.maximum(0).sum())

    def test_suppress_evaluation(self):
        self._test_IndicatorBox_suppress_evaluation('numpy')
        self._test_IndicatorBox_suppress_evaluation('numba')

    def _test_IndicatorBox_suppress_evaluation(self, accelerated):
        ig = ImageGeometry(10, 10)
        mask = self.create_circular_mask(ig)

        im = ig.allocate(2)
        ib = IndicatorBox(upper=mask, accelerated=accelerated)

        assert ib.suppress_evaluation == False

        v = ib(im)

        ib.set_suppress_evaluation(True)

        assert ib.suppress_evaluation == True
        v2 = ib(im)
        assert v2 == 0

        assert v != ib(im)

        with self.assertRaises(ValueError):
            ib.set_suppress_evaluation('string')

    def test_set_num_threads(self):
        ig = ImageGeometry(10, 10)
        mask = self.create_circular_mask(ig)

        for acc in [True, False]:
            im = ig.allocate(2)
            ib = IndicatorBox(upper=mask, accelerated=acc)

            assert ib.num_threads == cilmp.NUM_THREADS

            N = 10
            ib.set_num_threads(N)
            assert ib.num_threads == N
