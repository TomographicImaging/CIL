# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import unittest
import numpy
import numpy as np
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData, VectorData
from ccpi.framework import ImageGeometry,VectorGeometry
from ccpi.framework import AcquisitionGeometry
from ccpi.optimisation.algorithms import FISTA
from ccpi.optimisation.functions import LeastSquares
from ccpi.optimisation.functions import ZeroFunction
from ccpi.optimisation.functions import L1Norm

from ccpi.optimisation.operators import LinearOperatorMatrix
from ccpi.optimisation.operators import Identity
from ccpi.optimisation.operators import LinearOperator

import numpy.testing

try:
    from cvxpy import *
    cvx_not_installable = False
except ImportError:
    cvx_not_installable = True


def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]


def dt(steps):
    return steps[-1] - steps[-2]




class TestAlgorithms(unittest.TestCase):
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_FISTA_cvx(self):
        
        if False:
            if not cvx_not_installable:
                try:
                    # Problem data.
                    m = 30
                    n = 20
                    np.random.seed(1)
                    Amat = np.random.randn(m, n)
                    A = LinearOperatorMatrix(Amat)
                    bmat = np.random.randn(m)
                    bmat.shape = (bmat.shape[0], 1)
    
                    # A = Identity()
                    # Change n to equal to m.
    
                    #b = DataContainer(bmat)
                    vg = VectorGeometry(m)
    
                    b = vg.allocate('random')
    
                    # Regularization parameter
                    lam = 10
                    opt = {'memopt': True}
                    # Create object instances with the test data A and b.
                    f = LeastSquares(A, b, c=0.5)
                    g0 = ZeroFunction()
    
                    # Initial guess
                    #x_init = DataContainer(np.zeros((n, 1)))
                    x_init = vg.allocate()
                    f.gradient(x_init, out = x_init)
    
                    # Run FISTA for least squares plus zero function.
                    #x_fista0, it0, timing0, criter0 = FISTA(x_init, f, g0, opt=opt)
                    fa = FISTA(x_init=x_init, f=f, g=g0)
                    fa.max_iteration = 10
                    fa.run(10)
                    
                    # Print solution and final objective/criterion value for comparison
                    print("FISTA least squares plus zero function solution and objective value:")
                    print(fa.get_output())
                    print(fa.get_last_objective())
    
                    # Compare to CVXPY
    
                    # Construct the problem.
                    x0 = Variable(n)
                    objective0 = Minimize(0.5*sum_squares(Amat*x0 - bmat.T[0]))
                    prob0 = Problem(objective0)
    
                    # The optimal objective is returned by prob.solve().
                    result0 = prob0.solve(verbose=False, solver=SCS, eps=1e-9)
    
                    # The optimal solution for x is stored in x.value and optimal objective value
                    # is in result as well as in objective.value
                    print("CVXPY least squares plus zero function solution and objective value:")
                    print(x0.value)
                    print(objective0.value)
                    self.assertNumpyArrayAlmostEqual(
                        numpy.squeeze(x_fista0.array), x0.value, 6)
                except SolverError as se:
                    print (str(se))
                    self.assertTrue(True)
            else:
                self.assertTrue(cvx_not_installable)

    def stest_FISTA_Norm1_cvx(self):
        if not cvx_not_installable:
            try:
                opt = {'memopt': True}
                # Problem data.
                m = 30
                n = 20
                np.random.seed(1)
                Amat = np.random.randn(m, n)
                A = LinearOperatorMatrix(Amat)
                bmat = np.random.randn(m)
                #bmat.shape = (bmat.shape[0], 1)

                # A = Identity()
                # Change n to equal to m.
                vgb = VectorGeometry(m)
                vgx = VectorGeometry(n)
                b = vgb.allocate()
                b.fill(bmat)
                #b = DataContainer(bmat)

                # Regularization parameter
                lam = 10
                opt = {'memopt': True}
                # Create object instances with the test data A and b.
                f = LeastSquares(A, b, c=0.5)
                g0 = ZeroFunction()

                # Initial guess
                #x_init = DataContainer(np.zeros((n, 1)))
                x_init = vgx.allocate()

                # Create 1-norm object instance
                g1 = lam * L1Norm()

                g1(x_init)
                g1.prox(x_init, 0.02)

                # Combine with least squares and solve using generic FISTA implementation
                #x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1, opt=opt)
                fa = FISTA(x_init=x_init, f=f, g=g1)
                fa.max_iteration = 10
                fa.run(10)
                

                # Print for comparison
                print("FISTA least squares plus 1-norm solution and objective value:")
                print(fa.get_output())
                print(fa.get_last_objective())

                # Compare to CVXPY

                # Construct the problem.
                x1 = Variable(n)
                objective1 = Minimize(
                    0.5*sum_squares(Amat*x1 - bmat.T[0]) + lam*norm(x1, 1))
                prob1 = Problem(objective1)

                # The optimal objective is returned by prob.solve().
                result1 = prob1.solve(verbose=False, solver=SCS, eps=1e-9)

                # The optimal solution for x is stored in x.value and optimal objective value
                # is in result as well as in objective.value
                print("CVXPY least squares plus 1-norm solution and objective value:")
                print(x1.value)
                print(objective1.value)

                self.assertNumpyArrayAlmostEqual(
                    numpy.squeeze(x_fista1.array), x1.value, 6)
            except SolverError as se:
                print (str(se))
                self.assertTrue(True)
        else:
            self.assertTrue(cvx_not_installable)

    def skip_test_FBPD_Norm1_cvx(self):
        print ("test_FBPD_Norm1_cvx")
        if not cvx_not_installable:
            opt = {'memopt': True}
            # Problem data.
            m = 30
            n = 20
            np.random.seed(1)
            Amat = np.random.randn(m, n)
            A = LinearOperatorMatrix(Amat)
            bmat = np.random.randn(m)
            bmat.shape = (bmat.shape[0], 1)

            # A = Identity()
            # Change n to equal to m.

            b = DataContainer(bmat)

            # Regularization parameter
            lam = 10
            opt = {'memopt': True}
            # Initial guess
            x_init = DataContainer(np.random.randn(n, 1))

            # Create object instances with the test data A and b.
            f = LeastSquares(A, b, c=0.5, memopt=True)
            f.L = LinearOperator.PowerMethod(A, 25, x_init)[0]
            print ("Lipschitz", f.L)
            g0 = ZeroFun()


            # Create 1-norm object instance
            g1 = Norm1(lam)

            # Compare to CVXPY

            # Construct the problem.
            x1 = Variable(n)
            objective1 = Minimize(
                0.5*sum_squares(Amat*x1 - bmat.T[0]) + lam*norm(x1, 1))
            prob1 = Problem(objective1)

            # The optimal objective is returned by prob.solve().
            result1 = prob1.solve(verbose=False, solver=SCS, eps=1e-9)

            # The optimal solution for x is stored in x.value and optimal objective value
            # is in result as well as in objective.value
            print("CVXPY least squares plus 1-norm solution and objective value:")
            print(x1.value)
            print(objective1.value)

            # Now try another algorithm FBPD for same problem:
            x_fbpd1, itfbpd1, timingfbpd1, criterfbpd1 = FBPD(x_init,
                                                              Identity(), None, f, g1)
            print(x_fbpd1)
            print(criterfbpd1[-1])

            self.assertNumpyArrayAlmostEqual(
                numpy.squeeze(x_fbpd1.array), x1.value, 6)
        else:
            self.assertTrue(cvx_not_installable)
        # Plot criterion curve to see both FISTA and FBPD converge to same value.
        # Note that FISTA is very efficient for 1-norm minimization so it beats
        # FBPD in this test by a lot. But FBPD can handle a larger class of problems
        # than FISTA can.

        # Now try 1-norm and TV denoising with FBPD, first 1-norm.

        # Set up phantom size NxN by creating ImageGeometry, initialising the
        # ImageData object with this geometry and empty array and finally put some
        # data into its array, and display as image.
    def skip_test_FISTA_denoise_cvx(self):
        if not cvx_not_installable:
            opt = {'memopt': True}
            N = 64
            ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
            Phantom = ImageData(geometry=ig)

            x = Phantom.as_array()

            x[int(round(N/4)):int(round(3*N/4)),
              int(round(N/4)):int(round(3*N/4))] = 0.5
            x[int(round(N/8)):int(round(7*N/8)),
              int(round(3*N/8)):int(round(5*N/8))] = 1

            # Identity operator for denoising
            I = TomoIdentity(ig)

            # Data and add noise
            y = I.direct(Phantom)
            y.array = y.array + 0.1*np.random.randn(N, N)

            # Data fidelity term
            f_denoise = LeastSquares(I, y, c=0.5, memopt=True)
            x_init = ImageData(geometry=ig)
            f_denoise.L = LinearOperator.PowerMethod(I, 25, x_init)[0]

            # 1-norm regulariser
            lam1_denoise = 1.0
            g1_denoise = Norm1(lam1_denoise)

            # Initial guess
            x_init_denoise = ImageData(np.zeros((N, N)))

            # Combine with least squares and solve using generic FISTA implementation
            x_fista1_denoise, it1_denoise, timing1_denoise, \
                criter1_denoise = \
                FISTA(x_init_denoise, f_denoise, g1_denoise, opt=opt)

            print(x_fista1_denoise)
            print(criter1_denoise[-1])

            # Now denoise LS + 1-norm with FBPD
            x_fbpd1_denoise, itfbpd1_denoise, timingfbpd1_denoise,\
                criterfbpd1_denoise = \
                FBPD(x_init_denoise, I, None, f_denoise, g1_denoise)
            print(x_fbpd1_denoise)
            print(criterfbpd1_denoise[-1])

            # Compare to CVXPY

            # Construct the problem.
            x1_denoise = Variable(N**2)
            objective1_denoise = Minimize(
                0.5*sum_squares(x1_denoise - y.array.flatten()) + lam1_denoise*norm(x1_denoise, 1))
            prob1_denoise = Problem(objective1_denoise)

            # The optimal objective is returned by prob.solve().
            result1_denoise = prob1_denoise.solve(
                verbose=False, solver=SCS, eps=1e-12)

            # The optimal solution for x is stored in x.value and optimal objective value
            # is in result as well as in objective.value
            print("CVXPY least squares plus 1-norm solution and objective value:")
            print(x1_denoise.value)
            print(objective1_denoise.value)
            self.assertNumpyArrayAlmostEqual(
                x_fista1_denoise.array.flatten(), x1_denoise.value, 5)

            self.assertNumpyArrayAlmostEqual(
                x_fbpd1_denoise.array.flatten(), x1_denoise.value, 5)
        else:
            self.assertTrue(cvx_not_installable)


class TestFunction(unittest.TestCase):
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def create_simple_ImageData(self):
        N = 64
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        Phantom = ImageData(geometry=ig)

        x = Phantom.as_array()

        x[int(round(N/4)):int(round(3*N/4)),
          int(round(N/4)):int(round(3*N/4))] = 0.5
        x[int(round(N/8)):int(round(7*N/8)),
          int(round(3*N/8)):int(round(5*N/8))] = 1

        return (ig, Phantom)

    def _test_Norm2(self):
        print("test Norm2")
        opt = {'memopt': True}
        ig, Phantom = self.create_simple_ImageData()
        x = Phantom.as_array()
        print(Phantom)
        print(Phantom.as_array())

        norm2 = Norm2()
        v1 = norm2(x)
        v2 = norm2(Phantom)
        self.assertEqual(v1, v2)

        p1 = norm2.prox(Phantom, 1)
        print(p1)
        p2 = norm2.prox(x, 1)
        self.assertNumpyArrayEqual(p1.as_array(), p2)

        p3 = norm2.proximal(Phantom, 1)
        p4 = norm2.proximal(x, 1)
        self.assertNumpyArrayEqual(p3.as_array(), p2)
        self.assertNumpyArrayEqual(p3.as_array(), p4)

        out = Phantom.copy()
        p5 = norm2.proximal(Phantom, 1, out=out)
        self.assertEqual(id(p5), id(out))
        self.assertNumpyArrayEqual(p5.as_array(), p3.as_array())
# =============================================================================
#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')
#
#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())
#
#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)
# =============================================================================



if __name__ == '__main__':
    unittest.main()
    
