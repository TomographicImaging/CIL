#  Copyright 2022 United Kingdom Research and Innovation
#  Copyright 2022 The University of Manchester
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
import unittest
from cil.optimisation.functions import KullbackLeibler
from cil.framework import ImageGeometry
import numpy
import scipy
from utils import has_numba, initialise_tests

initialise_tests()

if has_numba:
    import numba

class TestKullbackLeiblerNumpy(unittest.TestCase):

    def setUp(self):

        M, N, K =  2, 3, 4
        self.ig = ImageGeometry(N, M, K)

        self.u1 = self.ig.allocate('random', seed = 500)
        self.g1 = self.ig.allocate('random', seed = 100)
        self.b1 = self.ig.allocate('random', seed = 1000)

        self.f = KullbackLeibler(b = self.g1, backend='numpy')
        self.f1 = KullbackLeibler(b = self.g1, eta = self.b1,  backend='numpy')
        self.tau = 400.4

    def test_signature(self):

        # with no data
        with self.assertRaises(TypeError):
            f = KullbackLeibler()

        with self.assertRaises(ValueError):
            f = KullbackLeibler(b=-1*self.g1)

    def test_call_method(self):

        # without eta
        numpy.testing.assert_allclose(0.0, self.f(self.g1))

        # with eta
        tmp_sum = (self.u1 + self.f1.eta).as_array()
        ind = tmp_sum >= 0
        tmp = scipy.special.kl_div(self.f1.b.as_array()[ind], tmp_sum[ind])
        numpy.testing.assert_allclose(self.f1(self.u1), numpy.sum(tmp) )

    def test_gradient_method(self):

        # without eta
        res1 = self.f.gradient(self.u1)
        res2 = self.u1.geometry.allocate()
        self.f.gradient(self.u1, out = res2)
        numpy.testing.assert_allclose(res1.as_array(), res2.as_array())

        # with eta
        res1 = self.f1.gradient(self.u1)
        res2 = self.u1.geometry.allocate()
        self.f1.gradient(self.u1, out = res2)
        numpy.testing.assert_allclose(res1.as_array(), res2.as_array())

    def test_proximal_method(self):

        # without eta
        res1 = self.f.proximal(self.u1, self.tau)
        res2 = self.u1.geometry.allocate()
        self.f.proximal(self.u1, self.tau, out = res2)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=4)

        # with eta
        res1 = self.f1.proximal(self.u1, self.tau)
        res2 = self.u1.geometry.allocate()
        self.f1.proximal(self.u1, self.tau, out = res2)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=4)

    def test_proximal_conjugate_method(self):

        # without eta
        res1 = self.f.proximal_conjugate(self.u1, self.tau)
        res2 = self.u1.geometry.allocate()
        self.f.proximal_conjugate(self.u1, self.tau, out = res2)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=4)

        # with eta
        res1 = self.f1.proximal_conjugate(self.u1, self.tau)
        res2 = self.u1.geometry.allocate()
        self.f1.proximal_conjugate(self.u1, self.tau, out = res2)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=4)

    def test_convex_conjugate_method(self):

        # with eta
        tmp = 1 - self.u1.as_array()
        ind = tmp>0
        xlogy = - scipy.special.xlogy(self.f1.b.as_array()[ind], tmp[ind])
        res1 = numpy.sum(xlogy) - self.f1.eta.dot(self.u1)
        res2 = self.f1.convex_conjugate(self.u1)
        numpy.testing.assert_equal(res1, res2)

        # without eta
        tmp = 1 - self.u1.as_array()
        ind = tmp>0
        xlogy = - scipy.special.xlogy(self.f.b.as_array()[ind], tmp[ind])
        res1 = numpy.sum(xlogy) - self.f.eta.dot(self.u1)
        res2 = self.f.convex_conjugate(self.u1)
        numpy.testing.assert_equal(res1, res2)

@unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
class TestKullbackLeiblerNumba(unittest.TestCase):
    def setUp(self):
        #numpy.random.seed(1)
        M, N, K =  2, 3, 4
        ig = ImageGeometry(N, M)

        u1 = ig.allocate('random', seed = 500)
        u1 = ig.allocate(0.2)
        #g1 = ig.allocate('random', seed = 100)
        g1 = ig.allocate(1)

        b1 = ig.allocate('random', seed = 1000)
        eta = ig.allocate(1e-3)

        mask = ig.allocate(1)

        mask.fill(0, horizontal_x=0)

        mask_c = ig.allocate(0)
        mask_c.fill(1, horizontal_x=0)

        f = KullbackLeibler(b=g1, backend="numba", eta=eta)
        f_np = KullbackLeibler(b=g1, backend="numpy", eta=eta)

        # mask is on vartical=0
        # separate the u1 vertical=0
        f_mask = KullbackLeibler(b=g1.copy(), backend="numba", mask=mask.copy(), eta=eta.copy())
        f_mask_c = KullbackLeibler(b=g1.copy(), backend="numba", mask=mask_c.copy(), eta=eta.copy())
        f_on_mask = KullbackLeibler(b=g1.get_slice(horizontal_x=0), backend="numba", eta=eta.get_slice(horizontal_x=0))
        u1_on_mask = u1.get_slice(horizontal_x=0)

        tau = 400.4
        self.tau = tau
        self.u1 = u1
        self.g1 = g1
        self.b1 = b1
        self.eta = eta
        self.f = f
        self.f_np = f_np
        self.mask = mask
        self.mask_c = mask_c
        self.f_mask = f_mask
        self.f_mask_c = f_mask_c
        self.f_on_mask = f_on_mask
        self.u1_on_mask = u1_on_mask

    def test_KullbackLeibler_numba_call(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f(u1), f_np(u1),  rtol=1e-5)

    def test_KullbackLeibler_numba_call_mask(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1
        g1 = self.g1
        mask = self.mask

        u1_on_mask = self.u1_on_mask
        f_on_mask = self.f_on_mask
        f_mask = self.f_mask
        f_mask_c = self.f_mask_c

        numpy.testing.assert_allclose(f_mask(u1) + f_mask_c(u1), f(u1),  rtol=1e-5)

    def test_KullbackLeibler_numba_proximal(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.proximal(u1,tau=tau).as_array(),
                                      f_np.proximal(u1,tau=tau).as_array(), rtol=7e-3)
        numpy.testing.assert_array_almost_equal(f.proximal(u1,tau=tau).as_array(),
        f_np.proximal(u1,tau=tau).as_array(), decimal=4)

    def test_KullbackLeibler_numba_proximal_arr(self):
        f = self.f
        f_np = self.f_np
        tau = self.u1.copy()
        tau.fill(self.tau)
        u1 = self.u1
        a = f.proximal(u1,tau=self.tau)
        b = f.proximal(u1,tau=tau)
        numpy.testing.assert_allclose(f.proximal(u1,tau=self.tau).as_array(),
                                      f.proximal(u1,tau=tau).as_array(), rtol=7e-3)
        numpy.testing.assert_array_almost_equal(f.proximal(u1,tau=self.tau).as_array(),
                                                f.proximal(u1,tau=tau).as_array(), decimal=4)

    def test_KullbackLeibler_numba_gradient(self):
        with self.subTest():
            f = self.f
            f_np = self.f_np
            u1 = self.u1
            grad_np = f_np.gradient(u1).as_array()
            numpy.testing.assert_allclose(f.gradient(u1).as_array(), grad_np, rtol=1e-3)
        with self.subTest("mask"):
            f = self.f_mask
            mask = self.mask>0
            grad_mask = f.gradient(u1).as_array()
            numpy.testing.assert_allclose(grad_mask[mask], grad_np[mask], rtol=1e-3)
            self.assertFalse(grad_mask[~mask].any())

    def test_KullbackLeibler_numba_convex_conjugate(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.convex_conjugate(u1), f_np.convex_conjugate(u1), rtol=1e-3)

    def test_KullbackLeibler_numba_proximal_conjugate_arr(self):
        f = self.f
        f_np = self.f_np
        tau = self.tau
        u1 = self.u1

        numpy.testing.assert_allclose(f.proximal_conjugate(u1,tau=tau).as_array(),
                        f_np.proximal_conjugate(u1,tau=tau).as_array(), rtol=1e-3)

    def test_KullbackLeibler_numba_convex_conjugate_mask(self):
        f = self.f
        tau = self.tau
        u1 = self.u1

        mask = self.mask
        f_mask = self.f_mask
        f_mask_c = self.f_mask_c
        f_on_mask = self.f_on_mask
        u1_on_mask = self.u1_on_mask

        numpy.testing.assert_allclose(
            f.convex_conjugate(u1),
            f_mask.convex_conjugate(u1) + f_mask_c.convex_conjugate(u1) ,\
                 rtol=1e-3)

    def test_KullbackLeibler_numba_proximal_conjugate_mask(self):
        f = self.f
        f_mask = self.f_mask
        f_mask_c = self.f_mask_c
        x = self.u1
        m = self.mask
        m_c = self.mask_c
        tau = self.tau

        out = x * 0
        out_c = x * 0
        f_mask_c.proximal_conjugate(x,tau=tau, out=out_c)
        f_mask.proximal_conjugate(x,tau=tau, out=out)
        numpy.testing.assert_allclose(f.proximal_conjugate(x,tau=tau).as_array(),
                                      (out + out_c).as_array(), rtol=7e-3)
        # print ("f.prox_conj\n"       , f.proximal_conjugate(x,tau=tau).as_array())
        # print ("f_mask.prox_conj\n"  , out.as_array())
        # print ("f_mask_c.prox_conj\n", out_c.as_array())
        b = f_mask_c.proximal_conjugate(x,tau=tau)
        a = f_mask.proximal_conjugate(x,tau=tau)
        numpy.testing.assert_allclose(f.proximal_conjugate(x,tau=tau).as_array(),
                                      (f_mask.proximal_conjugate(x,tau=tau) +\
                                      f_mask_c.proximal_conjugate(x, tau=tau)) .as_array(), rtol=7e-3)
