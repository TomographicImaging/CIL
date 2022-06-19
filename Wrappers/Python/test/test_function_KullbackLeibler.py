# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from cil.optimisation.functions import KullbackLeibler
from cil.framework import ImageGeometry
import numpy
import scipy


class Test_KL_Function(unittest.TestCase):
    
    def setUp(self):
        
        M, N, K =  2, 3, 4
        ig = ImageGeometry(N, M, K)
        
        self.u1 = ig.allocate('random', seed = 500)    
        self.g1 = ig.allocate('random', seed = 100)
        self.b1 = ig.allocate('random', seed = 1000)

        self.f = KullbackLeibler(b = self.g1, backend='numpy')  
        self.f1 = KullbackLeibler(b = self.g1, eta = self.b1,  backend='numpy') 
        self.tau = 400.4

    def test_signature(self):

        # with no data
        with self.assertRaises(ValueError):
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

        res1 = self.f.gradient(self.u1)
        res2 = self.u1.geometry.allocate()
        self.f.gradient(self.u1, out = res2) 
        numpy.testing.assert_allclose(res1.as_array(), res2.as_array())        

    def test_proximal_method(self):

        res1 = self.f.proximal(self.u1, self.tau)
        res2 = self.u1.geometry.allocate()   
        self.f.proximal(self.u1, self.tau, out = res2)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=4)  

    def test_proximal_conjugate_method(self):

        res1 = self.f.proximal_conjugate(self.u1, self.tau)
        res2 = self.u1.geometry.allocate()   
        self.f.proximal_conjugate(self.u1, self.tau, out = res2)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=4)  

    def test_convex_conjugate_method(self):

        pass

    def test_call_method_numba(self):
        pass


        
#         u2 = u1 * 0 + 2.
#         self.assertNumpyArrayAlmostEqual(0.0, f.convex_conjugate(u2))   
#         eta = b1
        
      
        
#         res_proximal_conj_out = u1.geometry.allocate()
#         proxc = f.proximal_conjugate(u1,tau)
#         f.proximal_conjugate(u1, tau, out=res_proximal_conj_out)
#         numpy.testing.assert_array_almost_equal(proxc.as_array(), res_proximal_conj_out.as_array())
