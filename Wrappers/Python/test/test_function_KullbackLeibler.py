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
               
