#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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
from utils import initialise_tests
from cil.optimisation.functions import L2NormSquared, MixedL21Norm, TotalVariation, ZeroFunction, IndicatorBox
from cil.optimisation.operators import GradientOperator
from cil.optimisation.algorithms import PDHG, PD3O, FISTA
from cil.utilities import dataexample
import numpy as np
import scipy.sparse as sp

initialise_tests()



class Test_PD3O(unittest.TestCase):

    def setUp(self):

        # default test data
        self.data = dataexample.CAMERA.get(size=(32, 32))
        
   
    def test_init_and_set_up(self):
        F1= 0.5 * L2NormSquared(b=self.data)
        H1 = 0.1*  MixedL21Norm()
        operator =  GradientOperator(self.data.geometry)
        G1= IndicatorBox(lower=0)

        algo_pd3o=PD3O(f=F1, g=G1, h=H1, operator=operator)
        self.assertEqual(algo_pd3o.gamma,0.99*2.0/F1.L )
        self.assertEqual(algo_pd3o.delta,F1.L/(0.99*2.0*operator.norm()**2) )
        np.testing.assert_array_equal(algo_pd3o.x.array, self.data.geometry.allocate(0).array)
        self.assertTrue(algo_pd3o.configured)
        
        algo_pd3o=PD3O(f=F1, g=G1, h=H1, operator=operator, gamma=3, delta=43.1, initial=self.data.geometry.allocate('random', seed=3))
        self.assertEqual(algo_pd3o.gamma,3 )
        self.assertEqual(algo_pd3o.delta,43.1 )
        np.testing.assert_array_equal(algo_pd3o.x.array, self.data.geometry.allocate('random', seed=3).array)
        self.assertTrue(algo_pd3o.configured)
        
        with self.assertWarnsRegex(UserWarning,"Please use PDHG instead." ):
            algo_pd3o=PD3O(f=ZeroFunction(), g=G1, h=H1, operator=operator, gamma=3, delta=43.1)
        
       
    def test_pd3o_vs_fista(self):
        alpha = 0.1  
        G = alpha * TotalVariation(max_iteration=5, lower=0)

        F= 0.5 * L2NormSquared(b=self.data)
        algo=FISTA(f=F, g=G,initial=0*self.data)
        algo.run(200)

        F1= 0.5 * L2NormSquared(b=self.data)
        H1 = alpha *  MixedL21Norm()
        operator =  GradientOperator(self.data.geometry)
        G1= IndicatorBox(lower=0)

        algo_pd3o=PD3O(f=F1, g=G1, h=H1, operator=operator, initial=0*self.data)
        algo_pd3o.run(500)

        np.testing.assert_allclose(algo.solution.as_array(), algo_pd3o.solution.as_array(), atol=1e-2)
        np.testing.assert_allclose(algo.objective[-1], algo_pd3o.objective[-1], atol=1e-2)
        
    def test_PD3O_PDHG_denoising(self):

        # compare the TV denoising problem using
        # FISTA via proximal TV
        # PDHG, PD3O

        # regularisation parameter
        alpha = 0.1        

        # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
        TV = TotalVariation(max_iteration=200)
        tv_cil = TV.proximal(self.data, tau=alpha)  

        # setup PDHG denoising      
        F = alpha * MixedL21Norm()
        operator = GradientOperator(self.data.geometry)
        G = 0.5 * L2NormSquared(b=self.data)
        pdhg = PDHG(f=F, g=G, operator=operator, update_objective_interval = 100, 
                    max_iteration = 2000)
        pdhg.run(verbose=1)

        # setup PD3O denoising  (F=ZeroFunction)   
        H = alpha * MixedL21Norm()
        norm_op = operator.norm()
        F = ZeroFunction()
        gamma = 1./norm_op
        delta = 1./norm_op

        pd3O = PD3O(f=F, g=G, h=H, operator=operator, gamma=gamma, delta=delta,
                    update_objective_interval = 100, 
                    max_iteration = 2000)
        pd3O.run(verbose=1)      

        # setup PD3O denoising  (H proximalble and G,F = 1/4 * L2NormSquared)   
        H = alpha * MixedL21Norm()
        G = 0.25 * L2NormSquared(b=self.data)
        F = 0.25 * L2NormSquared(b=self.data)
        gamma = 0.99*2./F.L
        delta = 1./(gamma*norm_op**2)

        pd3O_with_f = PD3O(f=F, g=G, h=H, operator=operator, gamma=gamma, delta=delta,
                    update_objective_interval = 100, 
                    max_iteration = 2000)
        pd3O_with_f.run(verbose=1)        

        # pd30 vs fista
        np.testing.assert_allclose(tv_cil.array, pd3O.solution.array,atol=1e-2) 

        # pd30 vs pdhg
        np.testing.assert_allclose(pdhg.solution.array, pd3O.solution.array,atol=1e-2) 

        # pd30_with_f vs pdhg
        np.testing.assert_allclose(pdhg.solution.array, pd3O_with_f.solution.array,atol=1e-2)               

        # objective values
        np.testing.assert_allclose(pdhg.objective[-1], pd3O_with_f.objective[-1],atol=1e-2) 
        np.testing.assert_allclose(pdhg.objective[-1], pd3O.objective[-1],atol=1e-2)         


            

