# -*- coding: utf-8 -*-
#  Copyright 2023 United Kingdom Research and Innovation
#  Copyright 2023 The University of Manchester
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

import numpy as np

from cil.framework import VectorData



from cil.utilities import dataexample
from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import ApproximateGradientSumFunction
from cil.optimisation.functions import SGFunction
from cil.optimisation.functions import SVRGFunction, LSVRGFunction
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.algorithms import GD 
from cil.framework import VectorData
from cil.optimisation.utilities import Sampler, SamplerRandom

from testclass import CCPiTestClass
from utils import  has_astra

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator

class TestApproximateGradientSumFunction(CCPiTestClass):

    def setUp(self):
        self.sampler=Sampler.random_with_replacement(5)
        self.initial = VectorData(np.zeros(10))
        self.b =  VectorData(np.random.normal(0,1,10))
        self.functions=[]
        for i in range(5):
            diagonal=np.zeros(10)
            diagonal[2*i:2*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            self.functions.append( LeastSquares(A, A.direct(self.b)))
            if i==0:
               self.objective=LeastSquares(A, A.direct(self.b))
            else:
               self.objective+=LeastSquares(A, A.direct(self.b))
        
    def test_ABC(self):
        with self.assertRaises(TypeError):
            self.stochastic_objective=ApproximateGradientSumFunction(self.functions, self.sampler)

        



class TestSGD(CCPiTestClass):

    def setUp(self):
        self.sampler=Sampler.random_with_replacement(5)
        self.data=dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data.reorder('astra')
        self.data2d=self.data.get_slice(vertical='centre')
        ag2D = self.data2d.geometry
        ag2D.set_angles(ag2D.angles, initial_angle=0.2, angle_unit='radian')
        ig2D = ag2D.get_ImageGeometry()
        self.A = ProjectionOperator(ig2D, ag2D, device = "cpu")
        self.n_subsets = 5
        self.partitioned_data=self.data2d.partition(self.n_subsets, 'sequential')
        self.A_partitioned = ProjectionOperator(ig2D, self.partitioned_data.geometry, device = "cpu")
        self.f_subsets = []
        for i in range(self.n_subsets):
            fi=LeastSquares(self.A_partitioned.operators[i],self. partitioned_data[i])
            self.f_subsets.append(fi)
        self.f=LeastSquares(self.A, self.data2d)
        self.f_stochastic=SGFunction(self.f_subsets,self.sampler)
        self.initial=ig2D.allocate()

    def test_approximate_gradient(self): #Test when we the approximate gradient is not equal to the full gradient 
        self.assertFalse((self.f_stochastic.full_gradient(self.initial)==self.f_stochastic.gradient(self.initial).array).all())

    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f=SGFunction(self.f_subsets)
        self.assertTrue(isinstance( f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    def test_direct(self):
        self.assertAlmostEqual(self.f_stochastic(self.initial), self.f(self.initial),1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(self.initial).array, self.f.gradient(self.initial).array,2)
    
    def test_value_error_with_only_one_function(self):
        with self.assertRaises(ValueError):
            SGFunction([self.f], self.sampler)
            pass
    def test_type_error_if_functions_not_a_list(self):
        with self.assertRaises(TypeError):
            SGFunction(self.f, self.sampler)

        
    
    def test_sampler_without_next(self):
        class bad_Sampler():
            def init(self):
                pass
        bad_sampler=bad_Sampler()
        with self.assertRaises(ValueError):
           SGFunction([self.f, self.f], bad_sampler)
  

    def test_SGD_simulated_parallel_beam_data(self): 

        rate = self.f.L
        alg = GD(initial=self.initial, 
                              objective_function=self.f, update_objective_interval=500,
                              rate=rate, alpha=1e8)
        alg.max_iteration = 200
        alg.run(verbose=0)
       
        
        objective=self.f_stochastic
        alg_stochastic = GD(initial=self.initial, 
                              objective_function=objective, update_objective_interval=500, 
                              step_size=1e-7, max_iteration =5000)
        alg_stochastic.run( self.n_subsets*50, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        
  
    def test_SGD_toy_example(self): 
        sampler=Sampler.random_with_replacement(5)
        initial = VectorData(np.zeros(25))
        b =  VectorData(np.random.normal(0,1,25))
        functions=[]
        for i in range(5):
            diagonal=np.zeros(25)
            diagonal[5*i:5*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))
            if i==0:
               objective=LeastSquares(A, A.direct(b))
            else:
               objective+=LeastSquares(A, A.direct(b))

        rate = objective.L / 3.
    
        alg = GD(initial=initial, 
                              objective_function=objective, update_objective_interval=1000,
                              rate=rate, atol=1e-9, rtol=1e-6)
        alg.max_iteration = 600
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        
        stochastic_objective=SGFunction(functions, sampler)
        self.assertAlmostEqual(stochastic_objective(initial), objective(initial))   
        self.assertNumpyArrayAlmostEqual(stochastic_objective.full_gradient(initial).array, objective.gradient(initial).array)
        

        
        alg_stochastic = GD(initial=initial, 
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.01, max_iteration =5000)
        alg_stochastic.run( 600, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), b.as_array(),3)



        
class TestSVRG(CCPiTestClass):
 #TODO: test update frequency, store_gradients, set_up, doing stochastic things etc 
    def setUp(self):
        self.sampler=Sampler.random_with_replacement(5, seed=1)
        self.data=dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data.reorder('astra')
        self.data2d=self.data.get_slice(vertical='centre')
        ag2D = self.data2d.geometry
        ig2D = ag2D.get_ImageGeometry()
        self.A = ProjectionOperator(ig2D, ag2D, device = "cpu")
        self.n_subsets = 5
        self.partitioned_data=self.data2d.partition(self.n_subsets, 'sequential')
        self.A_partitioned = ProjectionOperator(ig2D, self.partitioned_data.geometry, device = "cpu")
        self.f_subsets = []
        for i in range(self.n_subsets):
            fi=LeastSquares(self.A_partitioned.operators[i],self. partitioned_data[i])
            self.f_subsets.append(fi)
        self.f=LeastSquares(self.A, self.data2d)
        self.f_stochastic=SVRGFunction(self.f_subsets,self.sampler)
        self.initial=ig2D.allocate()


    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f=SVRGFunction(self.f_subsets)
        self.assertTrue(isinstance( f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    def test_direct(self):
        self.assertAlmostEqual(self.f_stochastic(self.initial), self.f(self.initial),1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(self.initial).array, self.f.gradient(self.initial).array,2)

    def test_value_error_with_only_one_function(self):
        with self.assertRaises(ValueError):
            SVRGFunction([self.f], self.sampler)
            pass
    def test_type_error_if_functions_not_a_list(self):
        with self.assertRaises(TypeError):
            SVRGFunction(self.f, self.sampler)

    def test_sampler_without_next(self):
        class bad_Sampler():
            def init(self):
                pass
        bad_sampler=bad_Sampler()
        with self.assertRaises(ValueError):
           SVRGFunction([self.f, self.f], bad_sampler)

    def test_SVRG_init(self):
        pass
    
    def test_SVRG_data_passes(self):
        pass
    
    def test_SVRG_update_frequency(self):
        pass
    
    def test_SVRG_store_gradients(self):
        pass
        
    

    def test_SVRG_simulated_parallel_beam_data(self): 

        rate = self.f.L
        alg = GD(initial=self.initial, 
                              objective_function=self.f, update_objective_interval=500,
                              rate=rate, alpha=1e8)
        alg.max_iteration = 200
        alg.run(verbose=0)


        objective=self.f_stochastic
        alg_stochastic = GD(initial=self.initial, 
                              objective_function=objective, update_objective_interval=500, 
                              step_size=5e-8, max_iteration =5000)
        alg_stochastic.run( self.n_subsets*25, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)



    def test_SVRG_toy_example(self): 
        sampler=Sampler.random_with_replacement(3, seed=1)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b =  VectorData(np.random.normal(0,3,15))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(15)
            diagonal[5*i:5*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))
            if i==0:
               objective=LeastSquares(A, A.direct(b))
            else:
               objective+=LeastSquares(A, A.direct(b))

        rate = objective.L / 3.

        alg = GD(initial=initial, 
                              objective_function=objective, update_objective_interval=1000,
                              rate=rate, atol=1e-9, rtol=1e-6)
        alg.max_iteration = 80
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        stochastic_objective=SVRGFunction(functions, sampler)
        self.assertAlmostEqual(stochastic_objective(initial), objective(initial))   
        self.assertNumpyArrayAlmostEqual(stochastic_objective.full_gradient(initial).array, objective.gradient(initial).array)
       

        alg_stochastic = GD(initial=initial, 
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.05, max_iteration =5000)
        
        alg_stochastic.run(14, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(stochastic_objective.data_passes), np.array([1.]+[4./3, 5./3, 6./3, 7./3, 8./3,11./3,12./3, 13./3, 14./3, 15./3., 16./3, 19./3, 20./3]))
        alg_stochastic.run(100, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), b.as_array(),3)

    def test_SVRG_toy_example_store_gradients(self): 
        sampler=Sampler.random_with_replacement(3, seed=1)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b =  VectorData(np.random.normal(0,3,15))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(15)
            diagonal[5*i:5*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))
            if i==0:
               objective=LeastSquares(A, A.direct(b))
            else:
               objective+=LeastSquares(A, A.direct(b))

        rate = objective.L / 3.

        alg = GD(initial=initial, 
                              objective_function=objective, update_objective_interval=1000,
                              rate=rate, atol=1e-9, rtol=1e-6)
        
        alg.max_iteration = 80
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        stochastic_objective=SVRGFunction(functions, sampler, store_gradients=True)
        self.assertAlmostEqual(stochastic_objective(initial), objective(initial))   
        self.assertNumpyArrayAlmostEqual(stochastic_objective.full_gradient(initial).array, objective.gradient(initial).array)



        alg_stochastic = GD(initial=initial, 
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.05, max_iteration =5000)
        alg_stochastic.run(10, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(stochastic_objective.data_passes), np.array([1.]+[4./3, 5./3, 6./3, 7./3,  8./3, 11./3, 12./3, 13./3, 14./3]))
      
        alg_stochastic.run( 100, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), b.as_array(),3)



class TestLSVRG(CCPiTestClass):
 #TODO: test update frequency, store_gradients, set_up, doing stochastic things etc 
    def setUp(self):
        self.sampler=Sampler.random_with_replacement(5, seed=1)
        self.data=dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data.reorder('astra')
        self.data2d=self.data.get_slice(vertical='centre')
        ag2D = self.data2d.geometry
        ig2D = ag2D.get_ImageGeometry()
        self.A = ProjectionOperator(ig2D, ag2D, device = "cpu")
        self.n_subsets = 5
        self.partitioned_data=self.data2d.partition(self.n_subsets, 'sequential')
        self.A_partitioned = ProjectionOperator(ig2D, self.partitioned_data.geometry, device = "cpu")
        self.f_subsets = []
        for i in range(self.n_subsets):
            fi=LeastSquares(self.A_partitioned.operators[i],self. partitioned_data[i])
            self.f_subsets.append(fi)
        self.f=LeastSquares(self.A, self.data2d)
        self.f_stochastic=LSVRGFunction(self.f_subsets,self.sampler)
        self.initial=ig2D.allocate()


    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f=LSVRGFunction(self.f_subsets)
        self.assertTrue(isinstance( f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    def test_direct(self):
        self.assertAlmostEqual(self.f_stochastic(self.initial), self.f(self.initial),1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(self.initial).array, self.f.gradient(self.initial).array,2)

    def test_value_error_with_only_one_function(self):
        with self.assertRaises(ValueError):
            LSVRGFunction([self.f], self.sampler)
            pass
    def test_type_error_if_functions_not_a_list(self):
        with self.assertRaises(TypeError):
            LSVRGFunction(self.f, self.sampler)

    def test_LSVRG_init(self):
        pass
    
    def test_LSVRG_data_passes(self):
        pass
    
    def test_LSVRG_update_frequency(self):
        pass
    
    def test_LSVRG_store_gradients(self):
        pass


    def test_sampler_without_next(self):
        class bad_Sampler():
            def init(self):
                pass
        bad_sampler=bad_Sampler()
        with self.assertRaises(ValueError):
           SVRGFunction([self.f, self.f], bad_sampler)


    def test_LSVRG_simulated_parallel_beam_data(self): 

        rate = self.f.L
        alg = GD(initial=self.initial, 
                              objective_function=self.f, update_objective_interval=500,
                              rate=rate, alpha=1e8)
        alg.max_iteration = 200
        alg.run(verbose=0)


        objective=self.f_stochastic
        alg_stochastic = GD(initial=self.initial, 
                              objective_function=objective, update_objective_interval=500, 
                              step_size=5e-8, max_iteration =5000)
        alg_stochastic.run( self.n_subsets*25, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)



    def test_LSVRG_toy_example(self): 
        sampler=Sampler.random_with_replacement(3, seed=1)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b =  VectorData(np.random.normal(0,3,15))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(15)
            diagonal[5*i:5*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))
            if i==0:
               objective=LeastSquares(A, A.direct(b))
            else:
               objective+=LeastSquares(A, A.direct(b))

        rate = objective.L / 3.

        alg = GD(initial=initial, 
                              objective_function=objective, update_objective_interval=1000,
                              rate=rate, atol=1e-9, rtol=1e-6)
        alg.max_iteration = 80
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        stochastic_objective=LSVRGFunction(functions, sampler)
        self.assertAlmostEqual(stochastic_objective(initial), objective(initial))   
        self.assertNumpyArrayAlmostEqual(stochastic_objective.full_gradient(initial).array, objective.gradient(initial).array)



        alg_stochastic = GD(initial=initial, 
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.05, max_iteration =5000)
        alg_stochastic.run( 100, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), b.as_array(),3)

    def test_LSVRG_toy_example_store_gradients(self): 
        sampler=Sampler.random_with_replacement(3, seed=1)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b =  VectorData(np.random.normal(0,3,15))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(15)
            diagonal[5*i:5*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))
            if i==0:
               objective=LeastSquares(A, A.direct(b))
            else:
               objective+=LeastSquares(A, A.direct(b))

        rate = objective.L / 3.

        alg = GD(initial=initial, 
                              objective_function=objective, update_objective_interval=1000,
                              rate=rate, atol=1e-9, rtol=1e-6)
        alg.max_iteration = 80
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        stochastic_objective=LSVRGFunction(functions, sampler, store_gradients=True)
        self.assertAlmostEqual(stochastic_objective(initial), objective(initial))   
        self.assertNumpyArrayAlmostEqual(stochastic_objective.full_gradient(initial).array, objective.gradient(initial).array)



        alg_stochastic = GD(initial=initial, 
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.05, max_iteration =5000)
        alg_stochastic.run( 100, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), b.as_array(),3)
