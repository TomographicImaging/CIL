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
from utils import has_astra

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy
    
    
class TestApproximateGradientSumFunction(CCPiTestClass):

    def setUp(self):
        self.sampler = Sampler.random_with_replacement(5)
        self.initial = VectorData(np.zeros(10))
        self.b = VectorData(np.random.normal(0, 1, 10))
        self.functions = []
        for i in range(5):
            diagonal = np.zeros(10)
            diagonal[2*i:2*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            self.functions.append(LeastSquares(A, A.direct(self.b)))
            if i == 0:
                self.objective = LeastSquares(A, A.direct(self.b))
            else:
                self.objective += LeastSquares(A, A.direct(self.b))

    def test_ABC(self):
        with self.assertRaises(TypeError):
            self.stochastic_objective = ApproximateGradientSumFunction(
                self.functions, self.sampler)


class TestSGD(CCPiTestClass):
    
    def setUp(self):
        
        
           
        self.sampler = Sampler.random_with_replacement(6)
        self.initial = VectorData(np.zeros(30))
        b = VectorData(np.array(range(30))/50)
        self.n_subsets = 6
        self.f_subsets = []
        for i in range(6):
            diagonal = np.zeros(30)
            diagonal[5*i:5*(i+1)] = 1
            Ai = MatrixOperator(np.diag(diagonal))
            self.f_subsets.append(LeastSquares(Ai, Ai.direct(b)))
        self.A=MatrixOperator(np.diag(np.ones(30)))
        self.f = LeastSquares(self.A, b)
        self.f_stochastic = SGFunction(self.f_subsets, self.sampler)
            
            

    def test_approximate_gradient_not_equal_full(self):
        self.assertFalse((self.f_stochastic.full_gradient(
            self.initial) == self.f_stochastic.gradient(self.initial).array).all())

    
    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f = SGFunction(self.f_subsets)
        self.assertTrue(isinstance(f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    
    def test_call(self):
        self.assertAlmostEqual(self.f_stochastic(
            self.initial), self.f(self.initial), 1)

    
    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(
            self.initial).array, self.f.gradient(self.initial).array, 2)

    
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
        bad_sampler = bad_Sampler()
        with self.assertRaises(ValueError):
            SGFunction([self.f, self.f], bad_sampler)

    
    def test_sampler_out_of_range(self):
        bad_sampler = Sampler.sequential(10)
        f = SGFunction([self.f, self.f], bad_sampler)
        with self.assertRaises(IndexError):
            f.gradient(self.initial)
            f.gradient(self.initial)
            f.gradient(self.initial)
            
    def test_partition_weights(self):
        f_stochastic=SGFunction(self.f_subsets, Sampler.sequential(self.n_subsets))
        self.assertListEqual(f_stochastic._partition_weights, [1 / self.n_subsets] * self.n_subsets)
        with self.assertRaises(ValueError):
            f_stochastic.set_data_partition_weights( list(range(self.n_subsets)))
        with self.assertRaises(ValueError):
            f_stochastic.set_data_partition_weights( [1])
        with self.assertRaises(ValueError):
            f_stochastic.set_data_partition_weights( [-1]+[2/(self.n_subsets-1)]*(self.n_subsets-1))
        a=[i/float(sum(range(self.n_subsets))) for i in range(self.n_subsets)]
        f_stochastic.set_data_partition_weights( a)
        self.assertListEqual(f_stochastic._partition_weights, a )
        f_stochastic.gradient(self.initial)
        for i in range(1,20):
            f_stochastic.gradient(self.initial)
            self.assertEqual(f_stochastic.data_passes[i], f_stochastic.data_passes[i-1]+a[i%self.n_subsets])
        
        
    
  
    
    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_SGD_toy_example(self):
        sampler = Sampler.random_with_replacement(5)
        initial = VectorData(np.zeros(25))
        b = VectorData(np.array(range(25)))
        functions = []
        for i in range(5):
            diagonal = np.zeros(25)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(0.5*LeastSquares(A, A.direct(b)))
            
        Aop=MatrixOperator(np.diag(np.ones(25)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 
        
        
        stochastic_objective = SGFunction(functions, sampler)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=1/stochastic_objective.L)
        alg_stochastic.run(600, verbose=0)
        self.assertAlmostEqual(stochastic_objective.data_passes[-1], 600/5)
        self.assertListEqual(stochastic_objective.data_passes_indices[-1], [stochastic_objective.function_num])

        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)
        


class TestSVRG(CCPiTestClass):

    def setUp(self):

       
           
        self.sampler = Sampler.random_with_replacement(5)
        self.initial = VectorData(np.zeros(30))
        b = VectorData(np.array(range(30))/50)
        self.n_subsets = 5
        self.f_subsets = []
        for i in range(5):
            diagonal = np.zeros(30)
            diagonal[6*i:6*(i+1)] = 1
            Ai = MatrixOperator(np.diag(diagonal))
            self.f_subsets.append(LeastSquares(Ai, Ai.direct(b)))
        self.A=MatrixOperator(np.diag(np.ones(30)))
        self.f = LeastSquares(self.A, b)
        self.f_stochastic = SVRGFunction(self.f_subsets, self.sampler)

    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f = SVRGFunction(self.f_subsets)
        self.assertTrue(isinstance(f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    def test_direct(self):
        self.assertAlmostEqual(self.f_stochastic(
            self.initial), self.f(self.initial), 1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(
            self.initial).array, self.f.gradient(self.initial).array, 2)

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
        bad_sampler = bad_Sampler()
        with self.assertRaises(ValueError):
            SVRGFunction([self.f, self.f], bad_sampler)

    def test_SVRG_init(self):
        self.assertEqual(self.f_stochastic.update_frequency,
                         2*self.f_stochastic.num_functions)
        self.assertListEqual(self.f_stochastic.data_passes, [])
        self.assertListEqual(self.f_stochastic.data_passes_indices, [])
        self.assertEqual(self.f_stochastic.store_gradients, False)
        f2 = SVRGFunction(self.f_subsets, update_frequency=2,
                          store_gradients=True)
        self.assertEqual(f2.update_frequency, 2)
        self.assertListEqual(f2.data_passes, [])
        self.assertListEqual(f2.data_passes_indices, [])
        self.assertEqual(f2.store_gradients, True)

    def test_SVRG_update_frequency_and_data_passes(self):
        objective = SVRGFunction(self.f_subsets, self.sampler)
        alg_stochastic = GD(initial=self.initial,
                            objective_function=objective, update_objective_interval=500,
                            step_size=5e-8, max_iteration=5000)
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            np.array(objective.data_passes), np.array([1., 6./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            np.array(objective.data_passes), np.array([1., 6./5, 7./5, 8./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes), np.array([
                                         1., 6./5, 7./5, 8./5, 9./5, 10./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes), np.array(
            [1., 6./5, 7./5, 8./5, 9./5, 10./5, 11./5, 12./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes), np.array(
            [1., 6./5, 7./5, 8./5, 9./5, 10./5, 11./5, 12./5, 13./5, 14./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes), np.array(
            [1., 6./5, 7./5, 8./5, 9./5, 10./5, 11./5, 12./5, 13./5, 14./5, 19./5, 20./5]))
        objective = SVRGFunction(
            self.f_subsets, self.sampler, update_frequency=3)
        alg_stochastic = GD(initial=self.initial,
                            objective_function=objective, update_objective_interval=500,
                            step_size=5e-8, max_iteration=5000)
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            np.array(objective.data_passes), np.array([1., 6./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            np.array(objective.data_passes), np.array([1., 6./5, 7./5, 12./5]))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes), np.array([
                                         1., 6./5, 7./5, 12./5, 13./5, 14./5]))
        alg_stochastic.run(2, verbose=0)

    def test_SVRG_store_gradients(self):
        objective = SVRGFunction(self.f_subsets, self.sampler)
        with self.assertRaises(AttributeError):
            objective._list_stored_gradients
        objective.gradient(self.initial)
        with self.assertRaises(AttributeError):
            objective._list_stored_gradients

        objective = SVRGFunction(
            self.f_subsets, self.sampler, store_gradients=True)
        with self.assertRaises(AttributeError):
            objective._list_stored_gradients
        objective.gradient(self.initial)
        self.assertNumpyArrayAlmostEqual(
            objective._list_stored_gradients[0].array, self.f_subsets[0].gradient(self.initial).array)
        self.assertNumpyArrayAlmostEqual(
            objective._list_stored_gradients[1].array, self.f_subsets[1].gradient(self.initial).array)


    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_SVRG_toy_example_and_data_passes(self):
        sampler = Sampler.sequential(3)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b = VectorData(np.random.normal(0, 3, 15))
        functions = []
        for i in range(3):
            diagonal = np.zeros(15)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(LeastSquares(A, A.direct(b)))
        Aop=MatrixOperator(np.diag(np.ones(15)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 
        
        
        stochastic_objective = SVRGFunction(functions, sampler)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=1/stochastic_objective.L)
        alg_stochastic.run(14, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(stochastic_objective.data_passes), np.array(
            [1.]+[4./3, 5./3, 6./3, 7./3, 8./3, 11./3, 12./3, 13./3, 14./3, 15./3., 16./3, 19./3, 20./3]), 4)
        self.assertListEqual(stochastic_objective.data_passes_indices[:3], [list(range(3)), [0], [1]])

        alg_stochastic.run(100, verbose=0)
        self.assertListEqual(stochastic_objective.data_passes_indices[-1], [stochastic_objective.function_num])

        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)
        



    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_SVRG_toy_example_store_gradients(self):
        sampler = Sampler.sequential(3)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b = VectorData(np.random.normal(0, 3, 15))
        functions = []
        for i in range(3):
            diagonal = np.zeros(15)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(LeastSquares(A, A.direct(b)))
        Aop=MatrixOperator(np.diag(np.ones(15)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 
        
        
        stochastic_objective = SVRGFunction(functions, sampler, store_gradients=True)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=1/stochastic_objective.L)

        alg_stochastic.run(10, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(stochastic_objective.data_passes), 
           np.array( [1.]+[4./3, 5./3, 6./3, 7./3,  8./3, 11./3, 12./3, 13./3, 14./3]), 4)
        self.assertListEqual(stochastic_objective.data_passes_indices[:2], [list(range(3)), [0]])

        alg_stochastic.run(100, verbose=0)
        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)

    

class TestLSVRG(CCPiTestClass):

    def setUp(self):

        
        self.sampler = Sampler.random_with_replacement(5)
        self.initial = VectorData(np.zeros(30))
        b = VectorData(np.array(range(30))/50)
        self.n_subsets = 5
        self.f_subsets = []
        for i in range(5):
            diagonal = np.zeros(30)
            diagonal[6*i:6*(i+1)] = 1
            Ai = MatrixOperator(np.diag(diagonal))
            self.f_subsets.append(LeastSquares(Ai, Ai.direct(b)))
        self.A=MatrixOperator(np.diag(np.ones(30)))
        self.f = LeastSquares(self.A, b)
        self.f_stochastic = LSVRGFunction(self.f_subsets, self.sampler)

    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f = LSVRGFunction(self.f_subsets)
        self.assertTrue(isinstance(f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    def test_direct(self):
        self.assertAlmostEqual(self.f_stochastic(
            self.initial), self.f(self.initial), 1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(
            self.initial).array, self.f.gradient(self.initial).array, 2)

    def test_value_error_with_only_one_function(self):
        with self.assertRaises(ValueError):
            LSVRGFunction([self.f], self.sampler)
            pass

    def test_type_error_if_functions_not_a_list(self):
        with self.assertRaises(TypeError):
            LSVRGFunction(self.f, self.sampler)

    def test_LSVRG_init(self):
        self.assertEqual(self.f_stochastic.update_prob, 1/self.n_subsets)
        self.assertListEqual(self.f_stochastic.data_passes, [])
        self.assertListEqual(self.f_stochastic.data_passes_indices, [])
        self.assertEqual(self.f_stochastic.store_gradients, False)

        f2 = LSVRGFunction(self.f_subsets, update_prob=1 /
                           2, store_gradients=True, seed=1)
        self.assertEqual(f2.update_prob, 1/2)
        self.assertListEqual(f2.data_passes, [])
        self.assertEqual(f2.store_gradients, True)

    def test_LSVRG_data_passes_and_update_prob_and_seed(self):
        objective = LSVRGFunction(self.f_subsets, self.sampler, update_prob=1)
        alg_stochastic = GD(initial=self.initial,  update_objective_interval=500,
                            objective_function=objective,                               step_size=5e-8, max_iteration=5000)
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            np.array(objective.data_passes), np.array([1., 2,]))
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes_indices[-1]), np.array(list(range(self.n_subsets))))
        alg_stochastic.run(2, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            np.array(objective.data_passes), np.array([1., 2., 3., 4.]))
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes_indices[-1]), np.array(list(range(self.n_subsets))))
        objective = LSVRGFunction(self.f_subsets, self.sampler, seed=3)
        alg_stochastic = GD(initial=self.initial,
                            objective_function=objective, update_objective_interval=500,
                            step_size=5e-8, max_iteration=5000)
        alg_stochastic.run(10, verbose=0)
        self.assertNumpyArrayAlmostEqual(np.array(objective.data_passes), 
            np.array([1., 2., 2.2, 2.4, 2.6, 3.6, 3.8, 4., 5., 5.2]))

    def test_LSVRG_store_gradients(self):
        objective = LSVRGFunction(self.f_subsets, self.sampler)
        with self.assertRaises(AttributeError):
            objective._list_stored_gradients
        objective.gradient(self.initial)
        with self.assertRaises(AttributeError):
            objective._list_stored_gradients

        objective = LSVRGFunction(
            self.f_subsets, self.sampler, store_gradients=True)
        with self.assertRaises(AttributeError):
            objective._list_stored_gradients
        objective.gradient(self.initial)
        self.assertNumpyArrayAlmostEqual(
            objective._list_stored_gradients[0].array, self.f_subsets[0].gradient(self.initial).array)
        self.assertNumpyArrayAlmostEqual(
            objective._list_stored_gradients[1].array, self.f_subsets[1].gradient(self.initial).array)

    def test_sampler_without_next(self):
        class bad_Sampler():
            def init(self):
                pass
        bad_sampler = bad_Sampler()
        with self.assertRaises(ValueError):
            SVRGFunction([self.f, self.f], bad_sampler)

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_LSVRG_toy_example_and_data_passes(self):
        sampler = Sampler.sequential(3)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b = VectorData(np.random.normal(0, 3, 15))
        functions = []
        for i in range(3):
            diagonal = np.zeros(15)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(LeastSquares(A, A.direct(b)))
        Aop=MatrixOperator(np.diag(np.ones(15)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 
        
        
        stochastic_objective = LSVRGFunction(functions, sampler)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=1/stochastic_objective.L)

        alg_stochastic.run(100, verbose=0)
        self.assertListEqual(stochastic_objective.data_passes_indices[-1], [stochastic_objective.function_num])

        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)
        



    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_LSVRG_toy_example_store_gradients(self):
        sampler = Sampler.sequential(3)
        initial = VectorData(np.zeros(15))
        np.random.seed(4)
        b = VectorData(np.random.normal(0, 3, 15))
        functions = []
        for i in range(3):
            diagonal = np.zeros(15)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(LeastSquares(A, A.direct(b)))
        Aop=MatrixOperator(np.diag(np.ones(15)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 
        
        
        stochastic_objective = LSVRGFunction(functions, sampler, store_gradients=True)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=1/stochastic_objective.L)



        alg_stochastic.run(100, verbose=0)
        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)

