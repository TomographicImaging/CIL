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


from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import ApproximateGradientSumFunction
from cil.optimisation.functions import SGFunction, SAGFunction, SAGAFunction
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


class approx_gradient_child_class_testing():

    def set_up(self):
        np.random.seed(10)
        self.sampler = Sampler.random_with_replacement(self.n_subsets)
        self.initial = VectorData(np.zeros(30))
        b = VectorData(np.array(range(30))/50)

        self.f_subsets = []
        for i in range(self.n_subsets):
            diagonal = np.zeros(30)
            diagonal[(30//self.n_subsets)*i:(30//self.n_subsets)*(i+1)] = 1
            Ai = MatrixOperator(np.diag(diagonal))
            self.f_subsets.append(LeastSquares(Ai, Ai.direct(b)))
        self.A=MatrixOperator(np.diag(np.ones(30)))
        self.f = LeastSquares(self.A, b)

        self.f_stochastic=self.stochastic_estimator(self.f_subsets, self.sampler)

    def test_approximate_gradient_not_equal_full(self):
        self.f_stochastic.gradient(self.initial)
        self.assertFalse((self.f_stochastic.full_gradient(
            self.initial+1) == self.f_stochastic.gradient(self.initial+1).array).all())

    def test_sampler(self):
        self.assertTrue(isinstance(self.f_stochastic.sampler, SamplerRandom))
        f = self.stochastic_estimator(self.f_subsets)
        self.assertTrue(isinstance(f.sampler, SamplerRandom))
        self.assertEqual(f.sampler._type, 'random_with_replacement')

    def test_call(self):
        self.assertAlmostEqual(self.f_stochastic(
            self.initial), self.f(self.initial), 1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(
            self.initial).array, self.f.gradient(self.initial).array, 2)

    def test_error_without_function(self):
        self.stochastic_estimator([self.f], self.sampler)
        with self.assertRaises((IndexError, ZeroDivisionError)):
            self.stochastic_estimator([], self.sampler)

    def test_type_error_if_functions_not_a_list(self):
        with self.assertRaises(TypeError):
            self.stochastic_estimator(self.f, self.sampler)

    def test_sampler_without_next(self):
        class bad_Sampler():
            def init(self):
                pass
        bad_sampler = bad_Sampler()
        with self.assertRaises(ValueError):
            self.stochastic_estimator([self.f, self.f], bad_sampler)


    def test_sampler_out_of_range(self):
        def g(index):
            return -2
        bad_sampler = Sampler.from_function(12,g)
        f = self.stochastic_estimator([self.f]*10, bad_sampler)
        with self.assertRaises(IndexError):
            f.gradient(self.initial)
            f.gradient(self.initial)

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_toy_example(self):
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

        stochastic_objective = self.stochastic_estimator(functions, sampler)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=0.05)
        alg_stochastic.run(600, verbose=0)

        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)


class TestSGD(CCPiTestClass, approx_gradient_child_class_testing):

    def setUp(self):
        self.stochastic_estimator=SGFunction
        self.n_subsets=6
        self.set_up()

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




class TestSAG(CCPiTestClass, approx_gradient_child_class_testing):

    def setUp(self):
        self.stochastic_estimator=SAGFunction
        self.n_subsets=6
        self.set_up()

    def test_warm_start_and_data_passes(self):
        f1=SAGFunction(self.f_subsets,Sampler.sequential(self.n_subsets))
        f=SAGFunction(self.f_subsets,Sampler.sequential(self.n_subsets))
        f.warm_start_approximate_gradients(self.initial)
        f1.gradient(self.initial)
        f.gradient(self.initial)
        self.assertEqual(f.function_num, 0)
        self.assertEqual(f1.function_num, 0)
        self.assertNumpyArrayAlmostEqual(np.array(f1.data_passes), np.array([1./f1.num_functions]))
        self.assertNumpyArrayAlmostEqual(np.array(f.data_passes), np.array([ 1+1./f1.num_functions]))
        self.assertNumpyArrayAlmostEqual(np.array(f.data_passes_indices[0]), np.array(list(range(f1.num_functions))+ [0]))
        self.assertNumpyArrayAlmostEqual(np.array(f1.data_passes_indices[0]), np.array([0]))
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[0].array, f1._list_stored_gradients[0].array)
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[0].array, self.f_subsets[0].gradient(self.initial).array)
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[1].array, self.f_subsets[1].gradient(self.initial).array)

        self.assertFalse((f._list_stored_gradients[3].array== f1._list_stored_gradients[3].array).all())

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_SAG_toy_example_warm_start(self):
        sampler=Sampler.random_with_replacement(3,seed=1)
        initial = VectorData(np.zeros(21))
        np.random.seed(4)
        b =  VectorData(np.random.normal(0,4,21))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(21)
            diagonal[7*i:7*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))

        Aop=MatrixOperator(np.diag(np.ones(21)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4)

        stochastic_objective=SAGFunction(functions, sampler)
        stochastic_objective.warm_start_approximate_gradients(initial)

        alg_stochastic = GD(initial=initial,
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.05, max_iteration =5000)
        alg_stochastic.run( 80, verbose=0)
        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)


class TestSAGA(CCPiTestClass,approx_gradient_child_class_testing):

    def setUp(self):
        self.stochastic_estimator=SAGAFunction
        self.n_subsets=6
        self.set_up()

    def test_warm_start_and_data_passes(self):
        f1=SAGAFunction(self.f_subsets,Sampler.sequential(self.n_subsets))
        f=SAGAFunction(self.f_subsets,Sampler.sequential(self.n_subsets))
        f.warm_start_approximate_gradients(self.initial)
        f1.gradient(self.initial)
        f.gradient(self.initial)

        self.assertEqual(f.function_num, 0)
        self.assertEqual(f1.function_num, 0)
        self.assertNumpyArrayAlmostEqual(np.array(f1.data_passes), np.array([1./f1.num_functions]))
        self.assertNumpyArrayAlmostEqual(np.array(f.data_passes), np.array([1+1./f1.num_functions]))
        self.assertNumpyArrayAlmostEqual(np.array(f.data_passes_indices[0]),np.array( list(range(self.n_subsets))+[0]))
        self.assertNumpyArrayAlmostEqual(np.array(f1.data_passes_indices[0]), np.array([0]))
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[0].array, f1._list_stored_gradients[0].array)
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[0].array, self.f_subsets[0].gradient(self.initial).array)
        self.assertNumpyArrayAlmostEqual(f._list_stored_gradients[1].array, self.f_subsets[1].gradient(self.initial).array)

        self.assertFalse((f._list_stored_gradients[1].array== f1._list_stored_gradients[1].array).all())
        self.assertNumpyArrayAlmostEqual(f1._list_stored_gradients[1].array, self.initial.array)

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_SAGA_toy_example_warm_start(self):
        sampler=Sampler.random_with_replacement(3,seed=1)
        initial = VectorData(np.zeros(21))
        np.random.seed(4)
        b =  VectorData(np.random.normal(0,4,21))
        functions=[]
        for i in range(3):
            diagonal=np.zeros(21)
            diagonal[7*i:7*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            functions.append( LeastSquares(A, A.direct(b)))

        Aop=MatrixOperator(np.diag(np.ones(21)))

        u_cvxpy = cvxpy.Variable(b.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - Aop.direct(b).array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4)

        stochastic_objective=SAGAFunction(functions, sampler)
        stochastic_objective.warm_start_approximate_gradients(initial)

        alg_stochastic = GD(initial=initial,
                              objective_function=stochastic_objective, update_objective_interval=1000,
                              step_size=0.05, max_iteration =5000)
        alg_stochastic.run( 100, verbose=0)
        np.testing.assert_allclose(p.value ,stochastic_objective(alg_stochastic.x) , atol=1e-1)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), u_cvxpy.value, 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)
