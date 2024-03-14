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
        
        
    
    @unittest.skipUnless(has_astra, "Requires ASTRA GPU")
    def test_SGD_simulated_parallel_beam_data(self):
        
        sampler = Sampler.random_with_replacement(5)
        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        data.reorder('astra')
        data2d = data.get_slice(vertical='centre')
        ag2D = data2d.geometry
        ag2D.set_angles(ag2D.angles, initial_angle=0.2, angle_unit='radian')
        ig2D = ag2D.get_ImageGeometry()
        
        A = ProjectionOperator(ig2D, ag2D, device="cpu")
        n_subsets = 5
        partitioned_data = data2d.partition(
            n_subsets, 'sequential')
        A_partitioned = ProjectionOperator(
            ig2D, partitioned_data.geometry, device="cpu")
        f_subsets = []
        for i in range(n_subsets):
            fi = LeastSquares(
                A_partitioned.operators[i],  partitioned_data[i])
            f_subsets.append(fi)
        f = LeastSquares(A, data2d)
        f_stochastic = SGFunction(f_subsets, sampler)
        initial = ig2D.allocate()
            
       
        alg = GD(initial=initial,
                 objective_function=f, update_objective_interval=500, alpha=1e8)
        alg.max_iteration = 200
        alg.run(verbose=0)

        objective = f_stochastic
        alg_stochastic = GD(initial=initial,
                            objective_function=objective, update_objective_interval=500,
                            step_size=1/f_stochastic.L, max_iteration=5000)
        alg_stochastic.run(n_subsets*50, verbose=0)
        self.assertAlmostEqual(objective.data_passes[-1], n_subsets*50/n_subsets)
        self.assertListEqual(objective.data_passes_indices[-1], [objective.function_num])
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), alg.x.as_array(), 3)
        
    

    def test_SGD_toy_example(self):
        sampler = Sampler.random_with_replacement(5)
        initial = VectorData(np.zeros(25))
        b = VectorData(np.array(range(25)))
        functions = []
        for i in range(5):
            diagonal = np.zeros(25)
            diagonal[5*i:5*(i+1)] = 1
            A = MatrixOperator(np.diag(diagonal))
            functions.append(LeastSquares(A, A.direct(b)))
            if i == 0:
                objective = LeastSquares(A, A.direct(b))
            else:
                objective += LeastSquares(A, A.direct(b))

        alg = GD(initial=initial,
                 objective_function=objective, update_objective_interval=1000, atol=1e-9, rtol=1e-6)
        alg.max_iteration = 600
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        stochastic_objective = SGFunction(functions, sampler)
        self.assertAlmostEqual(
            stochastic_objective(initial), objective(initial))
        self.assertNumpyArrayAlmostEqual(stochastic_objective.full_gradient(
            initial).array, objective.gradient(initial).array)

        alg_stochastic = GD(initial=initial,
                            objective_function=stochastic_objective, update_objective_interval=1000,
                            step_size=0.01, max_iteration=5000)
        alg_stochastic.run(600, verbose=0)
        self.assertAlmostEqual(stochastic_objective.data_passes[-1], 600/5)
        self.assertListEqual(stochastic_objective.data_passes_indices[-1], [stochastic_objective.function_num])
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), alg.x.as_array(), 3)
        self.assertNumpyArrayAlmostEqual(
            alg_stochastic.x.as_array(), b.as_array(), 3)

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        np.random.seed(10)
        n = 300  
        m = 100 
        A = np.random.normal(0,1, (m, n)).astype('float32')
        b = np.random.normal(0,1, m).astype('float32')

        Aop = MatrixOperator(A)
        bop = VectorData(b) 
        n_subsets = 10
        Ai = np.vsplit(A, n_subsets) 
        bi = [b[i:i+int(m/n_subsets)] for i in range(0, m, int(m/n_subsets))]     
        fi_cil = []
        for i in range(n_subsets):   
            Ai_cil = MatrixOperator(Ai[i])
            bi_cil = VectorData(bi[i])
            fi_cil.append(LeastSquares(Ai_cil, bi_cil, c = 0.5))
        F = LeastSquares(Aop, b=bop, c = 0.5)     
        ig = Aop.domain  
        initial= ig.allocate(0) 
        sampler=Sampler.random_with_replacement(n_subsets)
        F_SG=SGFunction(fi_cil, sampler)
        u_cvxpy = cvxpy.Variable(ig.shape[0])
        objective = cvxpy.Minimize( 0.5*cvxpy.sum_squares(Aop.A @ u_cvxpy - bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 

        step_size = 1./F_SG.L

        epochs = 200
        sgd = GD(initial = initial, objective_function = F_SG, step_size = step_size,
                    max_iteration = epochs * n_subsets, 
                    update_objective_interval = epochs * n_subsets)
        sgd.run(verbose=0)    

        np.testing.assert_allclose(p.value, sgd.objective[-1], atol=1e-1)
        np.testing.assert_allclose(u_cvxpy.value, sgd.solution.array, atol=1e-1)    