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

#TODO: remove unused packages 
import unittest
from utils import initialise_tests
import numpy
import numpy as np
from numpy import nan, inf
from cil.framework import VectorData
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.framework import BlockDataContainer
from cil.framework import BlockGeometry

from cil.optimisation.operators import IdentityOperator
from cil.optimisation.operators import GradientOperator, BlockOperator, MatrixOperator

from cil.optimisation.functions import LeastSquares, ZeroFunction, \
   L2NormSquared, OperatorCompositionFunction
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L1Norm, KullbackLeibler                     
from cil.optimisation.functions import IndicatorBox

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.algorithms import GD
from cil.optimisation.algorithms import CGLS
from cil.optimisation.algorithms import SIRT
from cil.optimisation.algorithms import FISTA
from cil.optimisation.algorithms import SPDHG
from cil.optimisation.algorithms import PDHG
from cil.optimisation.algorithms import LADMM


from cil.utilities import dataexample
from cil.utilities import noise as applynoise
import time
import warnings
from cil.optimisation.functions import Rosenbrock
from cil.optimisation.functions import ApproximateGradientSumFunction
from cil.optimisation.functions import SGFunction
#from cil.optimisation.utilities import Sampler 

from cil.framework import VectorData, VectorGeometry
from cil.utilities.quality_measures import mae, mse, psnr
# Fast Gradient Projection algorithm for Total Variation(TV)
from cil.optimisation.functions import TotalVariation
import logging
from testclass import CCPiTestClass
from utils import  has_astra

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator

class TestApproximateGradientSumFunction(CCPiTestClass):

    def setUp(self):
        self.sampler=Sampling(5)
        self.initial = VectorData(np.zeros(25))
        self.b =  VectorData(np.random.normal(0,1,25))
        self.functions=[]
        for i in range(5):
            diagonal=np.zeros(25)
            diagonal[5*i:5*(i+1)]=1
            A=MatrixOperator(np.diag(diagonal))
            self.functions.append( LeastSquares(A, A.direct(self.b)))
            if i==0:
               self.objective=LeastSquares(A, A.direct(self.b))
            else:
               self.objective+=LeastSquares(A, A.direct(self.b))
        self.stochastic_objective=ApproximateGradientSumFunction(self.functions, self.sampler)
    def test_init(self):
        with self.assertRaises(NotImplementedError):
            self.stochastic_objective.approximate_gradient(3, self.initial)
        with self.assertRaises(NotImplementedError):
            self.stochastic_objective.gradient( self.initial)
        self.assertAlmostEqual(self.stochastic_objective(self.initial), self.objective(self.initial))
        self.assertEqual(self.stochastic_objective.num_functions,5)
        
        #TODO: 
class Sampling(): #TO BE REPLACED BY SAMPLING CLASS THING WHEN THAT HAS BEEN MERGED 
    def __init__(self, num_subsets, prob=None, seed=99):
        self.num_subsets=num_subsets
        np.random.seed(seed)

        if prob==None:
            self.prob = [1/self.num_subsets] * self.num_subsets
        else:
            self.prob=prob
    def __next__(self):
        
            return int(np.random.choice(self.num_subsets, 1, p=self.prob))

class TestSGD(CCPiTestClass):

 ##   def test_SGD_one_function(self): 
  #      ig = ImageGeometry(12,13,14)
  #      initial = ig.allocate()
  #      b = ig.allocate('random')
  #      identity = IdentityOperator(ig)
  #      
  #      norm2sq = LeastSquares(identity, b)
  #      rate = norm2sq.L / 3.
  #      sampler=Sampling(1)

  #      objective=SGFunction([norm2sq], sampler)
  #      alg = GD(initial=initial, 
   #                           objective_function=objective, 
  #                            rate=rate, atol=1e-9, rtol=1e-6)
   #     alg.max_iteration = 1000
   #     alg.run(verbose=0)
   #     self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
   #     alg = GD(initial=initial, 
   #                           objective_function=objective, 
   #                           rate=rate, max_iteration=20,
   #                           update_objective_interval=2,
   #                           atol=1e-9, rtol=1e-6)#

 #       alg.run(20, verbose=0)
  #      self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
    def setUp(self):
        self.sampler=Sampling(5)
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
        f_subsets = []
        for i in range(self.n_subsets):
            fi=LeastSquares(self.A_partitioned.operators[i],self. partitioned_data[i])
            f_subsets.append(fi)
        self.f=LeastSquares(self.A, self.data2d)
        self.f_stochastic=SGFunction(f_subsets,self.sampler)
        self.initial=ig2D.allocate()

    def test_SGD_toy_example(self): 
        sampler=Sampling(5)
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
        alg.max_iteration = 400
        alg.run(verbose=0)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        
        objective=SGFunction(functions, sampler)
        alg_stochastic = GD(initial=initial, 
                              objective_function=objective, update_objective_interval=1000,
                              step_size=0.01, max_iteration =5000)
        alg_stochastic.run( 400, verbose=0)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), alg.x.as_array(),3)
        self.assertNumpyArrayAlmostEqual(alg_stochastic.x.as_array(), b.as_array(),3)


                                

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
        
  
    
    def test_approximate_gradient(self):
        self.assertFalse((self.f_stochastic.full_gradient(self.initial)==self.f_stochastic.gradient(self.initial).array).all())

    def test_sampler(self):
        pass #TODO: 

    def test_direct(self):
        self.assertAlmostEqual(self.f_stochastic(self.initial), self.f(self.initial),1)

    def test_full_gradient(self):
        self.assertNumpyArrayAlmostEqual(self.f_stochastic.full_gradient(self.initial).array, self.f.gradient(self.initial).array,2)
    


        
    