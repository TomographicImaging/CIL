# -*- coding: utf-8 -*-
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


import numpy as np

from cil.utilities.errors import InPlaceError
from cil.framework import AcquisitionGeometry
from cil.optimisation.operators import IdentityOperator
from cil.optimisation.functions import  KullbackLeibler, ConstantFunction, TranslateFunction, soft_shrinkage


from cil.optimisation.functions import  KullbackLeibler, WeightedL2NormSquared, L2NormSquared, \
    L1Norm, L2NormSquared, MixedL21Norm, LeastSquares, \
    SmoothMixedL21Norm, OperatorCompositionFunction, \
     IndicatorBox, TotalVariation,  SumFunction, SumScalarFunction, \
    WeightedL2NormSquared, MixedL11Norm, ZeroFunction


import numpy


from cil.framework import  BlockGeometry
from cil.optimisation.functions import TranslateFunction
from timeit import default_timer as timer

import numpy as np


from testclass import CCPiTestClass
from cil.utilities.quality_measures import mae


from utils import  initialise_tests



initialise_tests()


class TestFunction(CCPiTestClass):

    def setUp(self):

        self.ag = AcquisitionGeometry.create_Parallel2D()
        angles = np.linspace(0, 360, 10, dtype=np.float32)

        # default
        self.ag.set_angles(angles)
        self.ag.set_panel(10)

        self.ig = self.ag.get_ImageGeometry()

        scalar = 4

        b = self.ag.allocate('random', seed=2)
        weight_ls = self.ig.allocate('random', seed=2)

        numpy.random.seed(1)

        A = IdentityOperator(self.ig)
        b_ig = self.ig.allocate('random')
        c = numpy.float64(0.3)
        self.bg = BlockGeometry(self.ig, self.ig)

        self.func_geom_test_list = [
            (IndicatorBox(), self.ag),
            (KullbackLeibler(b=b, backend='numba'), self.ag),
            (KullbackLeibler(b=b, backend='numpy'), self.ag),
            (L1Norm(), self.ag),
            (L1Norm(), self.ig),
            (L1Norm(b=b), self.ag),
            (L1Norm(b=b, weight=b), self.ag),
            (TranslateFunction(L1Norm(), b), self.ag),
            (TranslateFunction(L2NormSquared(), b), self.ag),
            (L2NormSquared(), self.ag),
            (scalar * L2NormSquared(), self.ag),
            (SumFunction(L2NormSquared(), scalar * L2NormSquared()), self.ag),
            (SumScalarFunction(L2NormSquared(), 3), self.ag),
            (ConstantFunction(3), self.ag),
            (ZeroFunction(), self.ag),
            (L2NormSquared(b=b), self.ag),
            (L2NormSquared(), self.ag),
            (LeastSquares(A, b_ig, c, weight_ls), self.ig),
            (LeastSquares(A, b_ig, c), self.ig),
            (WeightedL2NormSquared(weight=b_ig), self.ig),
            (TotalVariation(backend='c', warm_start=False, max_iteration=100), self.ig),
            (TotalVariation(backend='numpy', warm_start=False, max_iteration=100), self.ig),
            (OperatorCompositionFunction(L2NormSquared(), A), self.ig),
            (MixedL21Norm(), self.bg),
            (SmoothMixedL21Norm(epsilon=0.3), self.bg),
            (MixedL11Norm(), self.bg)

        ]

        np.random.seed(5)
        self.data_arrays=[np.random.normal(0,1, (10,10)), np.array(range(0,65500, 655)).reshape((10,10)), np.random.uniform(-0.1,1,(10,10))]

    def get_result(self, function, method, x, *args):
        try:
            input=x.copy()
            if method == 'proximal':
                out= function.proximal(x, *args)
            elif method == 'proximal_conjugate':
                out= function.proximal_conjugate(x, *args)
            elif method == 'gradient':
                out= function.gradient(x, *args)
            self.assertDataArraysInContainerAllClose(input, x, rtol=1e-5, msg= "In case func."+method+'(data, *args) where func is  ' + function.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
            return out
        except NotImplementedError:
            print(function.__class__.__name__+method)
            return None
        
    def in_place_test(self, function, method,desired_result,   x, *args, ):
            out3 = x.copy()
            try:
                if method == 'proximal':
                    function.proximal(out3, *args, out=out3)
                elif method == 'proximal_conjugate':
                    function.proximal_conjugate(out3, *args, out=out3)
                elif method == 'gradient':
                    function.gradient(out3, *args, out=out3)
                self.assertDataArraysInContainerAllClose(desired_result, out3, rtol=1e-5, msg= "In place calculation failed for func."+method+'(data, *args, out=data) where func is  ' + function.__class__.__name__+ '. ')
      
            except (InPlaceError, NotImplementedError):
                print(function.__class__.__name__+method)
 
        
    def out_test(self, function, method,desired_result,   x, *args, ):
        input = x.copy()
        out2=0*(x.copy())
        try:
            if method == 'proximal':
                function.proximal(input, *args, out=out2)
            elif method == 'proximal_conjugate':
                function.proximal_conjugate(input, *args, out=out2)
            elif method == 'gradient':
                function.gradient(input, *args, out=out2)
            self.assertDataArraysInContainerAllClose(desired_result, out2, rtol=1e-5, msg= "Calculation failed using `out` in func."+method+'(x, *args, out=data) where func is  ' + function.__class__.__name__+ '. ')
            self.assertDataArraysInContainerAllClose(input, x,  rtol=1e-5, msg= "In case func."+method+'(data, *args, out=out) where func is  ' + function.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
      
        except (InPlaceError, NotImplementedError):
            print(function.__class__.__name__+method)
            
            
   
    def test_proximal_conjugate_out(self):
        for func, geom in self.func_geom_test_list:
            for data_array in self.data_arrays:
                data=geom.allocate(0)
                data.fill(data_array)
                result=self.get_result(func, 'proximal_conjugate', data, 0.5)
                self.out_test(func, 'proximal_conjugate', result,  data, 0.5)
                self.in_place_test(func, 'proximal_conjugate', result,  data, 0.5)
    
    def test_proximal_out(self):
        for func, geom in self.func_geom_test_list:
            for data_array in self.data_arrays:
                data=geom.allocate(0)
                data.fill(data_array)
                result=self.get_result(func, 'proximal', data, 0.5)
                self.out_test(func, 'proximal', result,  data, 0.5)
                self.in_place_test(func, 'proximal', result,  data, 0.5)
                
    def test_gradient_out(self):
        for func, geom in self.func_geom_test_list:
            if (not isinstance(func, TotalVariation)) and (not isinstance(func, IndicatorBox)):
                for data_array in self.data_arrays:
                    data=geom.allocate(0)
                    data.fill(data_array)
                    result=self.get_result(func, 'gradient', data)
                    self.out_test(func, 'gradient', result,  data)
                    self.in_place_test(func, 'gradient', result,  data)
                
    
        


