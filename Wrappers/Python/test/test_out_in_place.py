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
from cil.framework import AcquisitionGeometry, ImageGeometry, VectorGeometry
from cil.optimisation.operators import IdentityOperator
from cil.optimisation.functions import  KullbackLeibler, ConstantFunction, TranslateFunction, soft_shrinkage, BlockFunction
from cil.optimisation.operators import LinearOperator, MatrixOperator
from cil.optimisation.operators import SumOperator,  ZeroOperator, CompositionOperator, ProjectionMap
from cil.optimisation.operators import BlockOperator,\
    FiniteDifferenceOperator, SymmetrisedGradientOperator,  DiagonalOperator, MaskOperator, ChannelwiseOperator, BlurringOperator

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


class TestFunctionOutAndInPlace(CCPiTestClass):

    def setUp(self):

        ag = AcquisitionGeometry.create_Parallel2D()
        angles = np.linspace(0, 360, 10, dtype=np.float32)

        # default
        ag.set_angles(angles)
        ag.set_panel(10)

        ig = ag.get_ImageGeometry()

        scalar = 4

        b = ag.allocate('random', seed=2)
        weight_ls = ig.allocate('random', seed=2)

        A = IdentityOperator(ig)
        b_ig = ig.allocate('random')
        c = numpy.float64(0.3)
        bg = BlockGeometry(ig, ig)

        self.func_geom_test_list = [
            (IndicatorBox(), ag),
            (KullbackLeibler(b=b, backend='numba'), ag),
            (KullbackLeibler(b=b, backend='numpy'), ag),
            (L1Norm(), ag),
            (L1Norm(), ig),
            (L1Norm(b=b), ag),
            (L1Norm(b=b, weight=b), ag),
            (TranslateFunction(L1Norm(), b), ag),
            (TranslateFunction(L2NormSquared(), b), ag),
            (L2NormSquared(), ag),
            (scalar * L2NormSquared(), ag),
            (SumFunction(L2NormSquared(), scalar * L2NormSquared()), ag),
            (SumScalarFunction(L2NormSquared(), 3), ag),
            (ConstantFunction(3), ag),
            (ZeroFunction(), ag),
            (L2NormSquared(b=b), ag),
            (L2NormSquared(), ag),
            (LeastSquares(A, b_ig, c, weight_ls), ig),
            (LeastSquares(A, b_ig, c), ig),
            (WeightedL2NormSquared(weight=b_ig), ig),
            (TotalVariation(backend='c', warm_start=False, max_iteration=100), ig),
            (TotalVariation(backend='numpy', warm_start=False, max_iteration=100), ig),
            (OperatorCompositionFunction(L2NormSquared(), A), ig),
            (MixedL21Norm(), bg),
            (SmoothMixedL21Norm(epsilon=0.3), bg),
            (MixedL11Norm(), bg),
            (BlockFunction(L1Norm(),L2NormSquared()), bg)

        ]

        np.random.seed(5)
        self.data_arrays=[np.random.normal(0,1, (10,10)).astype(np.float32),  np.array(range(0,65500, 655), dtype=np.uint16).reshape((10,10)), np.random.uniform(-0.1,1,(10,10)).astype(np.float32)]

    def get_result(self, function, method, x, *args):
        try:
            input=x.copy() #To check that it isn't changed after function calls
            if method == 'proximal':
                out= function.proximal(x, *args)
            elif method == 'proximal_conjugate':
                out= function.proximal_conjugate(x, *args)
            elif method == 'gradient':
                out= function.gradient(x, *args)
            self.assertDataArraysInContainerAllClose(input, x, rtol=1e-5, msg= "In case func."+method+'(data, *args) where func is  ' + function.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
            return out
        except NotImplementedError:
            return None

    def in_place_test(self,desired_result, function, method,   x, *args, ):
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
                pass


    def out_test(self, desired_result, function, method,  x, *args, ):
        input = x.copy()
        out2=0*(x.copy())
        try:
            if method == 'proximal':
                ret = function.proximal(input, *args, out=out2)
            elif method == 'proximal_conjugate':
                ret = function.proximal_conjugate(input, *args, out=out2)
            elif method == 'gradient':
                ret = function.gradient(input, *args, out=out2)
            self.assertDataArraysInContainerAllClose(desired_result, out2, rtol=1e-5, msg= "Calculation failed using `out` in func."+method+'(x, *args, out=data) where func is  ' + function.__class__.__name__+ '. ')
            self.assertDataArraysInContainerAllClose(input, x,  rtol=1e-5, msg= "In case func."+method+'(data, *args, out=out) where func is  ' + function.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
            self.assertDataArraysInContainerAllClose(desired_result, ret, rtol=1e-5, msg= "Calculation failed returning with `out` in ret = func."+method+'(x, *args, out=data) where func is  ' + function.__class__.__name__+ '. ')
          
        except (InPlaceError, NotImplementedError):
            pass



    def test_proximal_conjugate_out(self):
        for func, geom in self.func_geom_test_list:
            for data_array in self.data_arrays:
                data=geom.allocate(None)
                data.fill(data_array)
                result=self.get_result(func, 'proximal_conjugate', data, 0.5)
                self.out_test(result, func,  'proximal_conjugate',  data, 0.5)
                self.in_place_test(result, func, 'proximal_conjugate',  data, 0.5)

    def test_proximal_out(self):
        for func, geom in self.func_geom_test_list:
            for data_array in self.data_arrays:
                data=geom.allocate(None)
                data.fill(data_array)
                result=self.get_result(func, 'proximal', data, 0.5)
                self.out_test(result, func, 'proximal',  data, 0.5)
                self.in_place_test(result,func,  'proximal',  data, 0.5)

    def test_gradient_out(self):
        for func, geom in self.func_geom_test_list:
                for data_array in self.data_arrays:
                    print(func.__class__.__name__)
                    data=geom.allocate(None)
                    data.fill(data_array)
                    result=self.get_result(func, 'gradient', data)
                    self.out_test(result, func, 'gradient',   data)
                    self.in_place_test(result, func, 'gradient',   data)



class TestOperatorOutAndInPlace(CCPiTestClass):
    def setUp(self):

        ig = ImageGeometry(10,10,channels=3)
        ig_2D=ImageGeometry(10,10)
        vg = VectorGeometry(10)

        mask = ig.allocate(True,dtype=bool)
        amask = mask.as_array()
        amask[2,1:3,:] = False
        amask[0,0,:] = False




        # Parameters for point spread function PSF (size and std)
        ks          = 10
        ksigma      = 5.0

        # Create 1D PSF and 2D as outer product, then normalise.
        w           = numpy.exp(-numpy.arange(-(ks-1)/2,(ks-1)/2+1)**2/(2*ksigma**2))
        w.shape     = (ks,1)
        PSF         = w*numpy.transpose(w)
        PSF         = PSF/(PSF**2).sum()
        PSF         = PSF/PSF.sum()
        PSF         = np.array([PSF]*3)

        np.random.seed(5)

        self.operator_geom_test_list = [
            (MatrixOperator(numpy.random.randn(10, 10)), vg),
            (ZeroOperator(ig), ig),
            (IdentityOperator(ig), ig),
            (3 * IdentityOperator(ig), ig),
            (DiagonalOperator(ig.allocate('random',seed=101)), ig),
            (MaskOperator(mask), ig),
            (ChannelwiseOperator(DiagonalOperator(ig_2D.allocate('random',seed=101)),3), ig),
            (BlurringOperator(PSF,ig), ig),
            (FiniteDifferenceOperator(ig, direction = 0, bnd_cond = 'Neumann') , ig),
            (FiniteDifferenceOperator(ig, direction = 0) , ig)]


        self.data_arrays=[np.random.normal(0,1, (3,10,10)).astype(np.float32),  np.array(range(0,65400, 218), dtype=np.uint16).reshape((3,10,10)), np.random.uniform(-0.1,1,(3,10,10)).astype(np.float32)]
        self.vector_data_arrays=[np.random.normal(0,1, (10)).astype(np.float32),  np.array(range(0,65400, 6540), dtype=np.uint16), np.random.uniform(-0.1,1,(10)).astype(np.float32)]






    def get_result(self, operator, method, x, *args): 
        try:
            input=x.copy() #To check that it isn't changed after function calls
            if method == 'direct':
                out= operator.direct(x, *args)
            elif method == 'adjoint':
                out= operator.adjoint(x, *args)

            self.assertDataArraysInContainerAllClose(input, x, rtol=1e-5, msg= "In case operator."+method+'(data, *args) where operator is  ' + operator.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
            return out
        except NotImplementedError:
            return None

    def in_place_test(self,desired_result, operator, method,   x, *args, ):
            out3 = x.copy()
            try:
                if method == 'direct':
                    operator.direct(out3, *args, out=out3)
                elif method == 'adjoint':
                    operator.adjoint(out3, *args, out=out3)

                self.assertDataArraysInContainerAllClose(desired_result, out3, rtol=1e-5, msg= "In place calculation failed for operator."+method+'(data, *args, out=data) where operator is  ' + operator.__class__.__name__+ '. ')

            except (InPlaceError, NotImplementedError):
                pass


    def out_test(self, desired_result, operator, method,  x, *args, ): 
        input = x.copy()
        out2=0*(x.copy())
        try:
            if method == 'direct':
                ret = operator.direct(input, *args, out=out2)
            elif method == 'adjoint':
                ret = operator.adjoint(input, *args, out=out2)

            self.assertDataArraysInContainerAllClose(desired_result, out2, rtol=1e-5, msg= "Calculation failed using `out` in operator."+method+'(x, *args, out=data) where func is  ' + operator.__class__.__name__+ '. ')
            self.assertDataArraysInContainerAllClose(input, x,  rtol=1e-5, msg= "In case operator."+method+'(data, *args, out=out) where operator is  ' + operator.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
            self.assertDataArraysInContainerAllClose(desired_result, ret, rtol=1e-5, msg= "Calculation failed using return and `out` in ret = operator."+method+'(x, *args, out=data) where func is  ' + operator.__class__.__name__+ '. ')
            
        except (InPlaceError, NotImplementedError):
            pass

    def test_direct_out(self):
        for operator, geom in self.operator_geom_test_list:
            for data_array in self.data_arrays:
                data=geom.allocate(None)
                try:
                    data.fill(data_array)
                except:
                    data.fill(data_array[0,0,:])
                result=self.get_result(operator, 'direct', data)
                self.out_test(result, operator,  'direct',  data)
                self.in_place_test(result, operator, 'direct',  data)

    def test_proximal_out(self):
        for operator, geom in self.operator_geom_test_list:
            for data_array in self.data_arrays:
                data=geom.allocate(None)
                try:
                    data.fill(data_array)
                except:
                    data.fill(data_array[0,0,:])
                result=self.get_result(operator, 'adjoint', data)
                self.out_test(result, operator, 'adjoint',  data)
                self.in_place_test(result,operator,  'adjoint',  data)
