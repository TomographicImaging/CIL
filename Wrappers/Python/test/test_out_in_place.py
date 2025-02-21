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
from cil.utilities import dataexample
from cil.framework import AcquisitionGeometry, ImageGeometry, VectorGeometry, DataContainer
from cil.framework.labels import AcquisitionType

from cil.optimisation.operators import IdentityOperator, WaveletOperator
from cil.optimisation.functions import  KullbackLeibler, ConstantFunction, TranslateFunction, soft_shrinkage, L1Sparsity, BlockFunction
from cil.optimisation.operators import LinearOperator, MatrixOperator  

from cil.optimisation.operators import SumOperator,  ZeroOperator, CompositionOperator, ProjectionMap
from cil.optimisation.operators import BlockOperator,\
    FiniteDifferenceOperator, SymmetrisedGradientOperator,  DiagonalOperator, MaskOperator, ChannelwiseOperator, BlurringOperator

from cil.optimisation.functions import  KullbackLeibler, WeightedL2NormSquared, L2NormSquared, \
    L1Norm, L2NormSquared, MixedL21Norm, LeastSquares, \
    SmoothMixedL21Norm, OperatorCompositionFunction, \
     IndicatorBox, TotalVariation,  SumFunction, SumScalarFunction, \
    WeightedL2NormSquared, MixedL11Norm, ZeroFunction

from cil.processors import AbsorptionTransmissionConverter, Binner, CentreOfRotationCorrector, MaskGenerator, Masker, Normaliser, Padder, \
RingRemover, Slicer, TransmissionAbsorptionConverter, PaganinProcessor, FluxNormaliser

import numpy
from utils import has_tigre, has_nvidia


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
        # [(function, geometry, test_proximal, test_proximal_conjugate, test_gradient), ...]
        self.func_geom_test_list = [
            (IndicatorBox(), ag, True, True, False),
            (KullbackLeibler(b=b, backend='numba'), ag, True, True, True),
            (KullbackLeibler(b=b, backend='numpy'), ag, True, True, True),
            (L1Norm(), ag, True, True, False),
            (L1Norm(), ig, True, True, False),
            (L1Norm(b=b), ag, True, True, False),
            (L1Norm(b=b, weight=b), ag, True, True, False),
            (TranslateFunction(L1Norm(), b), ag, True, True, False),
            (TranslateFunction(L2NormSquared(), b), ag, True, True, True),
            (L2NormSquared(), ag, True, True, True),
            (scalar * L2NormSquared(), ag, True, True, True),
            (SumFunction(L2NormSquared(), scalar * L2NormSquared()), ag, False, False, True),
            (SumScalarFunction(L2NormSquared(), 3), ag, True, True, True),
            (ConstantFunction(3), ag, True, True, True),
            (ZeroFunction(), ag, True, True, True),
            (L2NormSquared(b=b), ag, True, True, True),
            (L2NormSquared(), ag, True, True, True),
            (LeastSquares(A, b_ig, c, weight_ls), ig, False, False, True),
            (LeastSquares(A, b_ig, c), ig, False, False, True),
            (WeightedL2NormSquared(weight=b_ig), ig, True, True, True),
            (TotalVariation(backend='c', warm_start=False, max_iteration=100), ig, True, True, False),
            (TotalVariation(backend='numpy', warm_start=False, max_iteration=100), ig, True, True, False),
            (OperatorCompositionFunction(L2NormSquared(), A), ig, False, False, True),
            (MixedL21Norm(), bg, True, True, False),
            (SmoothMixedL21Norm(epsilon=0.3), bg, False, False, True),
            (MixedL11Norm(), bg, True, True, False),
            (BlockFunction(L1Norm(),L2NormSquared()), bg, True, True, False),
            (BlockFunction(L2NormSquared(),L2NormSquared()), bg, True, True, True),
            (L1Sparsity(WaveletOperator(ig)), ig, True, True, False)


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
            raise NotImplementedError(function.__class__.__name__+" raises a NotImplementedError for "+method)


    def in_place_test(self,desired_result, function, method,   x, *args, ):
            out3 = x.copy()
            try:
                try:
                    if method == 'proximal':
                        function.proximal(out3, *args, out=out3)
                    elif method == 'proximal_conjugate':
                        function.proximal_conjugate(out3, *args, out=out3)
                    elif method == 'gradient':
                        function.gradient(out3, *args, out=out3)
                    self.assertDataArraysInContainerAllClose(desired_result, out3, rtol=1e-5, msg= "In place calculation failed for func."+method+'(data, *args, out=data) where func is  ' + function.__class__.__name__+ '. ')

                except InPlaceError:
                    pass
            except NotImplementedError:
                raise NotImplementedError(function.__class__.__name__+" raises a NotImplementedError for "+method)

    def out_test(self, desired_result, function, method,  x, *args, ):
        input = x.copy()
        out2=0*(x.copy())
        try:
            try:
                if method == 'proximal':
                    ret = function.proximal(input, *args, out=out2)
                elif method == 'proximal_conjugate':
                    ret = function.proximal_conjugate(input, *args, out=out2)
                elif method == 'gradient':
                    ret = function.gradient(input, *args, out=out2)
                self.assertDataArraysInContainerAllClose(desired_result, out2, rtol=1e-5, msg= "Calculation failed using `out` in func."+method+'(x, *args, out=data) where func is  ' + function.__class__.__name__+ '. ')
                self.assertDataArraysInContainerAllClose(input, x,  rtol=1e-5, msg= "In case func."+method+'(data, *args, out=out) where func is  ' + function.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
                self.assertDataArraysInContainerAllClose(desired_result, ret, rtol=1e-5, msg= f"Calculation failed returning with `out` in ret = func.{method}(x, *args, out=data) where func is {function.__class__.__name__}")
            
            except InPlaceError:
                pass
        except NotImplementedError:
            raise NotImplementedError(function.__class__.__name__+" raises a NotImplementedError for "+method)



    def test_proximal_conjugate_out(self):
        for func, geom, _, test_proximal_conj, _ in self.func_geom_test_list:
            if test_proximal_conj:
                for data_array in self.data_arrays:
                    data=geom.allocate(None)
                    data.fill(data_array)
                    result=self.get_result(func, 'proximal_conjugate', data, 0.5)
                    self.out_test(result, func,  'proximal_conjugate',  data, 0.5)
                    self.in_place_test(result, func, 'proximal_conjugate',  data, 0.5)

    def test_proximal_out(self):
        for func, geom, test_proximal, _, _  in self.func_geom_test_list:
            if test_proximal:
                for data_array in self.data_arrays:
                    data=geom.allocate(None)
                    data.fill(data_array)
                    result=self.get_result(func, 'proximal', data, 0.5)
                    self.out_test(result, func, 'proximal',  data, 0.5)
                    self.in_place_test(result,func,  'proximal',  data, 0.5)

    def test_gradient_out(self):
        for func, geom, _, _, test_gradient in self.func_geom_test_list:
            if test_gradient:
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
            raise NotImplementedError(operator.__class__.__name__+" raises a NotImplementedError for "+method)

    def in_place_test(self,desired_result, operator, method,   x, *args, ):
            out3 = x.copy()
            try:
                try:
                    if method == 'direct':
                        operator.direct(out3, *args, out=out3)
                    elif method == 'adjoint':
                        operator.adjoint(out3, *args, out=out3)

                    self.assertDataArraysInContainerAllClose(desired_result, out3, rtol=1e-5, msg= "In place calculation failed for operator."+method+'(data, *args, out=data) where operator is  ' + operator.__class__.__name__+ '. ')

                except InPlaceError:
                    pass
            except NotImplementedError: 
                raise NotImplementedError(operator.__class__.__name__+" raises a NotImplementedError for "+method)


    def out_test(self, desired_result, operator, method,  x, *args):
        input = x.copy()
        out2=0*(x.copy())
        try:
            if method == 'direct':
                ret = operator.direct(input, *args, out=out2)
            elif method == 'adjoint':
                ret = operator.adjoint(input, *args, out=out2)

            self.assertDataArraysInContainerAllClose(desired_result, out2, rtol=1e-5, msg= "Calculation failed using `out` in operator."+method+'(x, *args, out=data) where func is  ' + operator.__class__.__name__+ '. ')
            self.assertDataArraysInContainerAllClose(input, x,  rtol=1e-5, msg= "In case operator."+method+'(data, *args, out=out) where operator is  ' + operator.__class__.__name__+ 'the input data has been incorrectly affected by the calculation. ')
            self.assertDataArraysInContainerAllClose(desired_result, ret, rtol=1e-5, msg= f"Calculation failed using return and `out` in ret = operator.{method}(x, *args, out=data) where func is {operator.__class__.__name__}")
            
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

class TestProcessorOutandInPlace(CCPiTestClass):
    def setUp(self):
        
        self.data_arrays=[np.random.normal(0,1, (10,20)).astype(np.float32),  
                          np.array(range(0,65500, 328), dtype=np.uint16).reshape((10,20)),
                          np.random.uniform(0,1,(10,20)).astype(np.float64)]

        ag_parallel_2D = AcquisitionGeometry.create_Parallel2D(detector_position=[0,1], units='m')
        angles = np.linspace(0, 360, 10, dtype=np.float32)
        ag_parallel_2D.set_angles(angles)
        ag_parallel_2D.set_panel(20)

        ag_parallel_3D = AcquisitionGeometry.create_Parallel3D(detector_position=[0,1,0], units='mm')
        # ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0, 1, 0], units='mm').set_panel([20,2], pixel_size=0.1).set_angles(angles)
        ag_parallel_3D.set_angles(angles)
        ag_parallel_3D.set_panel([20,2], pixel_size=0.1)

        ag_cone_2D = AcquisitionGeometry.create_Cone2D(source_position=[0,-10],detector_position=[0,10], units='mm')
        ag_cone_2D.set_angles(angles)
        ag_cone_2D.set_panel(20)

        ag_cone_3D = AcquisitionGeometry.create_Cone3D(source_position=[0,-10,0],detector_position=[0,10,0], units='mm')
        ag_cone_3D.set_angles(angles)
        ag_cone_3D.set_panel([20,2], pixel_size=0.1)

        ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0, 150, 0], units='mm').set_panel([20,2], pixel_size=0.1).set_angles(angles)
        
        self.geometry_test_list = [ag_parallel_2D, ag_parallel_3D, ag_cone_2D, ag_cone_3D]


    def fail(self, error_message):
        raise NotImplementedError(error_message)
    
    def out_check(self, processor, data, data_array_index, *args):
        """
        - Test to check the processor gives the same result using the out argument
        - Check the out argument doesn't change the orginal data
        - Check the processor still returns correctly with the out argument
        - Check the processor returns an error if the dtype of out is different to expected
        """
        data_copy=data.copy() #To check that data isn't changed after function calls
        try:
            processor.set_input(data_copy)
            out1 = processor.get_output()
            out2 = out1.copy()
            out2.fill(0)
            out3 = processor.get_output(out=out2)

            # check the processor gives the same result using the out argument
            self.assertDataContainerAllClose(out1, out2, rtol=1e-10, strict=True)
            # check the out argument doesn't change the orginal data
            self.assertDataContainerAllClose(data_copy, data, rtol=1e-10, strict=True)
            # check the processor still returns correctly with the out argument
            self.assertEqual(id(out2), id(out3))
            # check the processor returns an error if the dtype of out is different to expected
            out_wrong_type = out1.copy()
            out_wrong_type.array = numpy.array(out_wrong_type.array, dtype=np.int64)
            with self.assertRaises(TypeError):
                processor.get_output(out=out_wrong_type)

        except NotImplementedError:
            self.fail("out_test test not implemented for  " + processor.__class__.__name__)
        except Exception as e:
            error_message = '\nFor processor: ' + processor.__class__.__name__ + \
            '\nOn data_array index: ' + str(data_array_index) +\
            '\nFor geometry type: \n' + str(data.geometry) 
            raise type(e)(error_message + '\n\n' + str(e))

    def in_place_check(self, processor, data, data_array_index, *args):
        """
        Check the processor gives the correct result if the calculation is performed in place
        i.e. data is passed to the out argument
        if the processor output is a different sized array, the calcualtion
        cannot be perfomed in place, so we expect this to raise a ValueError
        """
        data_copy = data.copy()
        try:
            processor.set_input(data_copy)
            out1 = processor.get_output()

            # Check an error is raised for Processor's that change the dimensions of the data
            if out1.shape != data.shape:
                with self.assertRaises(ValueError):
                    processor.get_output(out=data)
            # Check an error is raised if the Processor returns a different data type (only MaskGenerator at the moment)
            elif isinstance(processor, MaskGenerator):
                    with self.assertRaises(TypeError):
                        processor.get_output(out=data)
            # Otherwise check Processor's work correctly in place
            else:
                processor.get_output(out=data_copy)
                self.assertDataContainerAllClose(out1, data_copy, rtol=1e-10, strict=True)

        except (InPlaceError, NotImplementedError):
            self.fail("in_place_test test not implemented for  " + processor.__class__.__name__)
        except Exception as e:
            error_message = '\nFor processor: ' + processor.__class__.__name__ + \
            '\nOn data_array index: ' + str(data_array_index) +\
            '\nFor geometry type: \n' + str(data.geometry) 
            raise type(e)(error_message + '\n\n' + str(e))

    def test_out(self):
        """
        Tests to check output from Processors, including:
        - Checks that Processors give the same results when the result is obtained
        directly or using the out argument, or in place
        - Checks that the out argument doesn't change the original data
        - Checks that Processors still return the output even when out is used
        All the tests are performed on different data types, geometry and Processors
        Any new Processors should be added to the processor_list
        """

        # perform the tests on different data and geometry types
        for geom in self.geometry_test_list:
            for i, data_array in enumerate(self.data_arrays):
                data=geom.allocate(None)
                if AcquisitionType.DIM2 & geom.dimension:
                    data.fill(data_array)
                else:
                    data.fill(np.repeat(data_array[:,None, :], repeats=2, axis=1))
                # Add new Processors here 
                processor_list = [
                    TransmissionAbsorptionConverter(min_intensity=0.01),
                    AbsorptionTransmissionConverter(),
                    RingRemover(info=False),
                    Slicer(roi={'horizontal':(None,None,None),'angle':(None,None,None)}), 
                    Slicer(roi={'horizontal':(1,3,2),'angle':(None,4,2)}), 
                    Binner(roi={'horizontal':(None,None,None),'angle':(None,None,None)}),
                    Binner(roi={'horizontal':(1,None,2),'angle':(None,4,2)}),
                    Padder(pad_width=0),
                    Padder(pad_width=1),
                    Normaliser(flat_field=data.get_slice(angle=0).as_array()*1, dark_field=data.get_slice(angle=0).as_array()*1e-5),
                    MaskGenerator.median(threshold_factor=3, window=7),
                    Masker.median(mask=data.copy()),
                    PaganinProcessor(),
                    FluxNormaliser(flux=1)
                    ]
                
                for processor in processor_list:
                    self.out_check(processor, data, i)
                    self.in_place_check(processor, data, i)
   
    def test_centre_of_rotation_xcorrelation_out(self):
        """
        Test the output of the centre of rotation xcorrelation processor 
        These tests are performed separately because the processor returns 
        changes to the geometry rather than the data
        """
        # CentreOfRotationCorrector.xcorrelation() works on parallel beam data only
        for geom in self.geometry_test_list[0:2]:
            for data_array in self.data_arrays:
                data=geom.allocate(None)
                try:
                    data.fill(data_array)
                except:
                    data.fill(np.repeat(data_array[:,None, :], repeats=2, axis=1))
                processor = CentreOfRotationCorrector.xcorrelation(ang_tol=180)
                processor.set_input(data)
                out1 = processor.get_output()
                # out_test fails because the processor only updates geometry, this is expected behaviour
                # self.out_test(result, processor, data)
                # test geometry instead
                input = data.copy()
                out2 = 0*(out1.copy())
                try:
                    processor.set_input(input)
                    out3 = processor.get_output(out=out2)
                    
                    numpy.testing.assert_allclose(out1.geometry.config.system.rotation_axis.position, out2.geometry.config.system.rotation_axis.position, err_msg= "Calculation failed using processor.set_input(data), processor.get_output(out=out) where func is  " + processor.__class__.__name__+ ".")
                    numpy.testing.assert_allclose(input.geometry.config.system.rotation_axis.position, data.geometry.config.system.rotation_axis.position,  err_msg= "In case processor.set_input(data), processor.get_output(out=data) where processor is  " + processor.__class__.__name__+ " the input data has been incorrectly affected by the calculation. ")
                    self.assertDataArraysInContainerAllClose(out2, out3,  rtol=1e-5, msg= "In case processor.set_input(data), output=processor.get_output(out=data) where processor is  " + processor.__class__.__name__+ " the processor incorrectly supresses the output. ")

                    processor.set_input(data)
                    processor.get_output(out=data)
                    numpy.testing.assert_array_equal(out1.geometry.config.system.rotation_axis.position, data.geometry.config.system.rotation_axis.position, err_msg= "Calculation failed using processor.set_input(data), processor.get_output(out=data) where func is  " + processor.__class__.__name__+ ".")
                    
                except (InPlaceError, NotImplementedError):
                    print("out_test_for_geometry test not implemented for  " + processor.__class__.__name__)
                    pass

                try:
                    if out1.shape != data.shape:
                        with self.assertRaises(ValueError):
                            processor.get_output(out=data)
                    else:
                        processor.get_output(out=data)
                        self.assertDataArraysInContainerAllClose(out1, data, rtol=1e-5, msg= "In place calculation failed for processor.set_input(data), processor.get_output(out=data) where processor is  " + processor.__class__.__name__+ "." )
                except (InPlaceError, NotImplementedError):
                    print("in_place_test test not implemented for  " + processor.__class__.__name__)
    
    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_centre_of_rotation_image_sharpness_out(self):
        """
        Test the output of the centre of rotation image_sharpness processor 
        These tests are performed separately because the processor returns 
        changes to the geometry rather than the data
        """
        # Here we need to use real data rather than random data otherwise the processor fails
        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        
        processor = Binner(roi={'horizontal':(0,-1,2),'vertical':(0,-1,2), 'angle':(0,-1,2)})
        processor.set_input(data)
        data = processor.get_output()

        processor = CentreOfRotationCorrector.image_sharpness()
        processor.set_input(data)
        out1 = processor.get_output()
        # out_test fails because the processor only updates geometry, this is expected behaviour
        # self.out_test(result, processor, data)
        # test geometry instead
        input = data.copy()
        out2 = 0*(out1.copy())
        try:
            processor.set_input(input)
            out3 = processor.get_output(out=out2)
            
            numpy.testing.assert_allclose(out1.geometry.config.system.rotation_axis.position, out2.geometry.config.system.rotation_axis.position, err_msg= "Calculation failed using processor.set_input(data), processor.get_output(out=out) where func is  " + processor.__class__.__name__+ ".")
            numpy.testing.assert_allclose(input.geometry.config.system.rotation_axis.position, data.geometry.config.system.rotation_axis.position,  err_msg= "In case processor.set_input(data), processor.get_output(out=data) where processor is  " + processor.__class__.__name__+ " the input data has been incorrectly affected by the calculation. ")
            self.assertDataArraysInContainerAllClose(out2, out3,  rtol=1e-5, msg= "In case processor.set_input(data), output=processor.get_output(out=data) where processor is  " + processor.__class__.__name__+ " the processor incorrectly supresses the output. ")

            processor.set_input(data)
            processor.get_output(out=data)
            numpy.testing.assert_array_equal(out1.geometry.config.system.rotation_axis.position, data.geometry.config.system.rotation_axis.position, err_msg= "Calculation failed using processor.set_input(data), processor.get_output(out=data) where func is  " + processor.__class__.__name__+ ".")
            
        except (InPlaceError, NotImplementedError):
            self.fail("out_test_for_geometry test not implemented for  " + processor.__class__.__name__)
            pass

        try:
            if out1.shape != data.shape:
                with self.assertRaises(ValueError):
                    processor.get_output(out=data)
            else:
                processor.get_output(out=data)
                self.assertDataArraysInContainerAllClose(out1, data, rtol=1e-5, msg= "In place calculation failed for processor.set_input(data), processor.get_output(out=data) where processor is  " + processor.__class__.__name__+ "." )
        except (InPlaceError, NotImplementedError):
            self.fail("in_place_test test not implemented for  " + processor.__class__.__name__)