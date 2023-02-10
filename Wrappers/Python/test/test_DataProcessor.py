# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
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

import unittest
import numpy
from cil.framework import DataContainer
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.utilities import dataexample
from timeit import default_timer as timer

from cil.framework import AX, CastDataContainer, PixelByPixelDataProcessor
from cil.recon import FBP

from cil.processors import CentreOfRotationCorrector
from cil.processors import TransmissionAbsorptionConverter, AbsorptionTransmissionConverter
from cil.processors import Slicer, Binner, MaskGenerator, Masker, Padder
import gc

from utils import has_astra, has_tigre, has_nvidia, has_tomophantom, initialise_tests, has_ipp

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator as AstraProjectionOperator

if has_tigre:
    from cil.plugins.tigre import ProjectionOperator as TigreProjectionOperator

if has_tomophantom:
    from cil.plugins import TomoPhantom

if has_ipp:
    from cil.processors.cilacc_binner import Binner_IPP

class TestPadder(unittest.TestCase):
    def setUp(self):
        ray_direction = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]

        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    rotation_axis_position=rotation_axis_position)

        # test int shortcut
        self.num_angles = 10
        angles = numpy.linspace(0, 360, self.num_angles, dtype=numpy.float32)

        self.num_channels = 10
        self.num_pixels = 5
        AG.set_channels(num_channels=self.num_channels)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel(self.num_pixels, pixel_size=0.1)

        data = AG.allocate('random')
        
        self.data = data
        self.AG = AG

    def test_constant_with_int(self):
        data = self.data
        AG = self.AG
        num_pad = 5
        b = Padder.constant(pad_width=num_pad)
        b.set_input(data)
        data_padded = b.process()

        data_new = numpy.zeros((self.num_channels + 2*num_pad, self.num_angles, self.num_pixels + 2*num_pad), dtype=numpy.float32)
        data_new[5:15,:,5:10] = data.as_array()
        # new_angles = numpy.zeros((20,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=20)
        geometry_padded.set_panel(15, pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_constant_with_tuple(self):
        data = self.data
        AG = self.AG
        # test tuple
        pad_tuple = (5,2)
        b = Padder.constant(pad_width=pad_tuple)
        b.set_input(data)
        data_padded = b.process()

        data_new = numpy.zeros((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels + sum(pad_tuple)),\
                     dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],\
            :,\
                pad_tuple[0]:-pad_tuple[1]] = data.as_array()
        # new_angles = numpy.zeros((17,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=self.num_channels+sum(pad_tuple))
        geometry_padded.set_panel(self.num_pixels+sum(pad_tuple), pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_constant_with_dictionary1(self):
        data = self.data
        AG = self.AG
        # test dictionary + constant values
        pad_tuple = (5,2)
        const = 5.0
        b = Padder.constant(pad_width={'channel':pad_tuple}, constant_values=const)
        b.set_input(data)
        data_padded = b.process()

        data_new = const * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],:,:] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)

        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_constant_with_dictionary2(self):
        data = self.data
        AG = self.AG
        # test dictionary + constant values
        pad_tuple1 = (5,2)
        pad_tuple2 = (2,5)
        const = 5.0
        b = Padder.constant(pad_width={'channel':pad_tuple1, 'horizontal':pad_tuple2}, constant_values=const)
        b.set_input(data)
        data_padded = b.process()

        data_new = const * numpy.ones((self.num_channels + sum(pad_tuple1), self.num_angles, \
            self.num_pixels + sum(pad_tuple2)),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple1[0]:-pad_tuple1[1],:,pad_tuple2[0]:-pad_tuple2[1]] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)
        geometry_padded.set_panel(self.num_pixels + sum(pad_tuple2),pixel_size=0.1)
        
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_edge_with_int(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        
        num_pad = 5
        b = Padder.edge(pad_width=num_pad)
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + 2*num_pad, self.num_angles, self.num_pixels + 2*num_pad), dtype=numpy.float32)
        data_new[5:15,:,5:10] = data.as_array()

        # new_angles = numpy.zeros((20,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=20)
        geometry_padded.set_panel(15, pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_edge_with_tuple(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test tuple
        pad_tuple = (5,2)
        b = Padder.edge(pad_width=pad_tuple)
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels + sum(pad_tuple)),\
                     dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],\
            :,\
                pad_tuple[0]:-pad_tuple[1]] = data.as_array()
        # new_angles = numpy.zeros((17,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=self.num_channels+sum(pad_tuple))
        geometry_padded.set_panel(self.num_pixels+sum(pad_tuple), pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_edge_with_dictionary(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test dictionary + constant values
        pad_tuple = (5,2)
        b = Padder.edge(pad_width={'channel':pad_tuple})
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],:,:] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)

        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_linear_ramp_with_int(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        
        num_pad = 5
        b = Padder.linear_ramp(pad_width=num_pad)
        b.set_input(data)
        data_padded = b.process()

        data_new = 0 * numpy.ones((self.num_channels + 2*num_pad, self.num_angles, self.num_pixels + 2*num_pad), dtype=numpy.float32)
        data_new[5:15,:,5:10] = data.as_array()

        # new_angles = numpy.zeros((20,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=20)
        geometry_padded.set_panel(15, pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        # numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
        self.assertAlmostEqual(data_padded.as_array().ravel()[0] , 0.)
        self.assertAlmostEqual(data_padded.as_array().ravel()[-1] , 0.)
    
    def test_linear_ramp_with_tuple(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test tuple
        pad_tuple = (5,2)
        b = Padder.edge(pad_width=pad_tuple)
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels + sum(pad_tuple)),\
                     dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],\
            :,\
                pad_tuple[0]:-pad_tuple[1]] = data.as_array()
        # new_angles = numpy.zeros((17,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=self.num_channels+sum(pad_tuple))
        geometry_padded.set_panel(self.num_pixels+sum(pad_tuple), pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_linear_ramp_with_dictionary(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test dictionary + constant values
        pad_tuple = (5,2)
        b = Padder.edge(pad_width={'channel':pad_tuple})
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],:,:] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)

        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)


class TestBinner_cillacc(unittest.TestCase):

    @unittest.skipUnless(has_ipp, "Requires IPP libraries")
    def test_binning_cpp(self):

        shape_in = [4,12,16,32]
        shape_out = [1,2,3,4]
        start_index = [1,2,1,3]
        binning = [3,4,5,6]

        binner_cpp = Binner_IPP(shape_in,shape_out,start_index,binning)

        # check clean up
        del binner_cpp
        gc.collect()

        shape_in = [1,2,2,2]
        shape_out = [1,1,1,1]
        start_index = [0,0,0,0]
        binning = [2,2,2,2]

        with self.assertRaises(ValueError):
            binner_cpp = Binner_IPP(shape_in,shape_out,start_index,binning)


    @unittest.skipUnless(has_ipp, "Requires IPP libraries")
    def test_binning_cpp_2D_data(self):

        data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        shape_in = [1] + list(data.shape)
        shape_out = [1,100,32,17]
        start_index = [0,10,5,3]
        binning = [1,1,3,4]

        binned_arr = numpy.empty(shape_out,dtype=numpy.float32)
        binned_by_hand = numpy.empty(shape_out,dtype=numpy.float32)


        binner_cpp = Binner_IPP(shape_in,shape_out,start_index,binning)
        binner_cpp.bin(data.array, binned_arr)

        k_out=0
        for k in range(start_index[1],start_index[1]+shape_out[1]):
            j_out=0
            for j in range(start_index[2],start_index[2]+shape_out[2]*binning[2], binning[2]):
                i_out = 0
                for i in range(start_index[3],start_index[3]+shape_out[3]*binning[3], binning[3]):
                    binned_by_hand[0,k_out,j_out,i_out]  = data.array[k,j:j+binning[2],i:i+binning[3]].mean()
                    i_out +=1
                j_out +=1
            k_out +=1

        numpy.testing.assert_allclose(binned_by_hand,binned_arr,atol=1e-6)


    @unittest.skipUnless(has_ipp, "Requires IPP libraries")
    def test_binning_cpp_4D(self):

        shape_in = [9,21,40,92]
        shape_out = [2,4,8,16]
        start_index = [3,4,5,6]
        binning = [2,3,4,5]

        data = numpy.random.rand(*shape_in).astype(numpy.float32)

        binned_arr =numpy.zeros(shape_out,dtype=numpy.float32)
        binned_by_hand =numpy.empty(shape_out,dtype=numpy.float32)

        binner_cpp = Binner_IPP(shape_in,shape_out,start_index,binning)
        binner_cpp.bin(data, binned_arr)

        l_out=0
        for l in range(start_index[0],start_index[0]+shape_out[0]*binning[0], binning[0]):
            k_out=0
            for k in range(start_index[1],start_index[1]+shape_out[1]*binning[1], binning[1]):
                j_out=0
                for j in range(start_index[2],start_index[2]+shape_out[2]*binning[2], binning[2]):
                    i_out = 0
                    for i in range(start_index[3],start_index[3]+shape_out[3]*binning[3], binning[3]):
                        binned_by_hand[l_out,k_out,j_out,i_out]  = data[l:l+binning[0],k:k+binning[1],j:j+binning[2],i:i+binning[3]].mean()
                        i_out +=1
                    j_out +=1
                k_out +=1
            l_out +=1

        numpy.testing.assert_allclose(binned_by_hand,binned_arr,atol=1e-6)


    @unittest.skipUnless(has_ipp, "Requires IPP libraries")
    def test_binning_cpp_2D(self):
        
        shape_in = [1,1,3,3]
        shape_out = [1,1,1,1]
        start_index = [0,0,0,0]
        binning = [1,1,2,2]

        data = numpy.random.rand(*shape_in).astype(numpy.float32)

        binned_arr =numpy.zeros(shape_out,dtype=numpy.float32)
        binned_by_hand =numpy.empty(shape_out,dtype=numpy.float32)

        binner_cpp = Binner_IPP(shape_in,shape_out,start_index,binning)
        binner_cpp.bin(data, binned_arr)

        l_out=0
        for l in range(start_index[0],start_index[0]+shape_out[0]*binning[0], binning[0]):
            k_out=0
            for k in range(start_index[1],start_index[1]+shape_out[1]*binning[1], binning[1]):
                j_out=0
                for j in range(start_index[2],start_index[2]+shape_out[2]*binning[2], binning[2]):
                    i_out = 0
                    for i in range(start_index[3],start_index[3]+shape_out[3]*binning[3], binning[3]):
                        binned_by_hand[l_out,k_out,j_out,i_out]  = data[l:l+binning[0],k:k+binning[1],j:j+binning[2],i:i+binning[3]].mean()
                        i_out +=1
                    j_out +=1
                k_out +=1
            l_out +=1

        numpy.testing.assert_allclose(binned_by_hand,binned_arr,atol=1e-6)


class TestBinner(unittest.TestCase):

    def test_set_up_processor(self):

        ig = ImageGeometry(20,22,23,0.1,0.2,0.3,0.4,0.5,0.6,channels=24)
        data = ig.allocate('random')

        channel = range(0,10,3)
        vertical = range(0,8,2)
        horizontal_y = range(0,22,1)
        horizontal_x = range(0,4,4)

        roi = {'horizontal_y':horizontal_y,'horizontal_x':horizontal_x,'vertical':vertical,'channel':channel}
        proc = Binner(roi,accelerated=True)
        proc._set_up_processor(data)

        # check set values
        self.assertTrue(proc._shape_in == list(data.shape))

        shape_out =[(channel.stop - channel.start)//channel.step,
        (vertical.stop - vertical.start)//vertical.step,
        (horizontal_y.stop - horizontal_y.start)//horizontal_y.step,
        (horizontal_x.stop - horizontal_x.start)//horizontal_x.step
        ]

        self.assertTrue(proc._shape_out == shape_out)
        self.assertTrue(proc._labels_in == ['channel','vertical','horizontal_y','horizontal_x'])
        numpy.testing.assert_array_equal(proc._processed_dims,[True,True,False,True])

        roi_ordered = [
            range(channel.start, shape_out[0] * channel.step, channel.step),
            range(vertical.start, shape_out[1] * vertical.step, vertical.step),
            range(horizontal_y.start, shape_out[2] * horizontal_y.step, horizontal_y.step),
            range(horizontal_x.start, shape_out[3] * horizontal_x.step, horizontal_x.step)
        ]

        self.assertTrue(proc._roi_ordered == roi_ordered)


    def test_process_acquisition_geometry_parallel2D(self):

        ag = AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,0.1).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'horizontal':(None,None,None)},

                # bin all
                {'channel':(None,None,4),'angle':(None,None,2),'horizontal':(None,None,16)},
        ]

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0.5,360.5,180,endpoint=False)).set_panel(8,[1.6,0.1]).set_channels(1),
        ]

        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Binning acquisition geometry with roi {}".format(i))

    def test_process_acquisition_geometry_parallel3D(self):

        ag = AcquisitionGeometry.create_Parallel3D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel([128,64],[0.1,0.2]).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # bin all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},
                
                # bin to single dimension
                {'vertical':(31,33,2)},
        ]


        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel3D().set_angles(numpy.linspace(0.5,360.5,180,endpoint=False)).set_panel([8,8],[1.6,1.6]).set_channels(1),
                AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,[0.1,0.4]).set_channels(4),        
        ]

        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Binning acquisition geometry with roi {}".format(i))


    def test_process_acquisition_geometry_cone2D(self):

        ag = AcquisitionGeometry.create_Cone2D([0,-50],[0,50]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,0.1).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'horizontal':(None,None,None)},

                # bin all
                {'channel':(None,None,4),'angle':(None,None,2),'horizontal':(None,None,16)},
        ]

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Cone2D([0,-50],[0,50]).set_angles(numpy.linspace(0.5,360.5,180,endpoint=False)).set_panel(8,[1.6,0.1]).set_channels(1),
        ]

        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Binning acquisition geometry with roi {}".format(i))


    def test_process_acquisition_geometry_cone3D(self):

        ag = AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,0]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel([128,64],[0.1,0.2]).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # bin all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},

                # shift detector with crop
                {'vertical':(32,65,2)},
                
                # bin to single dimension
                {'vertical':(31,33,2)},

        ]

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,0]).set_angles(numpy.linspace(0.5,360.5,180,endpoint=False)).set_panel([8,8],[1.6,1.6]).set_channels(1),
                AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,32*0.2/2]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel([128,16],[0.1,0.4]).set_channels(4),
                AcquisitionGeometry.create_Cone2D([0,-50],[0,50]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,[0.1,0.4]).set_channels(4),        
        ]

        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Binning acquisition geometry with roi {}".format(i))


    def test_process_image_geometry(self):

        ig_in = ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4)

        rois = [
                # same as input
                {'channel':(None,None,None),'vertical':(None,None,None),'horizontal_x':(None,None,None),'horizontal_y':(None,None,None)},

                # bin all
                {'channel':(None,None,3),'vertical':(None,None,7),'horizontal_x':(None,None,4),'horizontal_y':(None,None,5)},

                # crop and bin
                {'channel':(1,None,2),'vertical':(4,-8,4),'horizontal_x':(1,7,2),'horizontal_y':(4,-8,2)},
                
                # bin to single dimension
                {'channel':(None,None,4),'vertical':(None,None,28),'horizontal_x':(None,None,8),'horizontal_y':(None,None,16)},
        ]

        ig_gold = [ ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4),
                    ImageGeometry(2,3,4,0.4,1.0,2.1,center_y=-0.1,channels=1),
                    ImageGeometry(3,2,4,0.2,0.4,1.2,center_y=-0.4, center_z=-0.6, channels=1),
                    ImageGeometry(1,1,1,0.8,3.2,8.4,channels=1)
        ]

        #channel spacing isn't an initialisation argument
        ig_gold[1].channel_spacing=3
        ig_gold[2].channel_spacing=2
        ig_gold[3].channel_spacing=4


        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ig_in)
            ig_out = proc._process_image_geometry()
            self.assertEqual(ig_gold[i], ig_out, msg="Binning image geometry with roi {} failed".format(i))

        with self.assertRaises(ValueError):
            roi = {'wrong label':(None,None,None)}
            proc = Binner(roi=roi)
            proc.set_input(ig_in)
            ig_out = proc._process_image_geometry(ig_in)


        # binning/cropping offsets geometry
        ig_in = ImageGeometry(128,128,128,16,16,16,80,240,-160)
        ig_gold = ImageGeometry(10,10,10,48,48,48,-192,-32,-432)

        roi = {'vertical':(32,64,3),'horizontal_x':(32,64,3),'horizontal_y':(32,64,3)}
        proc = Binner(roi,accelerated=True)
        proc.set_input(ig_in)
        ig_out = proc.get_output()

        self.assertEqual(ig_gold, ig_out, msg="Binning image geometry with offset roi failed")


    def test_bin_array_consistency(self):

        ig = ImageGeometry(64,32,16,channels=8)
        data = ig.allocate('random')

        roi = {'horizontal_x':(1,-1,16),'horizontal_y':(1,-1,8),'channel':(1,-1,2),'vertical':(1,-1,4)}

        binner = Binner(roi)
        binner.set_input(data)

        shape_binned = binner._shape_out

        binned_arr_acc = numpy.empty(shape_binned,dtype=numpy.float32)
        binned_arr_numpy = numpy.empty(shape_binned,dtype=numpy.float32)
        binned_by_hand = numpy.empty(shape_binned,dtype=numpy.float32)

        binner._bin_array_numpy(data.array, binned_arr_numpy)
        binner._bin_array_acc(data.array, binned_arr_acc)


        l_out = 0
        for l in range(1,shape_binned[0]*2, 2):
            k_out = 0
            for k in range(1,shape_binned[1]*4, 4):
                j_out = 0
                for j in range(1,shape_binned[2]*8, 8):
                    i_out = 0
                    for i in range(1, shape_binned[3]*16, 16):
                        binned_by_hand[l_out,k_out,j_out,i_out]  = data.array[l:l+2,k:k+4,j:j+8,i:i+16].mean()
                        i_out +=1
                    j_out +=1
                k_out +=1
            l_out +=1

        numpy.testing.assert_allclose(binned_by_hand,binned_arr_numpy,atol=1e-6)
        numpy.testing.assert_allclose(binned_by_hand,binned_arr_acc,atol=1e-6)

    def test_bin_image_data(self):
        """
        Binning results tested with test_binning_cpp_ so this is checking wrappers with axis labels and geometry
        """
        ig = ImageGeometry(4,6,8,0.1,0.2,0.3,0.4,0.5,0.6,channels=10)
        data = ig.allocate('random')

        channel = range(0,10,2)
        vertical = range(0,8,2)
        horizontal_y = range(0,6,2)
        horizontal_x = range(0,4,2)

        roi = {'channel':channel,'vertical':vertical,'horizontal_x':horizontal_x,'horizontal_y':horizontal_y}
        proc = Binner(roi,accelerated=True)
        proc.set_input(data)
        binned_data = proc.get_output()

        ig_out = ImageGeometry(2,3,4,0.2,0.4,0.6,0.4,0.5,0.6,channels=5)
        ig_out.channel_spacing = 2

        binned_by_hand = ig_out.allocate(None)

        l_out = 0
        for l in channel:
            k_out = 0
            for k in vertical:
                j_out = 0
                for j in horizontal_y:
                    i_out = 0
                    for i in horizontal_x:
                        binned_by_hand.array[l_out,k_out,j_out,i_out]  = data.array[l:l+channel.step,k:k+vertical.step,j:j+horizontal_y.step,i:i+horizontal_x.step].mean()
                        i_out +=1
                    j_out +=1
                k_out +=1
            l_out+=1

        numpy.testing.assert_allclose(binned_data.array, binned_by_hand.array,atol=0.003) 
        self.assertEqual(binned_data.geometry, binned_by_hand.geometry)


        #test with `out`
        binned_data.fill(0)
        proc.get_output(out=binned_data)
        numpy.testing.assert_allclose(binned_data.array, binned_by_hand.array,atol=0.003) 
        self.assertEqual(binned_data.geometry, binned_by_hand.geometry)
   


    def test_bin_acquisition_data(self):
        """
        Binning results tested with test_binning_cpp_ so this is checking wrappers with axis labels and geometry
        """
        ag = AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,0]).set_angles(numpy.linspace(0,360,8,endpoint=False)).set_panel([4,6],[0.1,0.2]).set_channels(10)
        data = ag.allocate('random')

        channel = range(0,10,2)
        angle = range(0,8,2)
        vertical = range(0,6,2)
        horizontal = range(0,4,2)

        roi = {'channel':channel,'vertical':vertical,'horizontal':horizontal,'angle':angle}
        proc = Binner(roi,accelerated=True)
        proc.set_input(data)
        binned_data = proc.get_output()

        ag_out = AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,0]).set_angles(numpy.linspace(22.5,360+22.5,4,endpoint=False)).set_panel([2,3],[0.2,0.4]).set_channels(5)
        binned_by_hand = ag_out.allocate(None)

        l_out = 0
        for l in channel:
            k_out = 0
            for k in angle:
                j_out = 0
                for j in vertical:
                    i_out = 0
                    for i in horizontal:
                        binned_by_hand.array[l_out,k_out,j_out,i_out]  = data.array[l:l+channel.step,k:k+vertical.step,j:j+vertical.step,i:i+horizontal.step].mean()
                        i_out +=1
                    j_out +=1
                k_out +=1
            l_out+=1

        numpy.testing.assert_allclose(binned_data.array, binned_by_hand.array,atol=0.003) 
        self.assertEqual(binned_data.geometry, binned_by_hand.geometry)


        #test with `out`
        binned_data.fill(0)
        proc.get_output(out=binned_data)
        numpy.testing.assert_allclose(binned_data.array, binned_by_hand.array,atol=0.003) 
        self.assertEqual(binned_data.geometry, binned_by_hand.geometry)
   

    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_imagedata_full(self):
        """
        This test bins a reconstructed volume. It then uses that geometry as the reconstruction window and reconstructs again.

        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        data.log(out=data)
        data*=-1

        recon =FBP(data).run(verbose=0)

        roi = {'vertical':(20,40,5),'horizontal_y':(70,100,3),'horizontal_x':(-80,-40,2)}
        binner = Binner(roi)

        binner.set_input(recon.geometry)
        ig_roi = binner.get_output()

        binner.set_input(recon)
        recon_binned = binner.get_output()

        self.assertEqual(ig_roi, recon_binned.geometry, msg="Binned geometries not equal")

        recon_roi =FBP(data, ig_roi).run(verbose=0)

        # not a very tight tolerance as binning and fbp at lower res are not identical operations.
        numpy.testing.assert_allclose(recon_roi.array, recon_binned.array, atol=5e-3)


    @unittest.skipUnless(has_astra and has_nvidia, "ASTRA GPU not installed")
    def test_aqdata_full(self):
        """
        This test bins a sinogram. It then uses that geometry for the forward projection.

        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry
        ag.set_labels(['vertical','angle','horizontal'])

        phantom = dataexample.SIMULATED_SPHERE_VOLUME.get()

        PO = AstraProjectionOperator(phantom.geometry, ag)
        fp_full = PO.direct(phantom)


        roi = {'angle':(25,26),'vertical':(5,62,2),'horizontal':(-75,0,2)}
        binner = Binner(roi)

        binner.set_input(ag)
        ag_roi = binner.get_output()

        binner.set_input(fp_full)
        fp_binned = binner.get_output()

        self.assertEqual(ag_roi, fp_binned.geometry, msg="Binned geometries not equal")

        PO = AstraProjectionOperator(phantom.geometry, ag_roi)
        fp_roi = PO.direct(phantom)

        # not a very tight tolerance as binning and fp at lower res are not identical operations.
        numpy.testing.assert_allclose(fp_roi.array, fp_binned.array, atol=0.06)

    @unittest.skip
    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_aqdata_full_tigre(self):
        """
        This test slices a sinogram. It then uses that geometry for the forward projection.

        This ensures the offsets are correctly set and the same window of data is output in both cases.

        Tigre geometry bug means this does not pass.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry

        phantom = dataexample.SIMULATED_SPHERE_VOLUME.get()

        PO = TigreProjectionOperator(phantom.geometry, ag)
        fp_full = PO.direct(phantom)


        roi = {'angle':(25,26),'vertical':(5,62,2),'horizontal':(-75,0,2)}
        binner = Binner(roi)

        binner.set_input(ag)
        ag_roi = binner.get_output()

        binner.set_input(fp_full)
        fp_binned = binner.get_output()

        self.assertEqual(ag_roi, fp_binned.geometry, msg="Binned geometries not equal")

        PO = TigreProjectionOperator(phantom.geometry, ag_roi)
        fp_roi = PO.direct(phantom)

        show2D([fp_roi,fp_binned,fp_roi-fp_binned])
        numpy.testing.assert_allclose(fp_roi.array, fp_binned.array, atol=0.06)


class TestSlicer(unittest.TestCase):

    def test_set_up_processor(self):
        ig = ImageGeometry(20,22,23,0.1,0.2,0.3,0.4,0.5,0.6,channels=24)
        data = ig.allocate('random')

        channel = range(0,10,3)
        vertical = range(0,8,2)
        horizontal_y = range(0,22,1)
        horizontal_x = range(0,4,4)

        roi = {'horizontal_y':horizontal_y,'horizontal_x':horizontal_x,'vertical':vertical,'channel':channel}
        proc = Slicer(roi)
        proc._set_up_processor(data)

        # check set values
        self.assertTrue(proc._shape_in == list(data.shape))

        shape_out =[
            len(channel),
            len(vertical),
            len(horizontal_y),
            len(horizontal_x),
        ]

        self.assertTrue(proc._shape_out == shape_out)
        self.assertTrue(proc._labels_in == ['channel','vertical','horizontal_y','horizontal_x'])
        numpy.testing.assert_array_equal(proc._processed_dims,[True,True,False,True])

        roi_ordered = [
            range(channel.start, shape_out[0] * channel.step, channel.step),
            range(vertical.start, shape_out[1] * vertical.step, vertical.step),
            range(horizontal_y.start, shape_out[2] * horizontal_y.step, horizontal_y.step),
            range(horizontal_x.start, shape_out[3] * horizontal_x.step, horizontal_x.step)
        ]

        self.assertTrue(proc._roi_ordered == roi_ordered)


    def test_process_acquisition_geometry_parallel2D(self):

        ag = AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,0.1).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'horizontal':(None,None,None)},

                # slice all
                {'channel':(None,None,4),'angle':(None,None,2),'horizontal':(None,None,16)},
        ]

        pix_end1 = ag.pixel_num_h -1
        pix_start2 = 0
        pix_end2 = 7 * 16 # last pixel index sliced multiplied by step size
        offset = ag.pixel_size_h*((pix_start2)-(pix_end1 - pix_end2 ))/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel2D(detector_position=[offset, 0.  ]).set_angles(numpy.linspace(0,360,180,endpoint=False)).set_panel(8,[1.6,0.1]).set_channels(1),
        ]

        for i, roi in enumerate(rois):
            proc = Slicer(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Slicing acquisition geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ag_gold[i], ag_out))

    def test_process_acquisition_geometry_parallel3D(self):

        ag = AcquisitionGeometry.create_Parallel3D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel([128,64],[0.1,0.2]).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # slice all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},
                
                # slice to single dimension
                {'vertical':(31,33,2)},
        ]

        pix_end_h1 = ag.pixel_num_h -1
        pix_start_h2 = 0
        pix_end_h2 = 7 * 16 # last pixel index sliced multiplied by step size
        offset_h = ag.pixel_size_h*((pix_start_h2)-(pix_end_h1-pix_end_h2))/2

        pix_end_v1 = ag.pixel_num_v -1
        pix_start_v2 = 0
        pix_end_v2 = 7 * 8 # last pixel index sliced multiplied by step size
        offset_v = ag.pixel_size_v*((pix_start_v2)-(pix_end_v1- pix_end_v2))/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel3D(detector_position=[offset_h, 0, offset_v]).set_angles(numpy.linspace(0,360,180,endpoint=False)).set_panel([8,8],[1.6,1.6]).set_channels(1),
                AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,[0.1,0.4]).set_channels(4),        
        ]

        for i, roi in enumerate(rois):
            proc = Slicer(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Slicing acquisition geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ag_gold[i], ag_out))


    def test_process_acquisition_geometry_cone2D(self):

        ag = AcquisitionGeometry.create_Cone2D([0,-50],[0,50]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,0.1).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'horizontal':(None,None,None)},

                # slice all
                {'channel':(None,None,4),'angle':(None,None,2),'horizontal':(None,None,16)},
        ]

        pix_end1 = ag.pixel_num_h -1
        pix_start2 = 0
        pix_end2 = 7 * 16 # last pixel index sliced multiplied by step size
        offset = ag.pixel_size_h*((pix_start2)-(pix_end1-pix_end2) )/2
        
        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Cone2D([0,-50],[offset,50]).set_angles(numpy.linspace(0,360,180,endpoint=False)).set_panel(8,[1.6,0.1]).set_channels(1),
        ]

        for i, roi in enumerate(rois):
            proc = Slicer(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Slicing acquisition geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ag_gold[i], ag_out))


    def test_process_acquisition_geometry_cone3D(self):

        ag = AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,0]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel([128,64],[0.1,0.2]).set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # slice all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},

                # shift detector with crop
                {'vertical':(32,65,2)},
                
                # slice to single dimension
                {'vertical':(32,34,2)},

        ]

        # roi1
        pix_end_h1 = ag.pixel_num_h -1
        pix_start_h2 = 0
        pix_end_h2 = 7 * 16 # last pixel index sliced multiplied by step size
        offset_h = ag.pixel_size_h*((pix_start_h2)-(pix_end_h1-pix_end_h2))/2

        pix_end_v1 = ag.pixel_num_v -1
        pix_start_v2 = 0
        pix_end_v2 = 7 * 8 # last pixel index sliced multiplied by step size
        offset_v = ag.pixel_size_v*((pix_start_v2)-(pix_end_v1-pix_end_v2) )/2

        #roi2
        vert_range = range(32,min(65,ag.pixel_num_v),2)
        pix_end_v1 = ag.pixel_num_v -1
        pix_start_v2 = 32
        pix_end_v2 = 15 * 2 +  pix_start_v2 # last pixel index sliced multiplied by step size
        offset_v2 = ag.pixel_size_v*((pix_start_v2)-(pix_end_v1-pix_end_v2))/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Cone3D([0,-50,0],[offset_h,50,offset_v]).set_angles(numpy.linspace(0,360,180,endpoint=False)).set_panel([8,8],[1.6,1.6]).set_channels(1),
                AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,offset_v2]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel([128,16],[0.1,0.4]).set_channels(4),
                AcquisitionGeometry.create_Cone2D([0,-50],[0,50]).set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,[0.1,0.4]).set_channels(4),        
        ]

        for i, roi in enumerate(rois):
            proc = Slicer(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Slicing acquisition geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ag_gold[i], ag_out))


    def test_process_image_geometry(self):

        ig_in = ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4)

        rois = [
                # same as input
                {'channel':(None,None,None),'vertical':(None,None,None),'horizontal_x':(None,None,None),'horizontal_y':(None,None,None)},

                # slice all
                {'channel':(None,None,3),'vertical':(None,None,7),'horizontal_x':(None,None,4),'horizontal_y':(None,None,5)},

                # crop and slice
                {'channel':(1,None,2),'vertical':(4,-8,4),'horizontal_x':(1,7,2),'horizontal_y':(4,-8,2)},
                
                # slice to single dimension
                {'channel':(None,None,4),'vertical':(None,None,28),'horizontal_x':(4,5,8),'horizontal_y':(None,None,16)},
        ]

        offset_x =0.1*(8-1-1*4)/2
        offset_y =0.2*(16-1-3 * 5)/2
        offset_z =0.3*(28-1-3 * 7)/2

        ig_gold = [ ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4),
                    ImageGeometry(2,4,4,0.4,1.0,2.1,center_x=-offset_x,center_y=-offset_y,center_z=-offset_z,channels=2),
                    ImageGeometry(3,2,4,0.2,0.4,1.2,center_x=-0.05,center_y=-0.5,center_z=-1.05,channels=2),
                    ImageGeometry(1,1,1,0.8,3.2,8.4,center_x=0.05,center_y=-1.5,center_z=-4.05, channels=1),
        ]

        #channel spacing isn't an initialisation argument
        ig_gold[1].channel_spacing=3
        ig_gold[2].channel_spacing=2
        ig_gold[3].channel_spacing=4


        for i, roi in enumerate(rois):
            proc = Slicer(roi=roi)
            proc.set_input(ig_in)
            ig_out = proc._process_image_geometry()
            self.assertEqual(ig_gold[i], ig_out, msg="Slicer image geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ig_gold[i], ig_out))

        with self.assertRaises(ValueError):
            roi = {'wrong label':(None,None,None)}
            proc = Slicer(roi=roi)
            proc.set_input(ig_in)
            ig_out = proc._process_image_geometry(ig_in)


        # slicing/cropping offsets geometry
        ig_in = ImageGeometry(128,128,128,16,16,16,80,240,-160)
        ig_gold = ImageGeometry(11,11,11,48,48,48,-184,-24,-424)

        roi = {'vertical':(32,64,3),'horizontal_x':(32,64,3),'horizontal_y':(32,64,3)}
        proc = Slicer(roi)
        proc.set_input(ig_in)
        ig_out = proc.get_output()

        self.assertEqual(ig_gold, ig_out, msg="Slicing image geometry with offset roi failed. \nExpected:\n{0}\nGot\n{1}".format(ig_gold, ig_out))


    def test_slice_image_data(self):

        ig = ImageGeometry(4,6,8,0.1,0.2,0.3,0.4,0.5,0.6,channels=10)
        data = ig.allocate('random')

        channel = range(0,10,2)
        vertical = range(0,8,2)
        horizontal_y = range(0,6,2)
        horizontal_x = range(0,4,2)

        roi = {'channel':channel,'vertical':vertical,'horizontal_x':horizontal_x,'horizontal_y':horizontal_y}
        proc = Slicer(roi)
        proc.set_input(data)
        sliced_data = proc.get_output()

        ig_out = ImageGeometry(2,3,4,0.2,0.4,0.6,0.4-0.5*0.1,0.5-0.5*0.2,0.6-0.5*0.3,channels=5)
        ig_out.channel_spacing = 2

        sliced_by_hand = ig_out.allocate(None)


        sliced_by_hand.fill( data.array[(
            slice(channel.start,channel.stop,channel.step),
            slice(vertical.start,vertical.stop,vertical.step),
            slice(horizontal_y.start,horizontal_y.stop,horizontal_y.step),
            slice(horizontal_x.start,horizontal_x.stop,horizontal_x.step))]
            )

        numpy.testing.assert_allclose(sliced_data.array, sliced_by_hand.array,atol=0.003) 
        self.assertEqual(sliced_data.geometry, sliced_by_hand.geometry)


        #test with `out`
        sliced_data.fill(0)
        proc.get_output(out=sliced_data)
        numpy.testing.assert_allclose(sliced_data.array, sliced_by_hand.array,atol=0.003) 
        self.assertEqual(sliced_data.geometry, sliced_by_hand.geometry)
   


    def test_slice_acquisition_data(self):

        ag = AcquisitionGeometry.create_Cone3D([0,-50,0],[0,50,0]).set_angles(numpy.linspace(0,360,8,endpoint=False)).set_panel([4,6],[0.1,0.2]).set_channels(10)
        data = ag.allocate('random')

        channel = range(0,10,2)
        angle = range(0,8,2)
        vertical = range(0,6,2)
        horizontal = range(0,4,2)

        roi = {'channel':channel,'vertical':vertical,'horizontal':horizontal,'angle':angle}
        proc = Slicer(roi)
        proc.set_input(data)
        sliced_data = proc.get_output()

        pix_end_h1 = ag.pixel_num_h -1
        pix_start_h2 = 0
        pix_end_h2 = 1 * 2 # last pixel index sliced multiplied by step size
        offset_h = ag.pixel_size_h*((pix_start_h2)-(pix_end_h1-pix_end_h2))/2

        pix_end_v1 = ag.pixel_num_v -1
        pix_start_v2 = 0
        pix_end_v2 = 2 * 2 # last pixel index sliced multiplied by step size
        offset_v = ag.pixel_size_v*((pix_start_v2)-(pix_end_v1-pix_end_v2))/2

        ag_out = AcquisitionGeometry.create_Cone3D([0,-50,0],[offset_h,50,offset_v]).set_angles(numpy.linspace(0,360,4,endpoint=False)).set_panel([2,3],[0.2,0.4]).set_channels(5)
        sliced_by_hand = ag_out.allocate(None)

        sliced_by_hand.fill( data.array[(
            slice(channel.start,channel.stop,channel.step),
            slice(angle.start,angle.stop,angle.step),
            slice(vertical.start,vertical.stop,vertical.step),
            slice(horizontal.start,horizontal.stop,horizontal.step))]
            )

        numpy.testing.assert_allclose(sliced_data.array, sliced_by_hand.array,atol=0.003) 
        self.assertEqual(sliced_data.geometry, sliced_by_hand.geometry,msg="Expected:\n{0}\nGot\n{1}".format(sliced_by_hand.geometry, sliced_data.geometry))


        #test with `out`
        sliced_data.fill(0)
        proc.get_output(out=sliced_data)
        numpy.testing.assert_allclose(sliced_data.array, sliced_by_hand.array,atol=0.003) 
        self.assertEqual(sliced_data.geometry, sliced_by_hand.geometry,msg="Expected:\n{0}\nGot\n{1}".format(sliced_by_hand.geometry, sliced_data.geometry))


    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_imagedata_full(self):
        """
        This test slices a reconstructed volume. It then uses that geometry as the reconstruction window and reconstructs again.

        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        data.log(out=data)
        data*=-1

        recon =FBP(data).run(verbose=0)

        roi = {'vertical':(20,40,5),'horizontal_y':(70,100,3),'horizontal_x':(-80,-40,2)}
        slicer = Slicer(roi)

        slicer.set_input(recon.geometry)
        ig_roi = slicer.get_output()

        slicer.set_input(recon)
        recon_sliced = slicer.get_output()

        self.assertEqual(ig_roi, recon_sliced.geometry, msg="Sliced geometries not equal")

        recon_roi =FBP(data, ig_roi).run(verbose=0)

        numpy.testing.assert_allclose(recon_roi.array, recon_sliced.array, atol=5e-6)


    @unittest.skipUnless(has_astra and has_nvidia, "ASTRA GPU not installed")
    def test_aqdata_full(self):
        """
        This test slices a sinogram. It then uses that geometry for the forward projection.

        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry
        ag.set_labels(['vertical','angle','horizontal'])

        phantom = dataexample.SIMULATED_SPHERE_VOLUME.get()

        PO = AstraProjectionOperator(phantom.geometry, ag)
        fp_full = PO.direct(phantom)


        roi = {'angle':(25,30,2),'vertical':(5,50,2),'horizontal':(-50,0,2)}
        slicer = Slicer(roi)

        slicer.set_input(ag)
        ag_roi = slicer.get_output()

        slicer.set_input(fp_full)
        fp_sliced = slicer.get_output()

        numpy.testing.assert_allclose(fp_sliced.array, fp_full.array[5:50:2,25:30:2,-50::2])

        self.assertEqual(ag_roi, fp_sliced.geometry, msg="Sliced geometries not equal")

        PO = AstraProjectionOperator(phantom.geometry, ag_roi)
        fp_roi = PO.direct(phantom)

        numpy.testing.assert_allclose(fp_roi.array, fp_sliced.array, 1e-4)

    @unittest.skip
    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_aqdata_full_tigre(self):
        """
        This test slices a sinogram. It then uses that geometry for the forward projection.

        This ensures the offsets are correctly set and the same window of data is output in both cases.

        Tigre geometry bug means this does not pass.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry

        phantom = dataexample.SIMULATED_SPHERE_VOLUME.get()

        PO = TigreProjectionOperator(phantom.geometry, ag)
        fp_full = PO.direct(phantom)


        roi = {'angle':(25,30,2),'vertical':(5,50,2),'horizontal':(-50,0,2)}
        slicer = Slicer(roi)

        slicer.set_input(ag)
        ag_roi = slicer.get_output()

        slicer.set_input(fp_full)
        fp_sliced = slicer.get_output()

        numpy.testing.assert_allclose(fp_sliced.array, fp_full.array[25:30:2,5:50:2,-50::2])

        self.assertEqual(ag_roi, fp_sliced.geometry, msg="Sliced geometries not equal")

        PO = TigreProjectionOperator(phantom.geometry, ag_roi)
        fp_roi = PO.direct(phantom)

        numpy.testing.assert_allclose(fp_roi.array, fp_sliced.array, 1e-4)

class TestCentreOfRotation_parallel(unittest.TestCase):
    
    def setUp(self):
        data_raw = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        self.data_DLS = data_raw.log()
        self.data_DLS *= -1

    def test_CofR_xcorrelation(self):       

        corr = CentreOfRotationCorrector.xcorrelation(slice_index='centre', projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS)
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     
        
        corr = CentreOfRotationCorrector.xcorrelation(slice_index=67, projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS)
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)              

    @unittest.skipUnless(has_astra and has_nvidia, "ASTRA GPU not installed")
    def test_CofR_image_sharpness_astra(self):

        corr = CentreOfRotationCorrector.image_sharpness(search_range=20, backend='astra')
        corr.set_input(self.data_DLS)
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=1)    


    @unittest.skipUnless(False, "TIGRE not installed")
    def skiptest_test_CofR_image_sharpness_tigre(self): #currently not avaliable for parallel beam
        corr = CentreOfRotationCorrector.image_sharpness(search_range=20, backend='tigre')
        corr.set_input(self.data_DLS)
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     

    def test_CenterOfRotationCorrector(self):       
        corr = CentreOfRotationCorrector.xcorrelation(slice_index='centre', projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS)
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     
        
        corr = CentreOfRotationCorrector.xcorrelation(slice_index=67, projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS)
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)              


class TestCentreOfRotation_conebeam(unittest.TestCase):

    def setUp(self):
        angles = numpy.linspace(0, 360, 180, endpoint=False)

        ag_orig = AcquisitionGeometry.create_Cone2D([0,-100],[0,100])\
            .set_panel(64,0.2)\
            .set_angles(angles)\
            .set_labels(['angle', 'horizontal'])

        ig = ag_orig.get_ImageGeometry()
        phantom = TomoPhantom.get_ImageData(12, ig)

        if has_tigre:
            Op = TigreProjectionOperator(ig, ag_orig, direct_method='Siddon')
        else:
            Op = AstraProjectionOperator(ig, ag_orig)

        self.data_0 = Op.direct(phantom)

        ag_offset = AcquisitionGeometry.create_Cone2D([0,-100],[0,100],rotation_axis_position=(-0.150,0))\
            .set_panel(64,0.2)\
            .set_angles(angles)\
            .set_labels(['angle', 'horizontal'])

        if has_tigre:
            Op = TigreProjectionOperator(ig, ag_offset, direct_method='Siddon')
        else:
            Op = AstraProjectionOperator(ig, ag_offset)        
            
        self.data_offset = Op.direct(phantom)
        self.data_offset.geometry = ag_orig

    @unittest.skipUnless(has_tomophantom and has_astra and has_nvidia, "Tomophantom or ASTRA GPU not installed")
    def test_CofR_image_sharpness_astra(self):
        corr = CentreOfRotationCorrector.image_sharpness(backend='astra')
        ad_out = corr(self.data_0)
        self.assertAlmostEqual(0.000, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

        corr = CentreOfRotationCorrector.image_sharpness(backend='astra')
        ad_out = corr(self.data_offset)
        self.assertAlmostEqual(-0.150, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

    @unittest.skipUnless(has_tomophantom and has_tigre and has_nvidia, "Tomophantom or TIGRE GPU not installed")
    def test_CofR_image_sharpness_tigre(self): #currently not avaliable for parallel beam
        corr = CentreOfRotationCorrector.image_sharpness(backend='tigre')
        ad_out = corr(self.data_0)
        self.assertAlmostEqual(0.000, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

        corr = CentreOfRotationCorrector.image_sharpness(backend='tigre')
        ad_out = corr(self.data_offset)
        self.assertAlmostEqual(-0.150, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

class TestDataProcessor(unittest.TestCase):
   
    def test_DataProcessorBasic(self):

        dc_in = DataContainer(numpy.arange(10), True)
        dc_out = dc_in.copy()

        ax = AX()
        ax.scalar = 2
        ax.set_input(dc_in)

        #check results with out
        out_gold = dc_in*2
        ax.get_output(out=dc_out)
        numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #check results with return
        dc_out2 = ax.get_output()
        numpy.testing.assert_array_equal(dc_out2.as_array(), out_gold.as_array())

        #check call method
        dc_out2 = ax(dc_in)
        numpy.testing.assert_array_equal(dc_out2.as_array(), out_gold.as_array())

        #check storage mode
        self.assertFalse(ax.store_output)
        self.assertTrue(ax.output == None)
        ax.store_output = True
        self.assertTrue(ax.store_output)

        #check storing a copy and not a reference
        ax.set_input(dc_in)
        dc_out = ax.get_output()
        numpy.testing.assert_array_equal(ax.output.as_array(), out_gold.as_array())
        self.assertFalse(id(ax.output.as_array()) == id(dc_out.as_array()))

        #check recalculation on argument change
        ax.scalar = 3
        out_gold = dc_in*3
        ax.get_output(out=dc_out)
        numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #check recalculation on input change
        dc_in2 = dc_in.copy()
        dc_in2 *=2
        out_gold = dc_in2*3
        ax.set_input(dc_in2)
        ax.get_output(out=dc_out)
        numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #check auto recalculation if input modified (won't pass)
        dc_in2 *= 2
        out_gold = dc_in2*3
        ax.get_output(out=dc_out)
        #numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #raise error if input is deleted
        dc_in2 = dc_in.copy()
        ax.set_input(dc_in2)
        del dc_in2
        with self.assertRaises(ValueError):
            dc_out = ax.get_output()


    def test_DataProcessorChaining(self):
        shape = (2,3,4,5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range( size )])
        a = numpy.reshape(a, shape)
        ds = DataContainer(a, False, ['X', 'Y','Z' ,'W'])
        c = ds.get_slice(Y=0)
        c.reorder(['Z','W','X'])
        arr = c.as_array()
        #[ 0 60  1 61  2 62  3 63  4 64  5 65  6 66  7 67  8 68  9 69 10 70 11 71
        # 12 72 13 73 14 74 15 75 16 76 17 77 18 78 19 79]
        
        #print(arr)
    
        ax = AX()
        ax.scalar = 2

        numpy.testing.assert_array_equal(ax(c).as_array(), arr*2)

        ax.set_input(c)
        numpy.testing.assert_array_equal(ax.get_output().as_array(), arr*2)

        cast = CastDataContainer(dtype=numpy.float32)
        cast.set_input(c)
        out = cast.get_output()
        self.assertTrue(out.as_array().dtype == numpy.float32)
        out *= 0 
        axm = AX()
        axm.scalar = 0.5
        axm.set_input(c)
        axm.get_output(out)
        numpy.testing.assert_array_equal(out.as_array(), arr*0.5)
        
        #print("check call method of DataProcessor")
        numpy.testing.assert_array_equal(axm(c).as_array(), arr*0.5)        
    
        
        # check out in DataSetProcessor
        #a = numpy.asarray([i for i in range( size )])
           
        # create a PixelByPixelDataProcessor
        
        #define a python function which will take only one input (the pixel value)
        pyfunc = lambda x: -x if x > 20 else x
        clip = PixelByPixelDataProcessor()
        clip.pyfunc = pyfunc 
        clip.set_input(c)    
        #clip.apply()
        v = clip.get_output().as_array()
        
        self.assertTrue(v.max() == 19)
        self.assertTrue(v.min() == -79)
        
        #print ("clip in {0} out {1}".format(c.as_array(), clip.get_output().as_array()))
        
        #dsp = DataProcessor()
        #dsp.set_input(ds)
        #dsp.input = a
        # pipeline
    
        chain = AX()
        chain.scalar = 0.5
        chain.set_input_processor(ax)
        #print ("chain in {0} out {1}".format(ax.get_output().as_array(), chain.get_output().as_array()))
        numpy.testing.assert_array_equal(chain.get_output().as_array(), arr)
        
        #print("check call method of DataProcessor")
        numpy.testing.assert_array_equal(ax(chain(c)).as_array(), arr)        

class TestMaskGenerator(unittest.TestCase):       

    def test_MaskGenerator(self): 
    
        IG = ImageGeometry(voxel_num_x=10,
                        voxel_num_y=10)
        
        data = IG.allocate('random')
        
        data.as_array()[2,3] = float('inf')
        data.as_array()[4,5] = float('nan')
        
        # check special values - default
        m = MaskGenerator.special_values()
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[2,3] = 0
        mask_manual[4,5] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check nan
        m = MaskGenerator.special_values(inf=False)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[4,5] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check inf
        m = MaskGenerator.special_values(nan=False)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[2,3] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check threshold
        data = IG.allocate('random')
        data.as_array()[6,8] = 100
        data.as_array()[1,3] = 80
        
        m = MaskGenerator.threshold(None, 70)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[6,8] = 0
        mask_manual[1,3] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.threshold(None, 80)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[6,8] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check quantile
        data = IG.allocate('random')
        data.as_array()[6,8] = 100
        data.as_array()[1,3] = 80
        
        m = MaskGenerator.quantile(None, 0.98)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[6,8] = 0
        mask_manual[1,3] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.quantile(None, 0.99)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[6,8] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check mean
        IG = ImageGeometry(voxel_num_x=200,
                            voxel_num_y=200)
        #data = IG.allocate('random', seed=10)
        data = IG.allocate()
        numpy.random.seed(10)
        data.fill(numpy.random.rand(200,200))
        data.as_array()[7,4] += 10 * numpy.std(data.as_array()[7,:])
        
        m = MaskGenerator.mean(axis='horizontal_x')
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.mean(window=5)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check median
        m = MaskGenerator.median(axis='horizontal_x')
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.median()
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check movmean
        m = MaskGenerator.mean(window=10)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        #
        m = MaskGenerator.mean(window=20, axis='horizontal_y')
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.mean(window=10, threshold_factor=10)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check movmedian
        m = MaskGenerator.median(window=20)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check movmedian
        m = MaskGenerator.median(window=40)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)

class TestTransmissionAbsorptionConverter(unittest.TestCase):

    def test_TransmissionAbsorptionConverter(self):
            
        ray_direction = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    detector_direction_y=detector_direction_col,
                                                    rotation_axis_position=rotation_axis_position,
                                                    rotation_axis_direction=rotation_axis_direction)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((10, 5), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        ad = AG.allocate('random')
        
        s = TransmissionAbsorptionConverter(white_level=10, min_intensity=0.1)
        s.set_input(ad)
        data_exp = s.get_output()
        
        data_new = ad.as_array().copy()
        data_new /= 10
        data_new[data_new < 0.1] = 0.1
        data_new = -1 * numpy.log(data_new)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), data_new, rtol=1E-6)
        
        data_exp.fill(0)
        s.process(out=data_exp)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), data_new, rtol=1E-6)

class TestAbsorptionTransmissionConverter(unittest.TestCase):

    def test_AbsorptionTransmissionConverter(self):

        ray_direction = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    detector_direction_y=detector_direction_col,
                                                    rotation_axis_position=rotation_axis_position,
                                                    rotation_axis_direction=rotation_axis_direction)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((10, 5), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        ad = AG.allocate('random')
        
        s = AbsorptionTransmissionConverter(white_level=10)
        s.set_input(ad)
        data_exp = s.get_output()
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), numpy.exp(-ad.as_array())*10, rtol=1E-6)
        
        data_exp.fill(0)
        s.process(out=data_exp)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), numpy.exp(-ad.as_array())*10, rtol=1E-6)  


class TestMasker(unittest.TestCase):       

    def setUp(self):
        IG = ImageGeometry(voxel_num_x=10,
                        voxel_num_y=10)
        
        self.data_init = IG.allocate('random')
        
        self.data = self.data_init.copy()
        
        self.data.as_array()[2,3] = float('inf')
        self.data.as_array()[4,5] = float('nan')
        
        mask_manual = numpy.ones((10,10), dtype=bool)
        mask_manual[2,3] = 0
        mask_manual[4,5] = 0
        
        self.mask_manual = DataContainer(mask_manual, dimension_labels=self.data.dimension_labels) 
        self.mask_generated = MaskGenerator.special_values()(self.data)

    def test_Masker_Manual(self):
        self.Masker_check(self.mask_manual, self.data, self.data_init)

    def test_Masker_generated(self):
        self.Masker_check(self.mask_generated, self.data, self.data_init)

    def Masker_check(self, mask, data, data_init): 

        # test vaue mode
        m = Masker.value(mask=mask, value=10)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        data_test[2,3] = 10
        data_test[4,5] = 10
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)     
        
        # test mean mode
        m = Masker.mean(mask=mask)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp = numpy.sum(data_init.as_array())-(data_init.as_array()[2,3]+data_init.as_array()[4,5])
        tmp /= 98
        data_test[2,3] = tmp
        data_test[4,5] = tmp
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        
        # test median mode
        m = Masker.median(mask=mask)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp = data.as_array()[numpy.isfinite(data.as_array())]
        data_test[2,3] = numpy.median(tmp)
        data_test[4,5] = numpy.median(tmp)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)
        
        # test axis int
        m = Masker.median(mask=mask, axis=0)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp1 = data.as_array()[2,:][numpy.isfinite(data.as_array()[2,:])]
        tmp2 = data.as_array()[4,:][numpy.isfinite(data.as_array()[4,:])]
        data_test[2,3] = numpy.median(tmp1)
        data_test[4,5] = numpy.median(tmp2)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6) 
        
        # test axis str
        m = Masker.mean(mask=mask, axis=data.dimension_labels[1])
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp1 = data.as_array()[:,3][numpy.isfinite(data.as_array()[:,3])]
        tmp2 = data.as_array()[:,5][numpy.isfinite(data.as_array()[:,5])]
        data_test[2,3] = numpy.sum(tmp1) / 9
        data_test[4,5] = numpy.sum(tmp2) / 9
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)
        
        # test inline
        data = data_init.copy()
        m = Masker.value(mask=mask, value=10)
        m.set_input(data)
        m.process(out=data)
        
        data_test = data_init.copy().as_array()
        data_test[2,3] = 10
        data_test[4,5] = 10
        
        numpy.testing.assert_allclose(data.as_array(), data_test, rtol=1E-6) 
        
        # test mask numpy 
        data = data_init.copy()
        m = Masker.value(mask=mask.as_array(), value=10)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        data_test[2,3] = 10
        data_test[4,5] = 10
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        
        # test interpolate
        data = data_init.copy()
        m = Masker.interpolate(mask=mask, method='linear', axis='horizontal_y')
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        data_test[2,3] = (data_test[1,3] + data_test[3,3]) / 2
        data_test[4,5] = (data_test[3,5] + data_test[5,5]) / 2
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        

        
if __name__ == "__main__":
    
    d = TestDataProcessor()
    d.test_DataProcessorChaining()

