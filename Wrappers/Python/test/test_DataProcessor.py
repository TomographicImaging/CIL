#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
import numpy

import sys
import os
from unittest.mock import patch
import logging

from cil.framework import DataContainer, ImageGeometry, ImageData, VectorGeometry, AcquisitionData, AcquisitionGeometry

from cil.utilities import dataexample
from cil.utilities import quality_measures

from cil.framework import AX, CastDataContainer, PixelByPixelDataProcessor
from cil.recon import FBP

from cil.processors import CentreOfRotationCorrector
from cil.processors.CofR_xcorrelation import CofR_xcorrelation
from cil.processors.CofR_image_sharpness import CofR_image_sharpness
from cil.processors import TransmissionAbsorptionConverter, AbsorptionTransmissionConverter
from cil.processors import Slicer, Binner, MaskGenerator, Masker, Padder, PaganinProcessor, FluxNormaliser, Normaliser, LaminographyCorrector
import gc

from utils import has_numba
if has_numba:
    import numba

from scipy import constants
from scipy.fft import ifftshift

from utils import has_astra, has_tigre, has_nvidia, has_tomophantom, initialise_tests, has_ipp, has_matplotlib

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator as AstraProjectionOperator

if has_tigre:
    from cil.plugins.tigre import ProjectionOperator as TigreProjectionOperator

if has_tomophantom:
    from cil.plugins import TomoPhantom

if has_ipp:
    from cil.processors.cilacc_binner import Binner_IPP


@unittest.skipUnless(has_ipp, "Requires IPP libraries")
class TestBinner_cillacc(unittest.TestCase):
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
        data = ig.allocate('random', seed=42)

        channel = range(0,10,3)
        vertical = range(0,8,2)
        horizontal_y = range(0,22,1)
        horizontal_x = range(0,4,4)

        roi = {'horizontal_y':horizontal_y,'horizontal_x':horizontal_x,'vertical':vertical,'channel':channel}
        
        # check init non-default values
        proc = Binner(roi=roi,accelerated=False)
        self.assertTrue(proc._accelerated==False)
        
        proc = Binner(roi,accelerated=True)
        proc.set_input(data)
        proc._set_up()

        # check set values
        self.assertTrue(proc._shape_in == list(data.shape))

        shape_out =[(channel.stop - channel.start)//channel.step,
        (vertical.stop - vertical.start)//vertical.step,
        (horizontal_y.stop - horizontal_y.start)//horizontal_y.step,
        (horizontal_x.stop - horizontal_x.start)//horizontal_x.step
        ]

        self.assertTrue(proc._shape_out_full == shape_out)
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

        angles_full = numpy.linspace(0,360,360,endpoint=False)
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(angles_full).set_panel([128,64],[0.1,0.2], origin='bottom-left').set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # bin all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},

                # bin to single dimension
                {'vertical':(31,33,2)},


                # crop  asymmetrically
                {'vertical':(10,None,None),'horizontal':(None,-20,None)}
        ]

        #calculate offsets for geometry4
        ag4_offset_h = -ag.pixel_size_h*20/2
        ag4_offset_v = ag.pixel_size_v*10/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel3D().set_angles(numpy.linspace(0.5,360.5,180,endpoint=False)).set_panel([8,8],[1.6,1.6], origin='bottom-left').set_channels(1),
                AcquisitionGeometry.create_Parallel2D().set_angles(angles_full).set_panel(128,[0.1,0.4], origin='bottom-left').set_channels(4),
                AcquisitionGeometry.create_Parallel3D(detector_position=[ag4_offset_h, 0, ag4_offset_v]).set_angles(angles_full).set_panel([108,54],[0.1,0.2], origin='bottom-left').set_channels(4)
        ]

        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Binning acquisition geometry with roi {}".format(i))


    def test_process_acquisition_geometry_parallel3D_origin(self):

        #tests the geometry output with a non-default origin choice

        angles_full = numpy.linspace(0,360,360,endpoint=False)
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(angles_full).set_panel([128,64],[0.1,0.2], origin='top-right').set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # bin all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},

                # bin to single dimension
                {'vertical':(31,33,2)},


                # crop  asymmetrically
                {'vertical':(10,None,None),'horizontal':(None,-20,None)}
        ]

        #calculate offsets for geometry4
        ag4_offset_h = ag.pixel_size_h*20/2
        ag4_offset_v = -ag.pixel_size_v*10/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel3D().set_angles(numpy.linspace(0.5,360.5,180,endpoint=False)).set_panel([8,8],[1.6,1.6], origin='top-right').set_channels(1),
                AcquisitionGeometry.create_Parallel2D().set_angles(angles_full).set_panel(128,[0.1,0.4], origin='top-right').set_channels(4),
                AcquisitionGeometry.create_Parallel3D(detector_position=[ag4_offset_h, 0, ag4_offset_v]).set_angles(angles_full).set_panel([108,54],[0.1,0.2], origin='top-right').set_channels(4)
        ]

        for i, roi in enumerate(rois):
            proc = Binner(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Binning acquisition geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ag_gold[i], ag_out))



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
                {'vertical':(32,64,2)},

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

    def test_process_acquisition_geometry_cone3DFlex(self):
 
        source_position_set=[[0,-100000,0]]
        detector_position_set=[[0,0,0]]
        detector_direction_x_set=[[1, 0, 0]]
        detector_direction_y_set=[[0, 0, 1]]
        ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([128,64],[0.1,0.2]).set_channels(4)


        roi = {'channel':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)}

        proc = Binner(roi=roi)

        with self.assertRaises(NotImplementedError):
            proc.set_input(ag)

    def test_process_image_geometry(self):

        ig_in = ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4)

        rois = [
                # same as input
                {'channel':(None,None,None),'vertical':(None,None,None),'horizontal_x':(None,None,None),'horizontal_y':(None,None,None)},

                # bin all
                {'channel':(None,None,3),'vertical':(None,None,7),'horizontal_x':(None,None,4),'horizontal_y':(None,None,5)},

                # crop and bin
                {'channel':(1,None,2),'vertical':(4,-8,4),'horizontal_x':(1,7,2),'horizontal_y':(4,-8,2)},

                # bin to vector
                {'channel':(None,None,4),'vertical':(None,None,28),'horizontal_x':(None,None,4),'horizontal_y':(None,None,16)},

                #bin to single element
                {'channel':(None,None,4),'vertical':(None,None,28),'horizontal_x':(None,None,8),'horizontal_y':(None,None,16)},

        ]

        ig_gold = [ ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4),
                    ImageGeometry(2,3,4,0.4,1.0,2.1,center_y=-0.1,channels=1),
                    ImageGeometry(3,2,4,0.2,0.4,1.2,center_y=-0.4, center_z=-0.6, channels=1),
                    VectorGeometry(2, dimension_labels='horizontal_x'),
                    None
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
        data = ig.allocate('random', seed=42)

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
        data = ig.allocate('random', seed=42)

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
        data = ag.allocate('random', seed=42)

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



    def test_process_acquisition(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        geometry = AcquisitionGeometry.create_Parallel3D().set_angles([0,90]).set_panel([4,3])
        data_in = AcquisitionData(arr, False, geometry)

        roi = {'vertical':(None,None,2),'angle':(None,None,2),'horizontal':(None,None,2)}
        proc = Binner(roi)

        geometry_gold = AcquisitionGeometry.create_Parallel2D().set_angles([45]).set_panel(2,2)
        el1 = (0+1+4+5+12+13+16+17)/8
        el2 = (2+3+6+7+14+15+18+19)/8
        data_gold = numpy.array([el1,el2],dtype=numpy.float32)

        proc.set_input(data_in.geometry)
        geometry_out = proc.process()
        self.assertEqual(geometry_out, geometry_gold,
        msg="Binner failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_out, geometry_gold))

        proc.set_input(data_in)
        data_out = proc.process()

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Binner failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))

        data_out.fill(0)
        proc.process(out=data_out)

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Binner failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))


    def test_process_image(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        geometry = ImageGeometry(4,3,2)
        data_in = ImageData(arr, False, geometry)

        roi = {'vertical':(None,None,2),'horizontal_y':(None,None,2),'horizontal_x':(None,None,2)}
        proc = Binner(roi)

        geometry_gold = VectorGeometry(2,dimension_labels='horizontal_x')
        el1 = (0+1+4+5+12+13+16+17)/8
        el2 = (2+3+6+7+14+15+18+19)/8
        data_gold = numpy.array([el1,el2],dtype=numpy.float32)

        proc.set_input(data_in.geometry)
        geometry_out = proc.process()
        self.assertEqual(geometry_out, geometry_gold,
        msg="Binner failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_out, geometry_gold))

        proc.set_input(data_in)
        data_out = proc.process()

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Binner failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))

        data_out.fill(0)
        proc.process(out=data_out)

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Binner failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))


    def test_process_data_container(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        data_in = DataContainer(arr,False)

        #default labels
        roi = {'dimension_00':(None,None,2),'dimension_01':(None,None,2),'dimension_02':(None,None,2)}

        el1 = (0+1+4+5+12+13+16+17)/8
        el2 = (2+3+6+7+14+15+18+19)/8
        data_gold = numpy.array([el1,el2],dtype=numpy.float32)

        proc = Binner(roi)
        proc.set_input(data_in)
        data_out = proc.process()
        numpy.testing.assert_array_equal(data_gold, data_out.array)

        data_out.fill(0)
        proc.process(out=data_out)
        numpy.testing.assert_array_equal(data_gold, data_out.array)

        # custom labels
        data_in = DataContainer(arr,False,['LABEL_A','LABEL_B','LABEL_C'])

        roi = {'LABEL_A':(None,None,2),'LABEL_B':(None,None,2),'LABEL_C':(None,None,2)}

        proc = Binner(roi)
        proc.set_input(data_in)
        data_out = proc.process()
        numpy.testing.assert_array_equal(data_gold, data_out.array)


    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_imagedata_full_tigre(self):
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
    def test_aqdata_full_astra(self):
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


    @unittest.skipUnless(has_astra and has_nvidia, "ASTRA GPU not installed")
    def test_aqdata_full_origin_astra(self):
        """
        This test bins a sinogram. It then uses that geometry for the forward projection.

        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """
        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry
        ag.set_labels(['vertical','angle','horizontal'])
        ag.config.panel.origin='top-right'

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
        numpy.testing.assert_allclose(fp_roi.array, fp_binned.array, atol=0.08)


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

        numpy.testing.assert_allclose(fp_roi.array, fp_binned.array, atol=0.06)


class TestSlicer(unittest.TestCase):

    def test_set_up_processor(self):
        ig = ImageGeometry(20,22,23,0.1,0.2,0.3,0.4,0.5,0.6,channels=24)
        data = ig.allocate('random', seed=42)

        channel = range(0,10,3)
        vertical = range(0,8,2)
        horizontal_y = range(0,22,1)
        horizontal_x = range(0,4,4)

        roi = {'horizontal_y':horizontal_y,'horizontal_x':horizontal_x,'vertical':vertical,'channel':channel}
        proc = Slicer(roi)
        proc.set_input(data)
        proc._set_up()

        # check set values
        self.assertTrue(proc._shape_in == list(data.shape))

        shape_out =[
            len(channel),
            len(vertical),
            len(horizontal_y),
            len(horizontal_x),
        ]

        self.assertTrue(proc._shape_out_full == shape_out)
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

        angles_full = numpy.linspace(0,360,360,endpoint=False)
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(angles_full).set_panel([128,64],[0.1,0.2], origin='bottom-left').set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # slice all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},

                # slice to single dimension
                {'vertical':(31,33,2)},

                # slice  asymmetrically
                {'vertical':(10,None,None),'horizontal':(None,-20,None)}
        ]

        #calculate offsets for geometry2
        pix_end_h1 = ag.pixel_num_h -1
        pix_start_h2 = 0
        pix_end_h2 = 7 * 16 # last pixel index sliced multiplied by step size
        ag2_offset_h = ag.pixel_size_h*((pix_start_h2)-(pix_end_h1-pix_end_h2))/2

        pix_end_v1 = ag.pixel_num_v -1
        pix_start_v2 = 0
        pix_end_v2 = 7 * 8 # last pixel index sliced multiplied by step size
        ag2_offset_v = ag.pixel_size_v*((pix_start_v2)-(pix_end_v1- pix_end_v2))/2

        #calculate offsets for geometry4
        ag4_offset_h = -ag.pixel_size_h*20/2
        ag4_offset_v = ag.pixel_size_v*10/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel3D(detector_position=[ag2_offset_h, 0, ag2_offset_v]).set_angles(numpy.linspace(0,360,180,endpoint=False)).set_panel([8,8],[1.6,1.6]).set_channels(1),
                AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,[0.1,0.4]).set_channels(4),
                AcquisitionGeometry.create_Parallel3D(detector_position=[ag4_offset_h, 0, ag4_offset_v]).set_angles(angles_full).set_panel([108,54],[0.1,0.2]).set_channels(4)
        ]

        for i, roi in enumerate(rois):
            proc = Slicer(roi=roi)
            proc.set_input(ag)
            ag_out = proc._process_acquisition_geometry()

            self.assertEqual(ag_gold[i], ag_out, msg="Slicing acquisition geometry with roi {0}. \nExpected:\n{1}\nGot\n{2}".format(i,ag_gold[i], ag_out))

    def test_process_acquisition_geometry_parallel3D_origin(self):

        #tests the geometry output with a non-default origin choice

        angles_full = numpy.linspace(0,360,360,endpoint=False)
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(angles_full).set_panel([128,64],[0.1,0.2], origin='top-right').set_channels(4)

        rois = [
                # same as input
                {'channel':(None,None,None),'angle':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)},

                # slice all
                {'channel':(None,None,4),'angle':(None,None,2),'vertical':(None,None,8),'horizontal':(None,None,16)},

                # slice to single dimension
                {'vertical':(31,33,2)},

                # slice  asymmetrically
                {'vertical':(10,None,None),'horizontal':(None,-20,None)}
        ]

        #calculate offsets for geometry2
        pix_end_h1 = ag.pixel_num_h -1
        pix_start_h2 = 0
        pix_end_h2 = 7 * 16 # last pixel index sliced multiplied by step size
        ag2_offset_h = -ag.pixel_size_h*((pix_start_h2)-(pix_end_h1-pix_end_h2))/2

        pix_end_v1 = ag.pixel_num_v -1
        pix_start_v2 = 0
        pix_end_v2 = 7 * 8 # last pixel index sliced multiplied by step size
        ag2_offset_v = -ag.pixel_size_v*((pix_start_v2)-(pix_end_v1- pix_end_v2))/2

        #calculate offsets for geometry4
        ag4_offset_h = ag.pixel_size_h*20/2
        ag4_offset_v = -ag.pixel_size_v*10/2

        ag_gold = [
                ag.copy(),
                AcquisitionGeometry.create_Parallel3D(detector_position=[ag2_offset_h, 0, ag2_offset_v]).set_angles(numpy.linspace(0,360,180,endpoint=False)).set_panel([8,8],[1.6,1.6], origin='top-right').set_channels(1),
                AcquisitionGeometry.create_Parallel2D().set_angles(numpy.linspace(0,360,360,endpoint=False)).set_panel(128,[0.1,0.4], origin='top-right').set_channels(4),
                AcquisitionGeometry.create_Parallel3D(detector_position=[ag4_offset_h, 0, ag4_offset_v]).set_angles(angles_full).set_panel([108,54],[0.1,0.2], origin='top-right').set_channels(4)
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
                {'vertical':(32,64,2)},

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

    def test_process_acquisition_geometry_cone3DFlex(self):
 
        source_position_set=[[0,-100000,0], [0,-90000,0], [0,-90000,1], [0,-80000,0]]
        detector_position_set=[[0,0,0], [0,0,1], [0,0,2], [0,0,3]]
        detector_direction_x_set=[[1, 0, 0], [0.5, 0, 0], [1, 0, 0], [0.8,0,0]]
        detector_direction_y_set=[[0, 0, 1], [0,0,0.8], [0,0,1.1], [0,0,1.2]]
        ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([128,64],[0.1,0.2]).set_channels(4)

        roi_invalid = {'channel':(None,None,None),'vertical':(None,None,None),'horizontal':(None,None,None)}

        with self.assertRaises(NotImplementedError):
            proc = Slicer(roi=roi_invalid)
            proc.set_input(ag)
        
        roi_valid =  {'projection':(1,3,2)}

        slicer = Slicer(roi=roi_valid)
        sliced = slicer(ag)

        expected_source_position_set = source_position_set[1:3:2]
        expected_detector_position_set = detector_position_set[1:3:2]
        expected_detector_direction_x_set = detector_direction_x_set[1:3:2]
        expected_detector_direction_y_set = detector_direction_y_set[1:3:2]
        expected_num_positions = len(expected_source_position_set)

        expected_acq_geometry = AcquisitionGeometry.create_Cone3D_Flex(
            expected_source_position_set, expected_detector_position_set, expected_detector_direction_x_set, expected_detector_direction_y_set)
        expected_acq_geometry.set_panel([128,64],[0.1,0.2]).set_channels(4)

        numpy.testing.assert_allclose(expected_source_position_set, [x.position for x in sliced.config.system.source])
        numpy.testing.assert_allclose(expected_num_positions, sliced.config.system.num_positions)
        self.assertEqual(expected_acq_geometry, sliced)

        roi_channels = {'channel':(1,3,2)}

        slicer = Slicer(roi=roi_channels)
        sliced = slicer(ag)
        ag_expected = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([128,64],[0.1,0.2]).set_channels(1)

        self.assertEqual(ag_expected, sliced)

       

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
                {'channel':(None,None,4),'vertical':(None,None,28),'horizontal_x':(None,None,4),'horizontal_y':(None,None,16)},

                 # slice to single element
                {'channel':(None,None,4),'vertical':(None,None,28),'horizontal_x':(None, None,8),'horizontal_y':(None,None,16)},
        ]

        offset_x =0.1*(8-1-1*4)/2
        offset_y =0.2*(16-1-3 * 5)/2
        offset_z =0.3*(28-1-3 * 7)/2

        ig_gold = [ ImageGeometry(8,16,28,0.1,0.2,0.3,channels=4),
                    ImageGeometry(2,4,4,0.4,1.0,2.1,center_x=-offset_x,center_y=-offset_y,center_z=-offset_z,channels=2),
                    ImageGeometry(3,2,4,0.2,0.4,1.2,center_x=-0.05,center_y=-0.5,center_z=-1.05,channels=2),
                    VectorGeometry(2, dimension_labels='horizontal_x'),
                    None
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
        data = ig.allocate('random', seed=42)

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
        data = ag.allocate('random', seed=42)

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
    def test_imagedata_full_tigre(self):
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
    def test_aqdata_full_astra(self):
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


    @unittest.skipUnless(has_astra and has_nvidia, "ASTRA GPU not installed")
    def test_aqdata_full_origin_astra(self):
        """
        This test slices a sinogram. It then uses that geometry for the forward projection.

        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry
        ag.set_labels(['vertical','angle','horizontal'])
        ag.config.panel.origin='top-right'

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


    def test_process_acquisition(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        geometry = AcquisitionGeometry.create_Parallel3D().set_angles([0,90]).set_panel([4,3])
        data_in = AcquisitionData(arr, False, geometry)

        roi = {'vertical':(None,None,2),'angle':(None,None,2),'horizontal':(None,None,2)}
        proc = Slicer(roi)

        geometry_gold = AcquisitionGeometry.create_Parallel3D(detector_position=[-0.5,  0. ,  0. ]).set_angles([0]).set_panel([2,2],[2,2])
        data_gold = numpy.squeeze(data_in.array[::2,::2,::2])

        proc.set_input(data_in.geometry)
        geometry_out = proc.process()
        self.assertEqual(geometry_out, geometry_gold,
        msg="Slicer failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_out, geometry_gold))

        proc.set_input(data_in)
        data_out = proc.process()

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Slicer failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))

        data_out.fill(0)
        proc.process(out=data_out)

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Slicer failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))


    def test_process_image(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        geometry = ImageGeometry(4,3,2)
        data_in = ImageData(arr, False, geometry)

        roi = {'vertical':(None,None,2),'horizontal_y':(None,None,2),'horizontal_x':(None,None,2)}
        proc = Slicer(roi)

        geometry_gold = ImageGeometry(2,2,1, 2,2,2, -0.5,0, -0.5)
        data_gold = numpy.squeeze(data_in.array[::2,::2,::2])

        proc.set_input(data_in.geometry)
        geometry_out = proc.process()
        self.assertEqual(geometry_out, geometry_gold,
        msg="Slicer failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_out, geometry_gold))

        proc.set_input(data_in)
        data_out = proc.process()

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Slicer failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))

        data_out.fill(0)
        proc.process(out=data_out)

        numpy.testing.assert_array_equal(data_gold, data_out.array)
        self.assertEqual(data_out.geometry, geometry_gold,
        msg="Slicer failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, geometry_gold))


    def test_process_data_container(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        data_in = DataContainer(arr,False)

        #default labels
        roi = {'dimension_00':(None,None,2),'dimension_01':(None,None,2),'dimension_02':(None,None,2)}

        data_gold = numpy.squeeze(data_in.array[::2,::2,::2])
        proc = Slicer(roi)
        proc.set_input(data_in)
        data_out = proc.process()
        numpy.testing.assert_array_equal(data_gold, data_out.array)

        data_out.fill(0)
        proc.process(out=data_out)
        numpy.testing.assert_array_equal(data_gold, data_out.array)

        # custom labels
        data_in = DataContainer(arr,False,['LABEL_A','LABEL_B','LABEL_C'])

        roi = {'LABEL_A':(None,None,2),'LABEL_B':(None,None,2),'LABEL_C':(None,None,2)}

        proc = Slicer(roi)
        proc.set_input(data_in)
        data_out = proc.process()
        numpy.testing.assert_array_equal(data_gold, data_out.array)

class TestCofR_xcorrelation(unittest.TestCase):
    def setUp(self):
        data_raw = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        self.data_DLS = data_raw.log()
        self.data_DLS *= -1

        self.angles_list = [numpy.array([590, 0, 2, -310.5, 1]),
                       numpy.array([0.5, 10, 20, 180, 1]),
                       numpy.array([-360,-300, -240, -180.5, 0]),
                       numpy.array([-0.5, -90, -100, -179.5, 1])]
        
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(self.angles_list[0]).set_panel((2,2))
        ad = ag.allocate()
        ad.fill(numpy.ones([len(self.angles_list[0]), 2, 2]))
        self.data_test = ad

    def test_CofR_xcorrelation_init(self):
        # test default values are set
        processor = CofR_xcorrelation()
        self.assertEqual(processor.slice_index, 'centre')
        self.assertEqual(processor.projection_index, 0)
        self.assertEqual(processor.ang_tol, 0.1)

        # test non-default values are set
        processor = CofR_xcorrelation(11, 12, 13)
        self.assertEqual(processor.slice_index, 11)
        self.assertEqual(processor.projection_index, 12)
        self.assertEqual(processor.ang_tol, 13)

    def test_CofR_xcorrelation_check_input(self):
        
        # test default values
        processor = CofR_xcorrelation()
        processor.set_input(self.data_DLS)
        
        # test there is no error when slice_index is specified with different values in range
        slice_indices = [0, self.data_DLS.get_dimension_size('vertical')-1]
        for slice_index in slice_indices:
            processor = CofR_xcorrelation(slice_index=slice_index)
            processor.check_input(self.data_DLS)

        # test there is an error when slice index is specified with an un-recognised string, numbers out of range, or list
        slice_indices = ['a', -10, self.data_DLS.get_dimension_size('vertical'), [0,1]]
        for slice_index in slice_indices:
            with self.assertRaises(ValueError):
                processor = CofR_xcorrelation(slice_index=slice_index)
                processor.check_input(self.data_DLS)
        
        # test an error is raised passing string or out of range indices to projection index 
        projection_indices = ['a', [0, 'b'], -1, [self.data_DLS.get_dimension_size('angle'), 0]]
        for projection_index in projection_indices:
            with self.assertRaises(ValueError):
                processor = CofR_xcorrelation(projection_index=projection_index)
                processor.check_input(self.data_DLS)

        # test there is no error when an angle can be found 180 degrees +/- tolerance from the projection_index,
        processor = CofR_xcorrelation(projection_index=0, ang_tol=1)
        for angles in self.angles_list:
            self.data_test.geometry.set_angles(angles)
            processor.check_input(self.data_test)
        
        # test there is an error when no angle can be found 180 degrees +/- tolerance from the projection_index  
        processor = CofR_xcorrelation(projection_index=0, ang_tol=0.1)
        for angles in self.angles_list:
            self.data_test.geometry.set_angles(angles)
            with self.assertRaises(ValueError):
                processor.check_input(self.data_test)

        # test there is no error when projection indices are specified as list 180 degrees apart within tolerance
        processor = CofR_xcorrelation(projection_index=[0,3], ang_tol=1)
        for angles in self.angles_list:
            self.data_test.geometry.set_angles(angles)
            processor.check_input(self.data_test)

        # test there is no error when projection indices are specified as tuple 180 degrees apart within tolerance
        processor = CofR_xcorrelation(projection_index=(0,3), ang_tol=1)
        for angles in self.angles_list:
            self.data_test.geometry.set_angles(angles)
            processor.check_input(self.data_test)

        # test there is an error when projection indices are specified as >180 degrees apart within tolerance
        processor = CofR_xcorrelation(projection_index=[0,1], ang_tol=1)
        for angles in self.angles_list:
            self.data_test.geometry.set_angles(angles)
            with self.assertRaises(ValueError):
                processor.check_input(self.data_test)

        # test there is an error when more than 2 projection indices are specified
        processor = CofR_xcorrelation(projection_index=[0,3,1], ang_tol=1)
        for angles in self.angles_list:
            self.data_test.geometry.set_angles(angles)
            with self.assertRaises(ValueError):
                processor.check_input(self.data_test)

    def test_CofR_xcorrelation_return_180_index(self):
        # test finding angle 180 degrees apart to correlate with
        for angles in self.angles_list:
            index_180 = CofR_xcorrelation._return_180_index(angles, 0)
            self.assertEqual(angles[3], angles[index_180])

    def test_CofR_xcorrelation_process(self):
        # test processor returns expected output with DLS data
        processor = CofR_xcorrelation(projection_index=0, ang_tol=1)
        processor.set_input(self.data_DLS)
        data_out = processor.process()
        self.assertAlmostEqual(6.33, data_out.geometry.config.system.rotation_axis.position[0],places=2) 
        
        # test processor returns expected output with DLS data using out
        data = self.data_DLS
        processor = CofR_xcorrelation(projection_index=0, ang_tol=1)
        processor.set_input(data)
        processor.process(out = data)
        self.assertAlmostEqual(6.33, data.geometry.config.system.rotation_axis.position[0],places=2) 

        # test processor returns expected (but less accurate) output with DLS data with limited angles
        data_limited = Slicer(roi={'angle': ((abs(self.data_DLS.geometry.angles+86)).argmin(), (abs(self.data_DLS.geometry.angles-92)).argmin(), 1)})(self.data_DLS)
        processor = CofR_xcorrelation(slice_index = 'centre', projection_index = 0, ang_tol=5)
        processor.set_input(data_limited) 
        processor.get_output(out=data_limited)
        self.assertAlmostEqual(6.33, data_limited.geometry.config.system.rotation_axis.position[0],places=0) 

        # test there is an error when the target angle is not within tolerance
        processor = CentreOfRotationCorrector.xcorrelation(slice_index = 'centre', projection_index = 0, ang_tol=1)
        with self.assertRaises(ValueError):
            processor.set_input(data_limited)           


class TestCentreOfRotation_cone3D_Flex(unittest.TestCase):

    def setUp(self):
        source_position_set=[[0,-100000,0]]
        detector_position_set=[[0,0,0]]
        detector_direction_x_set=[[1, 0, 0]]
        detector_direction_y_set=[[0, 0, 1]]
        ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([128,64],[0.1,0.2]).set_channels(4)
        self.data = ag.allocate('random')

    def test_image_sharpness_acquisition_geometry_cone3DFlex(self):
        #mock the _configure_FBP method to bypass the backprojector setup
        with patch.object(CofR_image_sharpness, '_configure_FBP', return_value=None):
            corr = CofR_image_sharpness()

        with self.assertRaises(ValueError):
            corr.set_input(self.data)

    def test_x_corr_acquisition_geometry_cone3DFlex(self):
        corr = CofR_xcorrelation()

        with self.assertRaises(ValueError):
            corr.set_input(self.data)


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


class TestPadder(unittest.TestCase):

    def setUp(self):

        self.ag = AcquisitionGeometry.create_Parallel3D(detector_position=[-0.1, 0.,-0.2]).set_angles([0,90,180,270]).set_panel([16,16],[0.1,0.1]).set_channels(4)
        self.ag_pad_width = {'channel':(1,2),'vertical':(3,4),'horizontal':(5,6)}
        self.ag_padded = AcquisitionGeometry.create_Parallel3D(detector_position=[-0.05, 0., -0.15]).set_angles([0,90,180,270]).set_panel([27,23],[0.1,0.1]).set_channels(7)

        self.ag2 = AcquisitionGeometry.create_Parallel3D(detector_position=[-0.1, 0.,-0.2]).set_angles([0,90,180,270]).set_panel([16,16],[0.1,0.1],origin='top-right').set_channels(4)
        self.ag2_padded = AcquisitionGeometry.create_Parallel3D(detector_position=[-0.15, 0., -0.25]).set_angles([0,90,180,270]).set_panel([27,23],[0.1,0.1],origin='top-right').set_channels(7)

        self.ig = ImageGeometry(5,4,3,center_x=0.5,center_y=1,center_z=-0.5,channels=2)
        self.ig_pad_width = {'channel':(1,2),'vertical':(3,2),'horizontal_x':(2,1), 'horizontal_y':(2,3)}
        self.ig_padded = ImageGeometry(8,9,8,center_x=0,center_y=1.5,center_z=-1, channels=5)


        arr_in = numpy.arange(9, dtype=numpy.float32).reshape(3,3)
        ig = ImageGeometry(3,3)
        self.data_test = ImageData(arr_in, True, ig)


    def test_set_up(self):

        ig = ImageGeometry(20,22,23,0.1,0.2,0.3,0.4,0.5,0.6,channels=24)
        data = ig.allocate('random', seed=42)
        dim_order = ['channel','vertical','horizontal_y','horizontal_x']

        pad_width = {'horizontal_x':(3,4),'vertical':(5,6),'channel':(7,8)}
        pad_values= {'horizontal_x':(0.3,0.4),'vertical':(0.5,0.6),'channel':(0.7,0.8)}


        # check inputs
        proc = Padder('constant', pad_width=2, pad_values=0.1)
        proc.set_input(data)
        proc._set_up()
        self.assertListEqual(list(data.dimension_labels), proc._labels_in)
        self.assertListEqual(list(data.shape), proc._shape_in)

        #expected outputs
        gold_width_default = [(0,0),(2,2),(2,2),(2,2)]
        gold_value_default = [(0,0),(0.1,0.1),(0.1,0.1),(0.1,0.1)]
        gold_processed_dims_default = [0,1,1,1]
        gold_shape_out_default = numpy.array(data.shape) + [0,4,4,4]

        gold_width_tuple = [(0,0),(1,2),(1,2),(1,2)]
        gold_value_tuple = [(0,0),(0.1,0.2),(0.1,0.2),(0.1,0.2)]
        gold_shape_out_tuple = numpy.array(data.shape) + [0,3,3,3]

        gold_width_dict = [pad_width.get(x,(0,0)) for x in dim_order]
        gold_value_dict = [pad_values.get(x,(0,0)) for x in dim_order]
        gold_processed_dims_dict = [1,1,0,1]
        gold_shape_out_dict = numpy.array(data.shape) + [7+8,5+6,0,3+4]

        # check pad_width set-up
        proc = Padder('constant', pad_width=2, pad_values=0.1)
        proc.set_input(data)
        proc._set_up()
        self.assertListEqual(gold_width_default, proc._pad_width_param)
        self.assertListEqual(gold_value_default, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_default, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_default, proc._shape_out)

        proc = Padder('constant', pad_width=(1,2), pad_values=0.1)
        proc.set_input(data)
        proc._set_up()
        self.assertListEqual(gold_width_tuple, proc._pad_width_param)
        self.assertListEqual(gold_value_default, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_default, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_tuple, proc._shape_out)

        proc = Padder('constant', pad_width=pad_width, pad_values=0.1)
        proc.set_input(data)
        proc._set_up()
        gold_value_dict_custom = [(0,0) if x == (0,0) else (0.1,0.1)  for x in gold_width_dict]
        self.assertListEqual(gold_width_dict, proc._pad_width_param)
        self.assertListEqual(gold_value_dict_custom, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_dict, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_dict, proc._shape_out)

        # check pad_value set-up
        proc = Padder('constant', pad_width=2, pad_values=(0.1,0.2))
        proc.set_input(data)
        proc._set_up()
        self.assertListEqual(gold_width_default, proc._pad_width_param)
        self.assertListEqual(gold_value_tuple, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_default, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_default, proc._shape_out)

        proc = Padder('constant', pad_width=2, pad_values=pad_values)
        proc.set_input(data)
        proc._set_up()
        gold_width_dict_custom = [(0,0) if x == (0,0) else (2,2)  for x in gold_value_dict]
        gold_shape_out_dict_custom = numpy.array(data.shape) + [4,4,0,4]
        self.assertListEqual(gold_width_dict_custom, proc._pad_width_param)
        self.assertListEqual(gold_value_dict, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_dict, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_dict_custom, proc._shape_out)

        proc = Padder('constant', pad_width=pad_width, pad_values=(0.1,0.2))
        proc.set_input(data)
        proc._set_up()
        gold_value_dictionary_custom = [(0,0) if x == (0,0) else (0.1,0.2)  for x in gold_width_dict]
        self.assertListEqual(gold_width_dict, proc._pad_width_param)
        self.assertListEqual(gold_value_dictionary_custom, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_dict, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_dict, proc._shape_out)

        proc = Padder('constant', pad_width=pad_width, pad_values=pad_values)
        proc.set_input(data)
        proc._set_up()
        self.assertListEqual(gold_width_dict, proc._pad_width_param)
        self.assertListEqual(gold_value_dict, proc._pad_values_param)
        self.assertListEqual(gold_processed_dims_dict, proc._processed_dims)
        numpy.testing.assert_array_equal(gold_shape_out_dict, proc._shape_out)

        proc = Padder('constant', pad_width=pad_width, pad_values={'horizontal_x':(0.5,0.6)})
        # raise an error as not all axes values defined
        with self.assertRaises(ValueError):
            proc.set_input(data)
            proc._set_up()


    def test_process_acquisition_geometry(self):

        geometry = self.ag

        proc = Padder('constant', pad_width=self.ag_pad_width, pad_values=0.0)
        proc.set_input(geometry)
        geometry_padded = proc._process_acquisition_geometry()

        self.assertEqual(geometry_padded, self.ag_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_padded, self.ag_padded))


        proc = Padder('constant', pad_width={'angle':5}, pad_values=0.0)
        proc.set_input(geometry)
        geometry_padded = proc._process_acquisition_geometry()

        geometry_gold = geometry.copy()
        geometry_gold.config.angles.angle_data = [\
            -450., -360., -270., -180.,  -90.,\
            0.,   90.,  180.,  270.,\
            360., 450.,  540., 630., 720.]

        self.assertEqual(geometry_padded, geometry_gold,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_padded, geometry_gold))

    def test_process_acquisition_geometry_cone3DFlex(self):
 
        source_position_set=[[0,-100000,0]]
        detector_position_set=[[0,0,0]]
        detector_direction_x_set=[[1, 0, 0]]
        detector_direction_y_set=[[0, 0, 1]]
        ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([128,64],[0.1,0.2]).set_channels(4)

        proc = Padder('constant', pad_width=self.ag_pad_width, pad_values=0.0)

        with self.assertRaises(NotImplementedError):
            proc.set_input(ag)

    def test_process_acquisition_geometry_origin(self):
        geometry = self.ag2

        proc = Padder('constant', pad_width=self.ag_pad_width, pad_values=0.0)
        proc.set_input(geometry)
        geometry_padded = proc._process_acquisition_geometry()

        self.assertEqual(geometry_padded, self.ag2_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_padded, self.ag2_padded))


    def test_process_image_geometry(self):

        geometry = self.ig

        proc = Padder('constant', pad_width=self.ig_pad_width, pad_values=0.5)
        proc.set_input(geometry)
        geometry_padded = proc._process_image_geometry()

        self.assertEqual(geometry_padded, self.ig_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_padded, self.ig_padded))


    def test_process_data(self):

        geometry = self.ig
        data = geometry.allocate('random', seed=42)

        proc = Padder('constant', pad_width=self.ig_pad_width, pad_values=0.5)
        proc.set_input(data)
        arr_padded = proc._process_data(data)

        c = self.ig_pad_width['channel']
        v = self.ig_pad_width['vertical']
        hy = self.ig_pad_width['horizontal_y']
        hx = self.ig_pad_width['horizontal_x']

        data_gold = self.ig_padded.allocate(0.5)
        data_gold.array[c[0]:-c[1],v[0]:-v[1],hy[0]:-hy[1],hx[0]:-hx[1]] = data.array

        numpy.testing.assert_array_equal(data_gold.array, arr_padded)


    def test_process_acquisition(self):

        c = self.ag_pad_width['channel']
        v = self.ag_pad_width['vertical']
        h = self.ag_pad_width['horizontal']

        proc = Padder('constant', pad_width=self.ag_pad_width, pad_values=0.5)

        data_in = self.ag.allocate('random', seed=42)

        data_gold = self.ag_padded.allocate(0.5)
        data_gold.array[c[0]:-c[1],:,v[0]:-v[1],h[0]:-h[1]] = data_in.array

        proc.set_input(data_in.geometry)
        geometry_out = proc.process()
        self.assertEqual(geometry_out, self.ag_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_out, self.ag_padded))

        proc.set_input(data_in)
        data_out = proc.process()

        numpy.testing.assert_array_equal(data_gold.array, data_out.array)
        self.assertEqual(data_out.geometry, self.ag_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, self.ag_padded))

        data_out.fill(0)
        proc.process(out=data_out)

        numpy.testing.assert_array_equal(data_gold.array, data_out.array)
        self.assertEqual(data_out.geometry, self.ag_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, self.ag_padded))


    def test_process_image(self):

        c = self.ig_pad_width['channel']
        v = self.ig_pad_width['vertical']
        hy = self.ig_pad_width['horizontal_y']
        hx = self.ig_pad_width['horizontal_x']

        proc = Padder('constant', pad_width=self.ig_pad_width, pad_values=0.5)

        data_in = self.ig.allocate('random', seed=42)

        data_gold = self.ig_padded.allocate(0.5)
        data_gold.array[c[0]:-c[1],v[0]:-v[1],hy[0]:-hy[1],hx[0]:-hx[1]] = data_in.array

        proc.set_input(data_in.geometry)
        geometry_out = proc.process()
        self.assertEqual(geometry_out, self.ig_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(geometry_out, self.ig_padded))

        proc.set_input(data_in)
        data_out = proc.process()

        numpy.testing.assert_array_equal(data_gold.array, data_out.array)
        self.assertEqual(data_out.geometry, self.ig_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, self.ig_padded))

        data_out.fill(0)
        proc.process(out=data_out)

        numpy.testing.assert_array_equal(data_gold.array, data_out.array)
        self.assertEqual(data_out.geometry, self.ig_padded,
        msg="Padder failed with geometry mismatch. Got:\n{0}\nExpected:\n{1}".format(data_out.geometry, self.ig_padded))


    def test_process_data_container(self):

        arr=numpy.arange(24,dtype=numpy.float32).reshape(2,3,4)
        data_in = DataContainer(arr,False)

        #default labels
        pad_width = {'dimension_00':(1,2),'dimension_01':(2,3),'dimension_02':(4,3)}
        a = pad_width['dimension_00']
        b = pad_width['dimension_01']
        c = pad_width['dimension_02']

        shape_out = numpy.array(arr.shape) + [a[0]+a[1],b[0]+b[1],c[0]+c[1]]
        data_gold = numpy.ones(shape_out,dtype=numpy.float32) * 0.5
        data_gold[a[0]:-a[1],b[0]:-b[1],c[0]:-c[1]] = data_in.array

        proc = Padder('constant', pad_width=pad_width, pad_values=0.5)
        proc.set_input(data_in)
        data_out = proc.process()
        numpy.testing.assert_array_equal(data_gold, data_out.array)

        data_out.fill(0)
        proc.process(out=data_out)
        numpy.testing.assert_array_equal(data_gold, data_out.array)

        # custom labels
        data_in = DataContainer(arr,False,['LABEL_A','LABEL_B','LABEL_C'])

        pad_width = {'LABEL_A':(1,2),'LABEL_B':(2,3),'LABEL_C':(4,3)}

        proc = Padder('constant', pad_width=pad_width, pad_values=0.5)
        proc.set_input(data_in)
        data_out = proc.process()
        numpy.testing.assert_array_equal(data_gold, data_out.array)


    def test_results_constant(self):

        """
        in:
        0 1 2
        3 4 5
        6 7 8

        out:
        v v v v v
        v 0 1 2 v
        v 3 4 5 v
        v 6 7 8 v
        v v v v v
        """
        value = 0.5
        width = 1
        proc = Padder.constant(pad_width=width, constant_values=value)
        proc.set_input(self.data_test)
        data_out = proc.get_output()

        shape_padded = (3+width*2, 3+width*2)
        arr_gold = numpy.ones((shape_padded), dtype=numpy.float32) * value
        arr_gold[width:-width,width:-width] = self.data_test.array

        numpy.testing.assert_array_equal(arr_gold, data_out.array)


    def test_results_edge(self):

        """
        in:
        0 1 2
        3 4 5
        6 7 8

        out:
        0 0 1 2 2
        0 0 1 2 2
        3 3 4 5 5
        6 6 7 8 8
        6 6 7 8 8
        """

        width = 1
        proc = Padder.edge(pad_width=width)
        proc.set_input(self.data_test)
        data_out = proc.get_output()

        shape_padded = (3+width*2, 3+width*2)
        arr_gold = numpy.zeros((shape_padded), dtype=numpy.float32)
        arr_gold[width:-width,width:-width] = self.data_test.array
        arr_gold[0,:] = [0,0,1,2,2]
        arr_gold[-1,:] = [6,6,7,8,8]
        arr_gold[:,0] = [0,0,3,6,6]
        arr_gold[:,-1] = [2,2,5,8,8]

        numpy.testing.assert_array_equal(arr_gold, data_out.array)


    def test_results_linear_ramp(self):
        """
        in:
        0 1 2
        3 4 5
        6 7 8

        out:
        v   v           v           v           v                   v
        v   0           1           2           (v+2)/2             v
        v   3           4           5           (v+5)/2             v
        v   6           7           8           (v+8)/2             v
        v   (v+6)/2     (v+7)/2     (v+8)/2     ((v+8)/2 + v)/2     v
        v   v           v           v           v                   v
        """
        value = 0.5
        width = (1,2)
        proc = Padder.linear_ramp(pad_width=width, end_values=value)
        proc.set_input(self.data_test)
        data_out = proc.get_output()

        shape_padded = (3+width[0]+width[1], 3+width[0]+width[1])
        arr_gold = numpy.ones((shape_padded), dtype=numpy.float32) * value
        arr_gold[width[0]:-width[1],width[0]:-width[1]] = self.data_test.array

        arr_gold[-2,:] = (arr_gold[-3,:] + arr_gold[-1,:])/2
        arr_gold[:,-2] = (arr_gold[:,-3] + arr_gold[:,-1])/2

        numpy.testing.assert_array_equal(arr_gold, data_out.array)


    def test_results_reflect(self):
        """
        in:
        0 1 2
        3 4 5
        6 7 8

        out:
        4 3 4 5 4
        1 0 1 2 1
        4 3 4 5 4
        7 6 7 8 7
        4 3 4 5 4
        """

        width = 1
        proc = Padder.reflect(pad_width=width)
        proc.set_input(self.data_test)
        data_out = proc.get_output()

        shape_padded = (3+width*2, 3+width*2)
        arr_gold = numpy.zeros((shape_padded), dtype=numpy.float32)
        arr_gold[width:-width,width:-width] = self.data_test.array
        arr_gold[0,:] = [4,3,4,5,4]
        arr_gold[-1,:] = [4,3,4,5,4]
        arr_gold[:,0] = [4,1,4,7,4]
        arr_gold[:,-1] = [4,1,4,7,4]

        numpy.testing.assert_array_equal(arr_gold, data_out.array)

    def test_results_symmetric(self):
        """
        in:
        0 1 2
        3 4 5
        6 7 8

        out:
        0 0 1 2 2
        0 0 1 2 2
        3 3 4 5 5
        6 6 7 8 8
        6 6 7 8 8
        """

        width = 1
        proc = Padder.symmetric(pad_width=width)
        proc.set_input(self.data_test)
        data_out = proc.get_output()

        shape_padded = (3+width*2, 3+width*2)
        arr_gold = numpy.zeros((shape_padded), dtype=numpy.float32)
        arr_gold[width:-width,width:-width] = self.data_test.array
        arr_gold[0,:] = [0,0,1,2,2]
        arr_gold[-1,:] = [6,6,7,8,8]
        arr_gold[:,0] = [0,0,3,6,6]
        arr_gold[:,-1] = [2,2,5,8,8]

        numpy.testing.assert_array_equal(arr_gold, data_out.array)


    def test_results_wrap(self):
        """
        in:
        0 1 2
        3 4 5
        6 7 8

        out:
        8 6 7 8 6
        2 0 1 2 0
        5 3 4 5 3
        8 6 7 8 6
        2 0 1 2 0
        """

        width = 1
        proc = Padder.wrap(pad_width=width)
        proc.set_input(self.data_test)
        data_out = proc.get_output()

        shape_padded = (3+width*2, 3+width*2)
        arr_gold = numpy.zeros((shape_padded), dtype=numpy.float32)
        arr_gold[width:-width,width:-width] = self.data_test.array
        arr_gold[0,:] = [8,6,7,8,6]
        arr_gold[-1,:] = [2,0,1,2,0]
        arr_gold[:,0] = [8,2,5,8,2]
        arr_gold[:,-1] = [6,0,3,6,0]

        numpy.testing.assert_array_equal(arr_gold, data_out.array)


    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_pad_ad_full_tigre(self):
        """
        This test pads a acquisition data asymmetrically.
        It then compares the FBP of the padded and unpadded data on the same ImageGeometry.
        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

        data.log(out=data)
        data *=-1

        recon_orig = FBP(data).run(verbose=0)
        recon_orig.apply_circular_mask()

        proc = Padder('constant',pad_width=(5,40),pad_values=0.0)
        proc.set_input(data)
        data_padded = proc.get_output()

        recon_new = FBP(data_padded, recon_orig.geometry).run(verbose=0)
        recon_new.apply_circular_mask()

        numpy.testing.assert_allclose(recon_orig.array, recon_new.array, atol=1e-4)


        # switch panel origin
        data.geometry.config.panel.origin='top-right'

        recon_orig = FBP(data).run(verbose=0)
        recon_orig.apply_circular_mask()

        proc = Padder('constant',pad_width=(5,40),pad_values=0.0)
        proc.set_input(data)
        data_padded = proc.get_output()

        recon_new = FBP(data_padded, recon_orig.geometry).run(verbose=0)
        recon_new.apply_circular_mask()

        numpy.testing.assert_allclose(recon_orig.array, recon_new.array, atol=1e-4)


    @unittest.skipUnless(has_astra and has_nvidia, "ASTRA GPU not installed")
    def test_pad_id_full_astra(self):
        """
        This test pads an image data asymmetrically.
        It then compares the forward projection of the padded and unpadded phantom on the same AcquisitionGeometry.
        This ensures the offsets are correctly set and the same window of data is output in both cases.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry
        phantom = dataexample.SIMULATED_SPHERE_VOLUME.get()
        ag.set_labels(['vertical','angle','horizontal'])

        PO = AstraProjectionOperator(phantom.geometry, ag)
        fp_orig = PO.direct(phantom)

        proc = Padder('constant',pad_width={'vertical':(10,40),'horizontal_y':(40,10),'horizontal_x':(5,90)},pad_values=0.0)
        proc.set_input(phantom)
        phantom_padded = proc.get_output()

        PO = AstraProjectionOperator(phantom_padded.geometry, ag)
        fp_new = PO.direct(phantom_padded)

        numpy.testing.assert_allclose(fp_orig.array, fp_new.array, atol=1e-3)


    @unittest.skip
    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_pad_id_full_tigre(self):
        """
        This test pads an image data asymmetrically.
        It then compares the forward projection of the padded and unpadded phantom on the same AcquisitionGeometry.
        This ensures the offsets are correctly set and the same window of data is output in both cases.

        Tigre geometry bug means this does not pass.
        """

        ag = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().geometry
        phantom = dataexample.SIMULATED_SPHERE_VOLUME.get()

        PO = TigreProjectionOperator(phantom.geometry, ag)
        fp_orig = PO.direct(phantom)

        proc = Padder('constant',pad_width={'vertical':(10,40),'horizontal_y':(40,10),'horizontal_x':(5,90)},pad_values=0.0)
        proc.set_input(phantom)
        phantom_padded = proc.get_output()

        PO = TigreProjectionOperator(phantom_padded.geometry, ag)
        fp_new = PO.direct(phantom_padded)

        numpy.testing.assert_allclose(fp_orig.array, fp_new.array, atol=1e-3)


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
        AG = AcquisitionGeometry.create_Parallel3D().set_panel((10,10)).set_angles(1)

        data = IG.allocate('random', seed=42)

        data.as_array()[2,3] = float('inf')
        data.as_array()[4,5] = float('nan')

   
        data_as_image_data = data
        data_as_data_container = DataContainer(data.as_array().copy())
        data_as_acq_data = AcquisitionData(array=data.as_array().copy(), geometry=AG)

        data_objects = [data_as_image_data, data_as_data_container, data_as_acq_data]
        data_type_name = ['ImageData', 'DataContainer', 'AcquisitionData']

        for i, data in enumerate(data_objects):
            with self.subTest(data_type=data_type_name[i]):

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
                data.as_array()[2,3] = numpy.random.rand()
                data.as_array()[4,5] = numpy.random.rand()
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


        # Tests on larger data for checking mean and median
        IG = ImageGeometry(voxel_num_x=200,
                            voxel_num_y=200)

        AG = AcquisitionGeometry.create_Parallel3D().set_panel((200,200)).set_angles(1)
        data = IG.allocate('random', seed=2)
        data.as_array()[7,4] += 10 * numpy.std(data.as_array()[7,:])

        data_as_data_container = DataContainer(data.as_array().copy())
        data_as_image_data = data
        data_as_acq_data = AcquisitionData(array=data.as_array().copy(), geometry=AG)
        data_objects = [data_as_image_data, data_as_data_container, data_as_acq_data]

        for i, data in enumerate(data_objects):
            with self.subTest(data_type=data_type_name[i]):

                m = MaskGenerator.mean(axis=1) # this gives horizontal_x for ImageData, or 'dimension_01' for DataContainer
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
                m = MaskGenerator.median(axis=1)
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
                m = MaskGenerator.mean(window=20, axis=0) # this gives horizontal_y for ImageData, or 'dimension_00' for DataContainer
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

    def test_TransmissionAbsorptionConverter(self, accelerated=False):

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

        ad = AG.allocate('random', seed=42)

        s = TransmissionAbsorptionConverter(white_level=10, min_intensity=0.1,
                                            accelerated=accelerated)
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

    def test_TransmissionAbsorptionConverter_accelerated(self):
        self.test_TransmissionAbsorptionConverter(accelerated=True)

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

        ad = AG.allocate('random', seed=42)

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
        IG_2D = ImageGeometry(voxel_num_x=10,
                        voxel_num_y=10)
        IG_3D = ImageGeometry(voxel_num_x=5, 
                            voxel_num_y=5,
                            voxel_num_z=5)
        
        self.data_2D_init = IG_2D.allocate('random', seed=42)
        self.data_3D_init = IG_3D.allocate('random', seed=42)

        self.data_2D = self.data_2D_init.copy()
        self.data_3D = self.data_3D_init.copy()

        self.mask_coords_2D = [(2,3), (4,5)]
        self.mask_coords_3D = [(2,3,4), (3,1,2)]
        
        self.data_2D.as_array()[self.mask_coords_2D[0]] = float('inf')
        self.data_2D.as_array()[self.mask_coords_2D[1]] = float('nan')

        self.data_3D.as_array()[self.mask_coords_3D[0]] = float('inf')
        self.data_3D.as_array()[self.mask_coords_3D[1]] = float('nan')
        
        mask_2D_manual = numpy.ones((10,10), dtype=bool)
        mask_2D_manual[self.mask_coords_2D[0]] = 0
        mask_2D_manual[self.mask_coords_2D[1]] = 0

        mask_3D_manual = numpy.ones((5,5,5), dtype=bool)
        mask_3D_manual[self.mask_coords_3D[0]] = 0
        mask_3D_manual[self.mask_coords_3D[1]] = 0
        
        self.mask_2D_manual = DataContainer(mask_2D_manual, dimension_labels=self.data_2D.dimension_labels) 
        self.mask_2D_generated = MaskGenerator.special_values()(self.data_2D)

        self.mask_3D_manual = DataContainer(mask_3D_manual, dimension_labels=self.data_3D.dimension_labels)
        self.mask_3D_generated = MaskGenerator.special_values()(self.data_3D)

        # make a copy of mask_manual with 1s and 0s instead of bools:
        mask_int_manual = mask_2D_manual.astype(numpy.int32)
        self.mask_int_manual = DataContainer(mask_int_manual, dimension_labels=self.data_2D.dimension_labels)


    def test_Masker_Manual(self):
        self.Masker_check(self.mask_2D_manual, self.data_2D, self.data_2D_init, self.mask_coords_2D)
        self.Masker_check(self.mask_3D_manual, self.data_3D, self.data_3D_init, self.mask_coords_3D)

    def test_Masker_generated(self):
        self.Masker_check(self.mask_2D_generated, self.data_2D, self.data_2D_init, self.mask_coords_2D)
        self.Masker_check(self.mask_3D_generated, self.data_3D, self.data_3D_init, self.mask_coords_3D)

    def test_Masker_with_integer_mask(self):
        self.Masker_check(self.mask_int_manual, self.data_2D, self.data_2D_init, self.mask_coords_2D)

    def test_Masker_doesnt_modify_input_mask(self):
        mask = self.mask_2D_manual.copy()
        self.Masker_check(self.mask_2D_manual, self.data_2D, self.data_2D_init, self.mask_coords_2D)
        numpy.testing.assert_array_equal(mask.as_array(), self.mask_2D_manual.as_array())

        mask = self.mask_3D_manual.copy()
        self.Masker_check(self.mask_3D_manual, self.data_3D, self.data_3D_init, self.mask_coords_3D)
        numpy.testing.assert_array_equal(mask.as_array(), self.mask_3D_manual.as_array())



    def test_Masker_doesnt_modify_input_integer_mask(self):
        mask = self.mask_int_manual.copy()
        self.Masker_check(self.mask_int_manual, self.data_2D, self.data_2D_init, self.mask_coords_2D)
        numpy.testing.assert_array_equal(mask.as_array(), self.mask_int_manual.as_array())

    def Masker_check(self, mask, data, data_init, mask_coords): 

        # test value mode
        m = Masker.value(mask=mask, value=10)
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()

        for mask_coord in mask_coords:
            data_test[mask_coord] = 10
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)     
        
        # test mean mode
        m = Masker.mean(mask=mask)
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()
        tmp = numpy.sum(data_init.as_array())-numpy.sum([data_init.as_array()[i] for i in mask_coords])
        tmp /= (data_init.size - len(mask_coords))
        for mask_coord in mask_coords:
            data_test[mask_coord] = tmp

        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        
        # test median mode
        m = Masker.median(mask=mask)
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()
        tmp = data.as_array()[numpy.isfinite(data.as_array())]
        for mask_coord in mask_coords:
            data_test[mask_coord] = numpy.median(tmp)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)

        # test axis int
        m = Masker.median(mask=mask, axis=0)
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()

        for mask_coord in mask_coords:
            # get elements in mask_coord:
            if len(mask_coord) == 2:
                x, y = mask_coord
                tmp = data.as_array()[:,y][numpy.isfinite(data.as_array()[:,y])]
            else:
                x, y, z = mask_coord
                tmp = data.as_array()[:,y,z][numpy.isfinite(data.as_array()[:,y,z])]
            
            data_test[mask_coord] = numpy.median(tmp)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6) 
        
        # test axis str
        m = Masker.mean(mask=mask, axis=data.dimension_labels[1])
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()
        for mask_coord in mask_coords:
            # get elements in mask_coord:
            if len(mask_coord) == 2:
                x, y = mask_coord
                tmp = data.as_array()[x,:][numpy.isfinite(data.as_array()[x,:])]
            else:
                x, y, z = mask_coord
                tmp = data.as_array()[x,:,z][numpy.isfinite(data.as_array()[x,:,z])]
            data_test[mask_coord] = numpy.sum(tmp) / len(tmp)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)

        # test inline
        data = data_init.copy()
        m = Masker.value(mask=mask, value=10)
        m.set_input(data)
        m.process(out=data)

        data_test = data_init.copy().as_array()
        for mask_coord in mask_coords:
            data_test[mask_coord] = 10
        
        numpy.testing.assert_allclose(data.as_array(), data_test, rtol=1E-6) 
        
        # test mask numpy 
        data = data_init.copy()
        m = Masker.value(mask=mask.as_array(), value=10)
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()
        for mask_coord in mask_coords:
            data_test[mask_coord] = 10
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        
        # test interpolate
        data = data_init.copy()
        m = Masker.interpolate(mask=mask, method='linear', axis=0)
        m.set_input(data)
        res = m.process()

        data_test = data.copy().as_array()

        for mask_coord in mask_coords:
            if len(mask_coord) == 2:
                x, y = mask_coord
                data_test[mask_coord] = (data_test[x-1, y] + data_test[x+1, y]) / 2
            else:
                x, y, z = mask_coord
                data_test[mask_coord] = (data_test[x-1, y, z] + data_test[x+1, y, z]) / 2
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  


class TestPaganinProcessor(unittest.TestCase):

    def setUp(self):
        self.data_parallel = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data_cone = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(numpy.linspace(0,360,360,endpoint=False))\
            .set_panel([128,128],0.1)\
            .set_channels(4)

        self.data_multichannel = ag.allocate('random', seed=3)

        source_position_set=[[0,-100000,0]]
        detector_position_set=[[0,0,0]]
        detector_direction_x_set=[[1, 0, 0]]
        detector_direction_y_set=[[0, 0, 1]]
        cone_flex_ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([3,3])
        self.data_cone_flex  = cone_flex_ag.allocate('random', seed=3)

    def error_message(self,processor, test_parameter):
            return "Failed with processor " + str(processor) + " on test parameter " + test_parameter

    def test_PaganinProcessor_init(self):
        # test default values are initialised
        processor = PaganinProcessor()
        test_parameter = ['energy', 'wavelength', 'delta', 'beta', 'full_retrieval', 
                          'filter_type', 'pad', 'return_units']
        test_value = [40000, 1e2*(constants.h*constants.speed_of_light)/(40000*constants.electron_volt), 
                      1, 1e-2, True, 'paganin_method', 0, 'cm']

        for i in numpy.arange(len(test_value)):
            self.assertEqual(getattr(processor,test_parameter[i]), test_value[i], msg=self.error_message(processor, test_parameter[i]))

        # test non-default values are initialised
        processor = PaganinProcessor(1, 2, 3, 'keV', False, 'string', 19, 'mm')
        test_value = [3, 1e3*(constants.h*constants.speed_of_light)/(3000*constants.electron_volt), 1, 2, False, 'string', 19, 'mm']
        for i in numpy.arange(len(test_value)):
            self.assertEqual(getattr(processor,test_parameter[i]), test_value[i], msg=self.error_message(processor, test_parameter[i]))

        with self.assertRaises(ValueError):
            processor = PaganinProcessor(return_units='string')

    def test_PaganinProcessor_energy_to_wavelength(self):
        processor = PaganinProcessor()
        wavelength = processor._energy_to_wavelength(10, 'meV', 'mm')
        self.assertAlmostEqual(wavelength, 0.12398419)


    def test_PaganinProcessor_check_input(self):
        processor = PaganinProcessor()
        for data in [self.data_cone, self.data_parallel, self.data_multichannel]:
            processor.set_input(data)
            data2 = processor.get_input()
            numpy.testing.assert_allclose(data2.as_array(), data.as_array())

            # check there is an error when the wrong data type is input
            with self.assertRaises(TypeError):
                processor.set_input(data.geometry)

            with self.assertRaises(TypeError):
                processor.set_input(data.as_array())

            dc = DataContainer(data.as_array())
            with self.assertRaises(TypeError):
                processor.set_input(dc)

            # check with different data order
            data.reorder('astra')
            with self.assertRaises(ValueError):
                processor.set_input(data)
        with self.assertRaises(NotImplementedError):
            processor.set_input(self.data_cone_flex)


    def test_PaganinProcessor_set_geometry(self):
        processor = PaganinProcessor()
        data = self.data_cone
        # check there is an error when the data geometry does not have units
        processor.set_input(data)
        with self.assertRaises(ValueError):
            processor._set_geometry(data.geometry, None)
        
        # check there is no error when the geometry unit is provided
        data.geometry.config.units = 'um'
        processor._set_geometry(data.geometry, None)
        multiplier = 1e-4 # convert um to return units cm
        
        # check the processor finds the correct geometry values, scaled by the units
        self.assertAlmostEqual(processor.propagation_distance, data.geometry.dist_center_detector*multiplier, 
                         msg=self.error_message(processor, 'propagation_distance'))
        self.assertEqual(processor.magnification, data.geometry.magnification, 
                         msg=self.error_message(processor, 'magnification'))
        self.assertAlmostEqual(processor.pixel_size, data.geometry.pixel_size_h*multiplier, 
                         msg=self.error_message(processor, 'pixel_size'))
                
        # check there is an error when the data geometry does not have propagation distance, and it is not provided in override geometry
        processor.set_input(self.data_parallel)
        with self.assertRaises(ValueError):
            processor._set_geometry(self.data_parallel.geometry, None)
        
        # check override_geometry
        for data in [self.data_parallel, self.data_cone, self.data_multichannel]:
            processor.set_input(data)
            processor._set_geometry(self.data_cone.geometry, override_geometry={'propagation_distance':1,'magnification':2, 'pixel_size':3})
            
            self.assertEqual(processor.propagation_distance, 1, 
                            msg=self.error_message(processor, 'propagation_distance'))
            self.assertEqual(processor.magnification, 2, 
                            msg=self.error_message(processor, 'magnification'))
            self.assertEqual(processor.pixel_size, 3, 
                            msg=self.error_message(processor, 'pixel_size'))
        
        # check the processor goes back to values from geometry if the geometry over-ride is not passed
        processor.set_input(self.data_cone)
        processor._set_geometry(self.data_cone.geometry)
        self.assertAlmostEqual(processor.propagation_distance, self.data_cone.geometry.dist_center_detector*multiplier, 
                        msg=self.error_message(processor, 'propagation_distance'))
        self.assertEqual(processor.magnification, self.data_cone.geometry.magnification, 
                        msg=self.error_message(processor, 'magnification'))
        self.assertAlmostEqual(processor.pixel_size, self.data_cone.geometry.pixel_size_h*multiplier, 
                        msg=self.error_message(processor, 'pixel_size'))
        
        processor.set_input(self.data_parallel)
        with self.assertRaises(ValueError):
            processor._set_geometry(self.data_parallel.geometry)

        # check there is an error when the pixel_size_h and pixel_size_v are different
        self.data_parallel.geometry.pixel_size_h = 9
        self.data_parallel.geometry.pixel_size_h = 10
        with self.assertRaises(ValueError):
            processor._set_geometry(self.data_parallel.geometry, override_geometry={'propagation_distance':1})

    def test_PaganinProcessor_create_filter(self):
        image = self.data_cone.get_slice(angle=0).as_array()
        Nx, Ny = image.shape
        
        delta = 1
        beta = 2
        energy = 3
        processor =  PaganinProcessor(delta=delta, beta=beta, energy=energy, return_units='m')

        # check alpha and mu are calculated correctly
        wavelength = (constants.h*constants.speed_of_light)/(energy*constants.electron_volt)
        mu = 4.0*numpy.pi*beta/(wavelength)
        alpha = 60000*delta/mu/self.data_cone.geometry.magnification

        self.data_cone.geometry.config.units='m'
        processor.set_input(self.data_cone)
        processor._set_geometry(self.data_cone.geometry)
        processor.filter_Nx = Nx
        processor.filter_Ny = Ny
        processor._create_filter()
        
        self.assertEqual(processor.alpha, alpha, msg=self.error_message(processor, 'alpha'))
        self.assertEqual(processor.mu, mu, msg=self.error_message(processor, 'mu'))
        
        kx,ky = numpy.meshgrid( 
            numpy.arange(-Nx/2, Nx/2, 1, dtype=numpy.float64) * (2*numpy.pi)/(Nx*self.data_cone.geometry.pixel_size_h/self.data_cone.geometry.magnification),
            numpy.arange(-Ny/2, Ny/2, 1, dtype=numpy.float64) * (2*numpy.pi)/(Nx*self.data_cone.geometry.pixel_size_h/self.data_cone.geometry.magnification),
            sparse=False, 
            indexing='ij'
            )
        
        # check default filter is created with paganin_method
        filter =  ifftshift(1/(1. + alpha*(kx**2 + ky**2)))
        numpy.testing.assert_allclose(processor.filter, filter)

        # check generalised_paganin_method
        processor = PaganinProcessor(delta=delta, beta=beta, energy=energy, filter_type='generalised_paganin_method', return_units='m')
        processor.set_input(self.data_cone)
        processor._set_geometry(self.data_cone.geometry)
        processor.filter_Nx = Nx
        processor.filter_Ny = Ny
        processor._create_filter()
        filter = ifftshift(1/(1. - (2*alpha/(self.data_cone.geometry.pixel_size_h/self.data_cone.geometry.magnification)**2)*(numpy.cos(self.data_cone.geometry.pixel_size_h/self.data_cone.geometry.magnification*kx) + numpy.cos(self.data_cone.geometry.pixel_size_h/self.data_cone.geometry.magnification*ky) -2)))
        numpy.testing.assert_allclose(processor.filter, filter)

        # check unknown method raises error
        processor =  PaganinProcessor(delta=delta, beta=beta, energy=energy, filter_type='unknown_method', return_units='m')
        processor.set_input(self.data_cone)
        processor._set_geometry(self.data_cone.geometry)
        processor.filter_Nx = Nx
        processor.filter_Ny = Ny
        with self.assertRaises(ValueError):
            processor._create_filter()

        # check parameter override 
        processor =  PaganinProcessor(delta=delta, beta=beta, energy=energy, return_units='m')
        processor.set_input(self.data_cone)
        processor._set_geometry(self.data_cone.geometry)
        delta = 100
        beta=200
        processor.filter_Nx = Nx
        processor.filter_Ny = Ny
        processor._create_filter(override_filter={'delta':delta, 'beta':beta})
        
        # check alpha and mu are calculated correctly
        wavelength = (constants.h*constants.speed_of_light)/(energy*constants.electron_volt)
        mu = 4.0*numpy.pi*beta/(wavelength)
        alpha = 60000*delta/mu/self.data_cone.geometry.magnification
        self.assertEqual(processor.delta, delta, msg=self.error_message(processor, 'delta'))
        self.assertEqual(processor.beta, beta, msg=self.error_message(processor, 'beta'))
        self.assertEqual(processor.alpha, alpha, msg=self.error_message(processor, 'alpha'))
        self.assertEqual(processor.mu, mu, msg=self.error_message(processor, 'mu'))
        filter =  ifftshift(1/(1. + alpha*(kx**2 + ky**2)))
        numpy.testing.assert_allclose(processor.filter, filter)
        
        # test specifying alpha, delta and beta
        delta = 12
        beta = 13
        alpha = 14
        processor.filter_Nx = Nx
        processor.filter_Ny = Ny
        with self.assertLogs(level='WARN') as log:
            processor._create_filter(override_filter = {'delta':delta, 'beta':beta, 'alpha':alpha})
        wavelength = (constants.h*constants.speed_of_light)/(energy*constants.electron_volt)
        mu = 4.0*numpy.pi*beta/(wavelength)
        
        self.assertEqual(processor.delta, delta, msg=self.error_message(processor, 'delta'))
        self.assertEqual(processor.beta, beta, msg=self.error_message(processor, 'beta'))
        self.assertEqual(processor.alpha, alpha, msg=self.error_message(processor, 'alpha'))
        self.assertEqual(processor.mu, mu, msg=self.error_message(processor, 'mu'))
        filter =  ifftshift(1/(1. + alpha*(kx**2 + ky**2)))
        numpy.testing.assert_allclose(processor.filter, filter)

    def test_PaganinProcessor(self):

        wavelength = (constants.h*constants.speed_of_light)/(40000*constants.electron_volt)
        mu = 4.0*numpy.pi*1e-2/(wavelength)

        data_array = [self.data_cone, self.data_parallel, self.data_multichannel]
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        for data in data_array:
            data.geometry.config.units = 'm'
            data_abs = -(1/mu)*numpy.log(data)
            processor = PaganinProcessor(full_retrieval=True, return_units='m')
            processor.set_input(data)
            thickness = processor.get_output(override_geometry={'propagation_distance':1})
            self.assertLessEqual(quality_measures.mse(thickness, data_abs), 1e-5)
            processor = PaganinProcessor(full_retrieval=False)
            processor.set_input(data)
            filtered_image = processor.get_output(override_geometry={'propagation_distance':1})
            self.assertLessEqual(quality_measures.mse(filtered_image, data), 1e-5)

            # test with GPM
            processor = PaganinProcessor(full_retrieval=True, filter_type='generalised_paganin_method', return_units='m')
            processor.set_input(data)
            thickness = processor.get_output(override_geometry={'propagation_distance':1})
            self.assertLessEqual(quality_measures.mse(thickness, data_abs), 1e-5)
            processor = PaganinProcessor(full_retrieval=False, filter_type='generalised_paganin_method')
            processor.set_input(data)
            filtered_image = processor.get_output(override_geometry={'propagation_distance':1})
            self.assertLessEqual(quality_measures.mse(filtered_image, data), 1e-5)

            # test with padding
            processor = PaganinProcessor(full_retrieval=True, pad=10, return_units='m')
            processor.set_input(data)
            thickness = processor.get_output(override_geometry={'propagation_distance':1})
            self.assertLessEqual(quality_measures.mse(thickness, data_abs), 1e-5)
            processor = PaganinProcessor(full_retrieval=False, pad=10)
            processor.set_input(data)
            filtered_image = processor.get_output(override_geometry={'propagation_distance':1})
            self.assertLessEqual(quality_measures.mse(filtered_image, data), 1e-5)

            # test in-line
            thickness_inline = PaganinProcessor(full_retrieval=True, pad=10, return_units='m')(data, override_geometry={'propagation_distance':1})
            numpy.testing.assert_allclose(thickness.as_array(), thickness_inline.as_array())
            filtered_image_inline = PaganinProcessor(full_retrieval=False, pad=10)(data, override_geometry={'propagation_distance':1})
            numpy.testing.assert_allclose(filtered_image.as_array(), filtered_image_inline.as_array())
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def test_PaganinProcessor_2D(self):

        self.data_parallel.geometry.config.units = 'm'
        data_slice = self.data_parallel.get_slice(vertical=10)
        wavelength = (constants.h*constants.speed_of_light)/(40000*constants.electron_volt)
        mu = 4.0*numpy.pi*1e-2/(wavelength) 
        thickness = -(1/mu)*numpy.log(data_slice)

        processor = PaganinProcessor(pad=10)
        processor.set_input(data_slice)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        output = processor.get_output(override_geometry={'propagation_distance':1})
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.assertLessEqual(quality_measures.mse(output, thickness), 0.05)

    def test_PaganinProcessor_1angle(self):
        data = self.data_cone.get_slice(angle=1)
        data.geometry.config.units = 'm'
        wavelength = (constants.h*constants.speed_of_light)/(40000*constants.electron_volt)
        mu = 4.0*numpy.pi*1e-2/(wavelength) 
        thickness = -(1/mu)*numpy.log(data)

        processor = PaganinProcessor(pad=10)
        processor.set_input(data)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        output = processor.get_output(override_geometry={'propagation_distance':1})
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.assertLessEqual(quality_measures.mse(output, thickness), 0.05)

class TestLaminographyCorrector(unittest.TestCase):

    def setUp(self):
        self.data_parallel = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        # ag = AcquisitionGeometry.create_Parallel3D()\
        #     .set_angles(numpy.linspace(0,360,360,endpoint=False))\
        #     .set_panel([128,128],0.1)

        self.data_parallel.reorder('astra')

    def error_message(self, processor, test_parameter):
        return "Failed with processor " + str(processor) + " on test parameter " + test_parameter

    def test_LaminographyCorrector_init(self):
        # test default values are initialised
        processor = LaminographyCorrector()
        test_parameter = ['initial_parameters', 'parameter_bounds', 'parameter_tolerance', 
                          'coarse_binning', 'final_binning', 'angle_binning', 'reduced_volume', 'evaluations']
        test_value = [(30.0, 0.0), [(30, 40), (-10, 10)], (0.01, 0.01),
                      None, None, None, None, []]

        for i in numpy.arange(len(test_value)):
            self.assertEqual(getattr(processor, test_parameter[i]), test_value[i], 
                           msg=self.error_message(processor, test_parameter[i]))

        # test non-default values are initialised
        processor = LaminographyCorrector(initial_parameters=(25.0, 5.0), 
                                         parameter_bounds=[(20, 35), (-5, 15)],
                                         parameter_tolerance=(0.1, 0.1),
                                         coarse_binning=2,
                                         final_binning=1,
                                         angle_binning=2,
                                         reduced_volume=None)
        test_value = [(25.0, 5.0), [(20, 35), (-5, 15)], (0.1, 0.1),
                      2, 1, 2, None]

        for i in numpy.arange(len(test_value)):
            self.assertEqual(getattr(processor, test_parameter[i]), test_value[i],
                           msg=self.error_message(processor, test_parameter[i]))

    def test_LaminographyCorrector_check_input(self):
        processor = LaminographyCorrector()
        
        # test with parallel beam data - should work
        processor.set_input(self.data_parallel)
        data2 = processor.get_input()
        numpy.testing.assert_allclose(data2.as_array(), self.data_parallel.as_array())

        # check there is an error when the wrong data type is input
        with self.assertRaises(TypeError):
            processor.set_input(self.data_parallel.geometry)

        with self.assertRaises(TypeError):
            processor.set_input(self.data_parallel.as_array())

        dc = DataContainer(self.data_parallel.as_array())
        with self.assertRaises(TypeError):
            processor.set_input(dc)

        # check with different data order - should raise error
        data_reorder = self.data_parallel.copy()
        data_reorder.reorder('astra')
        data_reorder.reorder(['angle','horizontal','vertical'])
        with self.assertRaises(ValueError):
            processor.set_input(data_reorder)

        # # check that cone beam data raises NotImplementedError
        # with self.assertRaises(NotImplementedError):
        #     processor.set_input(self.data_cone)

        # # check that cone flex data raises NotImplementedError
        # with self.assertRaises(NotImplementedError):
        #     processor.set_input(self.data_cone_flex)

    def test_LaminographyCorrector_update_geometry(self):
        processor = LaminographyCorrector()
        
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_angles(numpy.linspace(0, numpy.pi, 10))
        ag.set_panel([512, 512], pixel_size=(1.0, 1.0))
        
        tilt_deg = 35.0
        cor_pix = 5.0
        
        ag_updated = processor.update_geometry(ag, tilt_deg, cor_pix)
        
        # Verify CoR was updated
        self.assertAlmostEqual(ag_updated.config.system.rotation_axis.position[0], cor_pix)
        
        # Verify rotation axis direction was modified (should be tilted)
        original_axis = numpy.array([0, 0, 1])
        tilted_axis = ag_updated.config.system.rotation_axis.direction
        assert not numpy.allclose(tilted_axis, original_axis)

class TestFluxNormaliser(unittest.TestCase):

    def setUp(self):
        self.data_parallel = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data_cone = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(numpy.linspace(0,360,360,endpoint=False))\
            .set_panel([128,128],0.1)\
            .set_channels(4)
        self.data_multichannel = ag.allocate('random', seed=42)
        self.data_slice = self.data_parallel.get_slice(vertical=1)
        self.data_reorder = self.data_cone.copy()
        self.data_reorder.reorder(['angle','horizontal','vertical'])
        self.data_single_angle = self.data_cone.get_slice(angle=1)
        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(numpy.linspace(0,3,3,endpoint=False))\
            .set_panel([3,3])
        arr = numpy.array([[[1,2,3],[1,2,3],[1,2,3]],
                        [[4,5,6],[4,5,6],[4,5,6]], 
                        [[7,8,9],[7,8,9],[7,8,9]]])
        self.data_simple = AcquisitionData(arr, geometry=ag)

        source_position_set=[[0,-100000,0]]*3
        detector_position_set=[[0,0,0]]*3
        detector_direction_x_set=[[1, 0, 0]]*3
        detector_direction_y_set=[[0, 0, 1]]*3
        cone_flex_ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set, detector_position_set, detector_direction_x_set, detector_direction_y_set).set_panel([3,3])
        self.cone_flex  = AcquisitionData(arr, geometry=cone_flex_ag)

    def error_message(self,processor, test_parameter):
            return "Failed with processor " + str(processor) + " on test parameter " + test_parameter

    def test_init(self):
        # test default values are initialised
        processor = FluxNormaliser()
        test_parameter = ['flux','roi','target']
        test_value = [None, None, 'mean']

        for i in numpy.arange(len(test_value)):
            self.assertEqual(getattr(processor, test_parameter[i]), test_value[i], msg=self.error_message(processor, test_parameter[i]))

        # test non-default values are initialised
        processor = FluxNormaliser(1,2,3)
        test_value = [1, 2, 3]
        for i in numpy.arange(len(test_value)):
            self.assertEqual(getattr(processor, test_parameter[i]), test_value[i], msg=self.error_message(processor, test_parameter[i]))

    def test_check_input(self):
        
        # check there is an error if no flux or roi is specified
        processor = FluxNormaliser()
        with self.assertRaises(ValueError):
            processor.check_input(self.data_cone)

        # check there's a not implemented error if cone flex geom is used:
        processor = FluxNormaliser(flux=[1,2,3])
        with self.assertRaises(NotImplementedError):
            processor.check_input(self.cone_flex)

    def test_calculate_flux(self):
        # check there is an error if flux array size is not equal to the number of angles in data
        processor = FluxNormaliser(flux = [1,2,3])
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor._calculate_flux()
        
        # check there is an error if roi is not specified as a dictionary
        processor = FluxNormaliser(roi='string')
        processor.set_input(self.data_cone)
        with self.assertRaises(TypeError):
            processor._calculate_flux()

        # check there is an error if roi is specified with float values
        processor = FluxNormaliser(roi={'horizontal':(1.5, 6.5)})
        processor.set_input(self.data_cone)
        with self.assertRaises(TypeError):
            processor._calculate_flux()

        # check there is an error if roi stop is greater than start
        processor = FluxNormaliser(roi={'horizontal':(10, 5)})
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor._calculate_flux()

        # check there is an error if roi stop is greater than the size of the axis
        processor = FluxNormaliser(roi={'horizontal':(0, self.data_cone.get_dimension_size('horizontal')+1)})
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor._calculate_flux()

        # check error raised with 0 flux
        processor = FluxNormaliser(flux = 0)
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor.get_output()

        processor = FluxNormaliser(flux = 0.0)
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor.get_output()

        processor = FluxNormaliser(flux=numpy.zeros(len(self.data_cone.geometry.angles)))
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor.get_output()

        processor = FluxNormaliser(flux=numpy.zeros(len(self.data_cone.geometry.angles), dtype=numpy.uint16))
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor.get_output()

    def test_calculate_target(self):

        # check target calculated with default method 'mean'
        processor = FluxNormaliser(flux=1)
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 1)

        processor = FluxNormaliser(flux=numpy.linspace(1,3,len(self.data_cone.geometry.angles)))
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 2)

        # check target calculated with method 'first'
        processor = FluxNormaliser(flux=1, target='first')
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 1)

        processor = FluxNormaliser(flux=numpy.linspace(1,3,len(self.data_cone.geometry.angles)),
                                   target='first')
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 1)

        # check target calculated with method 'last'
        processor = FluxNormaliser(flux=1, target='first')
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 1)

        processor = FluxNormaliser(flux=numpy.linspace(1,3,len(self.data_cone.geometry.angles)),
                                   target='last')
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 3)

        # check target calculated with float
        processor = FluxNormaliser(flux=1,
                                   target=55.0)
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        processor._calculate_target()
        self.assertAlmostEqual(processor.target_value, 55.0)

        # check error if target is an unrecognised string
        processor = FluxNormaliser(flux=1,
                                   target='string')
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        with self.assertRaises(ValueError):
            processor._calculate_target()

        # check error if target is not a string or floar
        processor = FluxNormaliser(flux=1,
                                   target={'string': 10})
        processor.set_input(self.data_cone)
        processor._calculate_flux()
        with self.assertRaises(TypeError):
            processor._calculate_target()
    
    @unittest.skipIf(not has_matplotlib, "matplotlib not installed")
    @patch('matplotlib.pyplot.show')
    def test_preview_configuration(self, mock_show):
        
        # Suppress backround range warning
        logging.disable(logging.CRITICAL)

        # Test error in preview configuration if there is no roi
        processor = FluxNormaliser(flux=10)
        processor.set_input(self.data_cone)
        with self.assertRaises(ValueError):
            processor.preview_configuration()
        
        # Test error in preview configuration if set_input not called
        roi = {'horizontal':(25,40)}
        processor = FluxNormaliser(roi=roi)
        with self.assertRaises(TypeError):
            processor.preview_configuration()

        # Test correct data is plotted
        roi = {'horizontal':(0,3),'vertical':(0,1)}
        processor = FluxNormaliser(roi=roi)
        processor.set_input(self.data_simple)
        
        fig = processor.preview_configuration()
        
        # Check slice plots
        slice_plot1 = fig.axes[0].images[0].get_array().data
        slice_plot2 = fig.axes[2].images[0].get_array().data
        numpy.testing.assert_allclose(self.data_simple.array[0], slice_plot1)
        numpy.testing.assert_allclose(self.data_simple.array[2], slice_plot2)

        # Check ROI plots
        for f in [fig.axes[0], fig.axes[2]]:
            roi_x_lower = f.lines[0].get_xdata()
            roi_x_upper = f.lines[1].get_xdata()
            roi_y_lower = f.lines[0].get_ydata()
            roi_y_upper = f.lines[1].get_ydata()
            numpy.testing.assert_allclose(roi_x_lower, [0,3])
            numpy.testing.assert_allclose(roi_x_upper, [0,3])
            numpy.testing.assert_allclose(roi_y_lower, [0,0])
            numpy.testing.assert_allclose(roi_y_upper, [1,1])

            roi_x_left = f.lines[2].get_xdata()
            roi_x_right = f.lines[3].get_xdata()
            roi_y_left = f.lines[2].get_ydata()
            roi_y_right = f.lines[3].get_ydata()
            numpy.testing.assert_allclose(roi_x_left, [0,0])
            numpy.testing.assert_allclose(roi_x_right, [3,3])
            numpy.testing.assert_allclose(roi_y_left, [0,1])
            numpy.testing.assert_allclose(roi_y_right, [0,1])

        # Check line data
        data_mean = self.data_simple.get_slice(vertical=1).array.mean(axis=1)
        plot_mean = fig.axes[4].lines[0].get_ydata()
        numpy.testing.assert_allclose(data_mean, plot_mean)

        data_min = self.data_simple.get_slice(vertical=1).array.min(axis=1)
        plot_min = fig.axes[4].lines[1].get_ydata()
        numpy.testing.assert_allclose(data_min, plot_min)

        data_max = self.data_simple.get_slice(vertical=1).array.max(axis=1)
        plot_max = fig.axes[4].lines[2].get_ydata()
        numpy.testing.assert_allclose(data_max, plot_max)

        # Test no error with preview_configuration with different data shapes
        for data in [self.data_cone, self.data_parallel, self.data_multichannel, 
                     self.data_slice, self.data_reorder, self.data_single_angle]:
            mock_show.reset_mock()

            roi = {'horizontal':(25,40)}
            processor = FluxNormaliser(roi=roi)
            processor.set_input(data)
            fig = processor.preview_configuration()
            
            mock_show.assert_called_once()

            # for 3D, check no error specifying a single angle to plot
            if data.geometry.dimension == '3D':
                processor.preview_configuration(angle=1)
            # if 2D, attempt to plot single angle should cause error
            else:
                with self.assertRaises(ValueError):
                    processor.preview_configuration(angle=1)

            # if data is multichannel, check no error specifying a single channel to plot
            if 'channel' in data.dimension_labels:
                processor.preview_configuration(angle=1, channel=1)
                processor.preview_configuration(channel=1)
            # if single channel, check specifying channel causes an error
            else:
                with self.assertRaises(ValueError):
                    processor.preview_configuration(channel=1)

        # Re-enable logging
        logging.disable(logging.NOTSET)

    def test_FluxNormaliser(self, accelerated=False):

        # Suppress backround range warning
        logging.disable(logging.CRITICAL)

        #Test flux with no target
        processor = FluxNormaliser(flux=1, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        numpy.testing.assert_allclose(data_norm.array, self.data_cone.array)
        
        #Test flux with target
        processor = FluxNormaliser(flux=10, target=5.0, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        numpy.testing.assert_allclose(data_norm.array, 0.5*self.data_cone.array)
        
        #Test flux array with no target
        flux = numpy.arange(1,2,(2-1)/(self.data_cone.get_dimension_size('angle')))
        processor = FluxNormaliser(flux=flux, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        data_norm_test = self.data_cone.copy()
        for a in range(data_norm_test.get_dimension_size('angle')):
            data_norm_test.array[a,:,:] /= flux[a]
            data_norm_test.array[a,:,:]*= numpy.mean(flux.ravel())
        numpy.testing.assert_allclose(data_norm.array, data_norm_test.array, atol=1e-6)

        # #Test flux array with target
        flux = numpy.arange(1,2,(2-1)/(self.data_cone.get_dimension_size('angle')))
        norm_value = 5.0
        processor = FluxNormaliser(flux=flux, target=norm_value, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        data_norm_test = self.data_cone.copy()
        for a in range(data_norm_test.get_dimension_size('angle')):
            data_norm_test.array[a,:,:] /= flux[a]
            data_norm_test.array[a,:,:]*= norm_value
        numpy.testing.assert_allclose(data_norm.array, data_norm_test.array, atol=1e-6)

        # #Test roi with no target
        roi = {'vertical':(0,10), 'horizontal':(0,10)}
        processor = FluxNormaliser(roi=roi, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        numpy.testing.assert_allclose(data_norm.array, self.data_cone.array)

        # #Test roi with norm_value
        roi = {'vertical':(0,10), 'horizontal':(0,10)}
        processor = FluxNormaliser(roi=roi, target=5.0, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        numpy.testing.assert_allclose(data_norm.array, 5*self.data_cone.array)

        # # Test roi with just one dimension
        roi = {'vertical':(0,2)}
        processor = FluxNormaliser(roi=roi, target=5, accelerated=accelerated)
        processor.set_input(self.data_cone)
        data_norm = processor.get_output()
        numpy.testing.assert_allclose(data_norm.array, 5*self.data_cone.array)

        # test roi with different data shapes and different flux values per projection
        for data in [ self.data_cone, self.data_parallel, self.data_multichannel,
                      self.data_slice, self.data_reorder]:
            roi = {'horizontal':(25,40)}
            processor = FluxNormaliser(roi=roi, target=5, accelerated=accelerated)
            processor.set_input(data)
            data_norm = processor.get_output()

            ax = data.get_dimension_axis('horizontal')
            slc = [slice(None)]*len(data.shape)
            slc[ax] = slice(25,40)
            axes=[ax]
            if 'vertical' in data.dimension_labels:
                axes.append(data.get_dimension_axis('vertical'))
            flux = numpy.mean(data.array[tuple(slc)], axis=tuple(axes))
            slice_proj = [slice(None)]*len(data.shape)
            proj_axis = data.get_dimension_axis('angle')
            data_norm_test = data.copy()
            h_size = data.get_dimension_size('horizontal')
            if 'vertical' in data.dimension_labels:
                v_size = data.get_dimension_size('vertical')
            else:
                v_size = 1
            proj_size = h_size*v_size
            for i in range(len(data.geometry.angles)*data.geometry.channels):
                data_norm_test.array.flat[i*proj_size:(i+1)*proj_size] /=flux.flat[i]
                data_norm_test.array.flat[i*proj_size:(i+1)*proj_size] *=5
            numpy.testing.assert_allclose(data_norm.array, data_norm_test.array, atol=1e-6, 
            err_msg='Flux Normaliser roi test failed with data shape: ' + str(data.shape) + ' and configuration:\n' + str(data.geometry.config.system))

        data = self.data_single_angle
        processor = FluxNormaliser(roi=roi, target=5, accelerated=accelerated)
        processor.set_input(data)
        data_norm = processor.get_output()
        ax = data.get_dimension_axis('horizontal')
        slc = [slice(None)]*len(data.shape)
        slc[ax] = slice(25,40)
        axes=[ax,data.get_dimension_axis('vertical')]
        flux = numpy.mean(data.array[tuple(slc)], axis=tuple(axes))

        numpy.testing.assert_allclose(data_norm.array, 5/flux*data.array, atol=1e-6, 
        err_msg='Flux Normaliser roi test failed with data shape: ' + str(data.shape) + ' and configuration:\n' + str(data.geometry.config.system))

        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    @unittest.skipUnless(has_numba, "Skipping because numba isn't installed")
    def test_FluxNormaliser_accelerated(self):
        self.test_FluxNormaliser(accelerated=True)

    def test_FluxNormaliser_preserves_input(self):
        processor = FluxNormaliser(flux=10, target=5.0)
        data = self.data_cone.copy()
        processor.set_input(data)
        data_norm = processor.get_output()
        numpy.testing.assert_allclose(data_norm.array, 0.5*self.data_cone.array)
        numpy.testing.assert_allclose(data.array, self.data_cone.array)

        processor = FluxNormaliser(flux=10, target=5.0)
        data = self.data_cone.copy()
        data_norm = self.data_cone.copy()
        processor.set_input(data)
        processor.get_output(out=data_norm)
        numpy.testing.assert_allclose(data_norm.array, 0.5*self.data_cone.array)
        numpy.testing.assert_allclose(data.array, self.data_cone.array)


class TestNormaliser(unittest.TestCase):

    def setUp(self):
        self.data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

    def test_normaliser_standard(self):
        flat_field = numpy.ones(self.data.shape[1::]) * 1.5
        dark_field = numpy.ones(self.data.shape[1::]) * 0.5
        normaliser = Normaliser(flat_field=flat_field, dark_field=dark_field)
        normalised_data = normaliser(self.data)
        test_out =  self.data.array-normalised_data.array
        numpy.testing.assert_allclose(test_out, 0.5)

        # test default with out
        out = self.data.geometry.allocate(None)
        normaliser(self.data, out=out)
        numpy.testing.assert_allclose(out.array, normalised_data.array)

        # test default inplace
        normaliser(self.data, out=self.data)
        numpy.testing.assert_allclose(self.data.array, normalised_data.array)

    def test_bad_input(self):
        arr = numpy.ones((5)) 
        normaliser = Normaliser(flat_field=arr, dark_field=None)
        normaliser.set_input(self.data)

        with self.assertRaises(ValueError):
            out = normaliser.get_output()

        arr = numpy.ones((5,6)) 
        normaliser.set_flat_field(arr)
        with self.assertRaises(ValueError):
            out = normaliser.get_output()      

        arr = numpy.ones((5,6)) 
        normaliser.set_dark_field(arr)
        normaliser.set_flat_field(None)
        with self.assertRaises(ValueError):
            out = normaliser.get_output()     

    def test_no_offset(self):
        #test with no dark-field
        flat_field = numpy.ones(self.data.shape[1::]) * 2.0
        dark_field = None
        normaliser = Normaliser(flat_field=flat_field, dark_field=dark_field)
        normalised_data = normaliser(self.data)
        test_out =  normalised_data.array/self.data.array
        numpy.testing.assert_allclose(test_out, 0.5)

        # test default with out
        out = self.data.geometry.allocate(None)
        normaliser(self.data, out=out)
        numpy.testing.assert_allclose(out.array, normalised_data.array)

        # test default inplace
        normaliser(self.data, out=self.data)
        numpy.testing.assert_allclose(self.data.array, normalised_data.array)

    def test_no_flat(self): 
        #test with no flat-field
        flat_field = None
        dark_field = numpy.ones(self.data.shape[1::]) * 0.5
        normaliser = Normaliser(flat_field=flat_field, dark_field=dark_field)
        normalised_data = normaliser(self.data)
        test_out =  self.data.array-normalised_data.array
        numpy.testing.assert_allclose(test_out, 0.5)

        # test default with out
        out = self.data.geometry.allocate(None)
        normaliser(self.data, out=out)
        numpy.testing.assert_allclose(out.array, normalised_data.array)

        # test default inplace
        normaliser(self.data, out=self.data)
        numpy.testing.assert_allclose(self.data.array, normalised_data.array)

    def test_no_flat_no_dark(self):
        #test with no flat or dark-field
        flat_field = None
        dark_field = None
        normaliser = Normaliser(flat_field=flat_field, dark_field=dark_field)
        with self.assertRaises(ValueError):
            normalised_data = normaliser(self.data)

    def test_zeros_in_flat(self):
        # test with zeros in flat field
        flat_field = numpy.ones(self.data.shape[1::])
        flat_field[0,0] = 0
        dark_field = None
        normaliser = Normaliser(flat_field=flat_field, dark_field=dark_field)
        with numpy.testing.assert_warns(UserWarning):
            normalised_data = normaliser(self.data)
        test_out = self.data.array.copy()
        test_out[:,0,0] = 1e-5
        numpy.testing.assert_allclose(normalised_data.array, test_out)
