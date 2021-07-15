# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev, Edoardo Pasca and Srikanth Nagella

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

'''
Unit tests for Readers
@author: Mr. Srikanth Nagella
'''
import unittest

import numpy.testing
import wget
import os
from cil.io import NXTomoReader
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, AcquisitionGeometry


class TestNXTomoReader(unittest.TestCase):

    def setUp(self):
        # wget.download(
        #     'https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        self.filename = '24737_fd.nxs'
        # self.data_dir = os.path.join(os.getcwd(), 'test_nxtomo')
        # if not os.path.exists(self.data_dir):
        #     os.mkdir(self.data_dir)

        # self.ag3d = AcquisitionGeometry.create_Parallel3D()\
        #     .set_angles([i for i in range(0, 181, 2)])\
        #     .set_panel([135, 160])\
        #     .set_channels(1)\
        #     .set_labels(['horizontal', 'vertical', 'angle'])

        # self.ad3d = self.ag3d.allocate('random_int')

        # self.flats = numpy.ones(shape=(1, 135, 160))
        # self.darks = numpy.zeros(shape=(1, 135, 160))
        # self.image_key_ids = numpy.zeros(shape=(93,135,160))
        # self.image_key_ids[0] = 1
        # self.image_key_ids[1] = 2

    # def tearDown(self):
    #     os.remove(self.filename)

    def test_get_dimensions(self):
        nr = NXTomoReader(self.filename)
        self.assertEqual(nr.get_sinogram_dimensions(),
                         (135, 91, 160), "Sinogram dimensions are not correct")

    def test_get_projection_dimensions(self):
        nr = NXTomoReader(self.filename)
        self.assertEqual(nr.get_projection_dimensions(
        ), (91, 135, 160), "Projection dimensions are not correct")

    def test_load_projection_without_dimensions(self):
        nr = NXTomoReader(self.filename)
        projections = nr.load_projection()
        self.assertEqual(projections.shape, (91, 135, 160),
                         "Loaded projection data dimensions are not correct")

    def test_load_projection_with_dimensions(self):
        nr = NXTomoReader(self.filename)
        projections = nr.load_projection(
            (slice(0, 1), slice(0, 135), slice(0, 160)))
        self.assertEqual(projections.shape, (1, 135, 160),
                         "Loaded projection data dimensions are not correct")

    def test_load_projection_compare_single(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection(
            (slice(0, 1), slice(0, 135), slice(0, 160)))
        numpy.testing.assert_array_equal(
            projections_part, projections_full[0:1, :, :])

    def test_load_projection_compare_multi(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection(
            (slice(0, 3), slice(0, 135), slice(0, 160)))
        numpy.testing.assert_array_equal(
            projections_part, projections_full[0:3, :, :])

    def test_load_projection_compare_random(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection(
            (slice(1, 8), slice(5, 10), slice(8, 20)))
        numpy.testing.assert_array_equal(
            projections_part, projections_full[1:8, 5:10, 8:20])

    def test_load_projection_compare_full(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection(
            (slice(None, None), slice(None, None), slice(None, None)))
        numpy.testing.assert_array_equal(
            projections_part, projections_full[:, :, :])

    def test_load_flat_compare_full(self):
        nr = NXTomoReader(self.filename)
        flats_full = nr.load_flat()
        flats_part = nr.load_flat(
            (slice(None, None), slice(None, None), slice(None, None)))
        numpy.testing.assert_array_equal(flats_part, flats_full[:, :, :])

    def test_load_dark_compare_full(self):
        nr = NXTomoReader(self.filename)
        darks_full = nr.load_dark()
        darks_part = nr.load_dark(
            (slice(None, None), slice(None, None), slice(None, None)))
        numpy.testing.assert_array_equal(darks_part, darks_full[:, :, :])

    def test_projection_angles(self):
        nr = NXTomoReader(self.filename)
        angles = nr.get_projection_angles()
        self.assertEqual(angles.shape, (91,),
                         "Loaded projection number of angles are not correct")

    def test_get_acquition_data(self):
        nr = NXTomoReader(self.filename)
        acq_data = nr.get_acquisition_data()
        self.assertEqual(acq_data.geometry.geom_type, 'parallel')
        self.assertEqual(acq_data.geometry.angles.shape, (91,),
                         'AcquisitionGeometry.angles is not correct')
        self.assertEqual(acq_data.geometry.pixel_num_h, 160,
                         'AcquisitionGeometry.pixel_num_h is not correct')
        self.assertEqual(acq_data.geometry.pixel_size_h, 1,
                         'AcquisitionGeometry.pixel_size_h is not correct')
        self.assertEqual(acq_data.geometry.pixel_num_v, 135,
                         'AcquisitionGeometry.pixel_num_v is not correct')
        self.assertEqual(acq_data.geometry.pixel_size_v, 1,
                         'AcquisitionGeometry.pixel_size_v is not correct')

    def test_get_acquisition_data_whole(self):
        nr = NXTomoReader(self.filename)
        acq_data = nr.get_acquisition_data()
        acq_data_whole = nr.get_acquisition_data_whole()
        self.assertEqual(acq_data.geometry.geom_type,
                         acq_data_whole.geometry.geom_type)
        self.assertEqual(acq_data.geometry.angles.shape,
                         acq_data_whole.geometry.angles.shape,
                         'AcquisitionGeometry.angles is not correct')
        self.assertEqual(acq_data.geometry.pixel_num_h,
                         acq_data_whole.geometry.pixel_num_h,
                         'AcquisitionGeometry.pixel_num_h is not correct')
        self.assertEqual(acq_data.geometry.pixel_size_h,
                         acq_data_whole.geometry.pixel_size_h,
                         'AcquisitionGeometry.pixel_size_h is not correct')
        self.assertEqual(acq_data.geometry.pixel_num_v,
                         acq_data_whole.geometry.pixel_num_v,
                         'AcquisitionGeometry.pixel_num_v is not correct')
        self.assertEqual(acq_data.geometry.pixel_size_v,
                         acq_data_whole.geometry.pixel_size_v,
                         'AcquisitionGeometry.pixel_size_v is not correct')

    def test_get_acquisition_data_subset(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.get_acquisition_data_subset(
            ymin=5, ymax=10).as_array()
        numpy.testing.assert_array_equal(
            projections_part, projections_full[:, 5:10, :])

    def test_get_acquisition_data_slice(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.get_acquisition_data_slice(y_slice=5).as_array()
        numpy.testing.assert_array_equal(
            projections_part, projections_full[:, 5, :])

    def test_get_acquisition_data_batch(self):
        nr = NXTomoReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.get_acquisition_data_batch(bmin=5, bmax=7).as_array()
        numpy.testing.assert_array_equal(
            projections_part, projections_full[5:7, :, :])


if __name__ == "__main__":
    unittest.main()
