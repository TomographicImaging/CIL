#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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
from cil.framework import ImageGeometry
from cil.framework.labels import ImageDimension

initialise_tests()


class TestImageGeometry(unittest.TestCase):
    def setUp(self):
        self.ig = ImageGeometry(2,3,4,channels=5)

    # get_slice tests ---------------------------------------------------------------------------------------------------------------

    def test_get_slice_vertical(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        ig = self.ig.copy()
        ig.set_labels(non_default_dimension_labels)
        ig.voxel_size_z = 5.5
        sub = ig.get_slice(vertical = 1)
        self.assertTrue( sub.shape == (2,5,3))
        self.assertEqual(sub.voxel_size_z,5.5)
        self.assertEqual(sub.center_x,0)
        self.assertEqual(sub.center_y,0)
        self.assertEqual(sub.center_z,-0.5*sub.voxel_size_z)

    def test_get_slice_vertical_centre(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(vertical = 'centre')
        self.assertTrue( sub.shape == (2,5,3))
        self.assertEqual(sub.center_x,0)
        self.assertEqual(sub.center_y,0)
        self.assertEqual(sub.center_z,0)
 

    def test_get_slice_horizontal_x(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(horizontal_x = 1)
        self.assertTrue(sub.shape == (5,3,4))
        self.assertEqual(sub.center_x,0.5)
        self.assertEqual(sub.center_y,0)
        self.assertEqual(sub.center_z,0)

    def test_get_slice_horizontal_x_centre(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(horizontal_x = 'centre')
        self.assertTrue(sub.shape == (5,3,4))
        self.assertEqual(sub.center_x,0)
        self.assertEqual(sub.center_y,0)
        self.assertEqual(sub.center_z,0)

    def test_get_slice_channel(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(channel = 1)
        self.assertTrue(sub.shape == (2,3,4))
        self.assertTrue(sub.channels == 1)

    def test_get_slice_channel_centre(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(channel = 'centre')
        self.assertTrue(sub.shape == (2,3,4))
        self.assertTrue(sub.channels == 1)

    def test_get_slice_horizontal_y(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["HORIZONTAL_Y"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(horizontal_y = 0)
        self.assertTrue( sub.shape == (2,))
        self.assertEqual(sub.center_x,0)
        self.assertEqual(sub.center_y,-1)
        self.assertEqual(sub.center_z,0)

    def test_get_slice_horizontal_y_centre(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["HORIZONTAL_Y"]]
        self.ig.set_labels(non_default_dimension_labels)
        sub = self.ig.get_slice(horizontal_y = 'centre')
        self.assertTrue( sub.shape == (2,))
        self.assertEqual(sub.center_x,0)
        self.assertEqual(sub.center_y,0)
        self.assertEqual(sub.center_z,0)

    def test_get_slice_horizontal_x_and_horizontal_y(self):
        sub = self.ig.get_slice(horizontal_x=0,horizontal_y=0)
        self.assertTrue( sub.shape == (5,4))

    # test get_centre_slice ---------------------------------------------------------------------------------------------------------------

    def test_get_centre_slice(self):
        sub = self.ig.get_centre_slice()
        sub2 = self.ig.get_slice(vertical='centre')
        self.assertTrue( sub == sub2)

    # -------------------------------------------------------------------------------------------------------------------------------

    def test_shape_change_with_new_labels(self):
        new_dimension_labels = [ImageDimension["HORIZONTAL_Y"], ImageDimension["CHANNEL"], ImageDimension["VERTICAL"], ImageDimension["HORIZONTAL_X"]]
        self.ig.set_labels(new_dimension_labels)
        self.assertTrue( self.ig.shape == (3,5,4,2))