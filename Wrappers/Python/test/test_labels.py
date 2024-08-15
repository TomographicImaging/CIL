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

import numpy as np

import unittest

from cil.framework.labels import (_LabelsBase, 
                                  FillTypes, UnitsAngles, 
                                  AcquisitionTypes, AcquisitionDimensions, 
                                  ImageDimensionLabels, AcquisitionDimensionLabels, Backends)

from cil.framework import AcquisitionGeometry, ImageGeometry

class Test_Lables(unittest.TestCase):

    def test_labels_validate(self):

        input_good = ["3D", AcquisitionDimensions.DIM3]
        input_bad = ["bad_str", "DIM3", UnitsAngles.DEGREE]

        for input in input_good:
            self.assertTrue(AcquisitionDimensions.validate(input))

        for input in input_bad:
            with self.assertRaises(ValueError):
                AcquisitionDimensions.validate(input)


    def test_labels_get_enum_member(self):
        out_gold = AcquisitionDimensions.DIM3

        input_good = ["3D", AcquisitionDimensions.DIM3]
        input_bad = ["bad_str", "DIM3", UnitsAngles.DEGREE]

        for input in input_good:
            out =  AcquisitionDimensions.get_enum_member(input)
            self.assertEqual(out, out_gold)
            self.assertTrue(isinstance(out, AcquisitionDimensions))
        
        for input in input_bad:
            with self.assertRaises(ValueError):
                AcquisitionDimensions.get_enum_member(input)


    def test_labels_get_enum_value(self):
        out_gold = "3D"

        input_good = ["3D", AcquisitionDimensions.DIM3]
        input_bad = ["bad_str", "DIM3", UnitsAngles.DEGREE]

        for input in input_good:
            out =  AcquisitionDimensions.get_enum_value(input)
            self.assertEqual(out, out_gold)
            self.assertTrue(isinstance(out, str))
        
        for input in input_bad:
            with self.assertRaises(ValueError):
                AcquisitionDimensions.get_enum_value(input)


    def test_labels_eq(self):
        self.assertTrue(_LabelsBase.__eq__(AcquisitionDimensions.DIM3, "3D"))
        self.assertTrue(_LabelsBase.__eq__(AcquisitionDimensions.DIM3, AcquisitionDimensions.DIM3))

        self.assertFalse(_LabelsBase.__eq__(AcquisitionDimensions.DIM3, "DIM3"))
        self.assertFalse(_LabelsBase.__eq__(AcquisitionDimensions.DIM3, "2D"))
        self.assertFalse(_LabelsBase.__eq__(AcquisitionDimensions.DIM3, AcquisitionDimensions.DIM2))
        self.assertFalse(_LabelsBase.__eq__(AcquisitionDimensions.DIM3, AcquisitionDimensions))


    def test_labels_contains(self):
        self.assertTrue(_LabelsBase.__contains__(AcquisitionDimensions, "3D"))
        self.assertTrue(_LabelsBase.__contains__(AcquisitionDimensions, AcquisitionDimensions.DIM3))
        self.assertTrue(_LabelsBase.__contains__(AcquisitionDimensions, AcquisitionDimensions.DIM2))

        self.assertFalse(_LabelsBase.__contains__(AcquisitionDimensions, "DIM3"))
        self.assertFalse(_LabelsBase.__contains__(AcquisitionDimensions, AcquisitionDimensions))


    def test_backends(self):
        self.assertTrue('astra' in Backends)
        self.assertTrue('cil' in Backends)
        self.assertTrue('tigre' in Backends)
        self.assertTrue(Backends.ASTRA in Backends)
        self.assertTrue(Backends.CIL in Backends)
        self.assertTrue(Backends.TIGRE in Backends)

    def test_fill_types(self):
        self.assertTrue('random' in FillTypes)
        self.assertTrue('random_int' in FillTypes)
        self.assertTrue(FillTypes.RANDOM in FillTypes)
        self.assertTrue(FillTypes.RANDOM_INT in FillTypes)
    
    def test_units_angles(self):
        self.assertTrue('degree' in UnitsAngles)
        self.assertTrue('radian' in UnitsAngles)
        self.assertTrue(UnitsAngles.DEGREE in UnitsAngles)
        self.assertTrue(UnitsAngles.RADIAN in UnitsAngles)

    def test_acquisition_type(self):
        self.assertTrue('parallel' in AcquisitionTypes)
        self.assertTrue('cone' in AcquisitionTypes)
        self.assertTrue(AcquisitionTypes.PARALLEL in AcquisitionTypes)
        self.assertTrue(AcquisitionTypes.CONE in AcquisitionTypes)

    def test_acquisition_dimension(self):
        self.assertTrue('2D' in AcquisitionDimensions)
        self.assertTrue('3D' in AcquisitionDimensions)
        self.assertTrue(AcquisitionDimensions.DIM2 in AcquisitionDimensions)
        self.assertTrue(AcquisitionDimensions.DIM3 in AcquisitionDimensions)

    def test_image_dimension_labels(self):
        self.assertTrue('channel' in ImageDimensionLabels)
        self.assertTrue('vertical' in ImageDimensionLabels)
        self.assertTrue('horizontal_x' in ImageDimensionLabels)
        self.assertTrue('horizontal_y' in ImageDimensionLabels)
        self.assertTrue(ImageDimensionLabels.CHANNEL in ImageDimensionLabels)
        self.assertTrue(ImageDimensionLabels.VERTICAL in ImageDimensionLabels)
        self.assertTrue(ImageDimensionLabels.HORIZONTAL_X in ImageDimensionLabels)
        self.assertTrue(ImageDimensionLabels.HORIZONTAL_Y in ImageDimensionLabels)

    def test_image_dimension_labels_default_order(self):
        
        order_gold = [ImageDimensionLabels.CHANNEL, ImageDimensionLabels.VERTICAL, ImageDimensionLabels.HORIZONTAL_Y, ImageDimensionLabels.HORIZONTAL_X]

        order = ImageDimensionLabels.get_default_order_for_engine("cil")
        self.assertEqual(order,order_gold )

        order = ImageDimensionLabels.get_default_order_for_engine("tigre")
        self.assertEqual(order,order_gold)

        order = ImageDimensionLabels.get_default_order_for_engine("astra")
        self.assertEqual(order, order_gold)

        with self.assertRaises(ValueError):
            order = AcquisitionDimensionLabels.get_default_order_for_engine("bad_engine")  


    def test_image_dimension_labels_get_order(self):
        ig = ImageGeometry(4, 8, 1, channels=2)
        ig.set_labels(['channel', 'horizontal_y', 'horizontal_x'])

        # for 2D all engines have the same order
        order_gold = [ImageDimensionLabels.CHANNEL, ImageDimensionLabels.HORIZONTAL_Y, ImageDimensionLabels.HORIZONTAL_X]
        order = ImageDimensionLabels.get_order_for_engine("cil", ig)
        self.assertEqual(order, order_gold)

        order = ImageDimensionLabels.get_order_for_engine("tigre", ig)
        self.assertEqual(order, order_gold)

        order = ImageDimensionLabels.get_order_for_engine("astra", ig)
        self.assertEqual(order, order_gold)

    def test_image_dimension_labels_check_order(self):
        ig = ImageGeometry(4, 8, 1, channels=2)
        ig.set_labels(['horizontal_x', 'horizontal_y', 'channel'])

        with self.assertRaises(ValueError):
            ImageDimensionLabels.check_order_for_engine("cil", ig)
        
        with self.assertRaises(ValueError):
            ImageDimensionLabels.check_order_for_engine("tigre", ig)

        with self.assertRaises(ValueError):
            ImageDimensionLabels.check_order_for_engine("astra", ig)

        ig.set_labels(['channel', 'horizontal_y', 'horizontal_x'])
        self.assertTrue( ImageDimensionLabels.check_order_for_engine("cil", ig))
        self.assertTrue( ImageDimensionLabels.check_order_for_engine("tigre", ig))
        self.assertTrue( ImageDimensionLabels.check_order_for_engine("astra", ig))

    def test_acquisition_dimension_labels(self):
        self.assertTrue('channel' in AcquisitionDimensionLabels)
        self.assertTrue('angle' in AcquisitionDimensionLabels)
        self.assertTrue('vertical' in AcquisitionDimensionLabels)
        self.assertTrue('horizontal' in AcquisitionDimensionLabels)
        self.assertTrue(AcquisitionDimensionLabels.CHANNEL in AcquisitionDimensionLabels)
        self.assertTrue(AcquisitionDimensionLabels.ANGLE in AcquisitionDimensionLabels)
        self.assertTrue(AcquisitionDimensionLabels.VERTICAL in AcquisitionDimensionLabels)
        self.assertTrue(AcquisitionDimensionLabels.HORIZONTAL in AcquisitionDimensionLabels)

    def test_acquisition_dimension_labels_default_order(self):
        order = AcquisitionDimensionLabels.get_default_order_for_engine("cil")
        self.assertEqual(order, [AcquisitionDimensionLabels.CHANNEL, AcquisitionDimensionLabels.ANGLE, AcquisitionDimensionLabels.VERTICAL, AcquisitionDimensionLabels.HORIZONTAL])

        order = AcquisitionDimensionLabels.get_default_order_for_engine("tigre")
        self.assertEqual(order, [AcquisitionDimensionLabels.CHANNEL, AcquisitionDimensionLabels.ANGLE, AcquisitionDimensionLabels.VERTICAL, AcquisitionDimensionLabels.HORIZONTAL])

        order = AcquisitionDimensionLabels.get_default_order_for_engine("astra")
        self.assertEqual(order, [AcquisitionDimensionLabels.CHANNEL, AcquisitionDimensionLabels.VERTICAL, AcquisitionDimensionLabels.ANGLE, AcquisitionDimensionLabels.HORIZONTAL])

        with self.assertRaises(ValueError):
            order = AcquisitionDimensionLabels.get_default_order_for_engine("bad_engine")  

    def test_acquisition_dimension_labels_get_order(self):

        ag = AcquisitionGeometry.create_Parallel2D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel(4)\
            .set_channels(8)\
            .set_labels(['angle', 'horizontal', 'channel'])
        
        # for 2D all engines have the same order
        order_gold = [AcquisitionDimensionLabels.CHANNEL, AcquisitionDimensionLabels.ANGLE, AcquisitionDimensionLabels.HORIZONTAL]
        order = AcquisitionDimensionLabels.get_order_for_engine("cil", ag)
        self.assertEqual(order, order_gold)

        order = AcquisitionDimensionLabels.get_order_for_engine("tigre", ag)
        self.assertEqual(order, order_gold)

        order = AcquisitionDimensionLabels.get_order_for_engine("astra", ag)
        self.assertEqual(order, order_gold)


        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel((4,2))\
            .set_labels(['angle', 'horizontal', 'vertical'])
        

        order_gold = [AcquisitionDimensionLabels.ANGLE, AcquisitionDimensionLabels.VERTICAL, AcquisitionDimensionLabels.HORIZONTAL]
        order = AcquisitionDimensionLabels.get_order_for_engine("cil", ag)
        self.assertEqual(order, order_gold)

        order = AcquisitionDimensionLabels.get_order_for_engine("tigre", ag)
        self.assertEqual(order, order_gold)

        order_gold = [AcquisitionDimensionLabels.VERTICAL, AcquisitionDimensionLabels.ANGLE, AcquisitionDimensionLabels.HORIZONTAL]
        order = AcquisitionDimensionLabels.get_order_for_engine("astra", ag)
        self.assertEqual(order, order_gold)


    def test_acquisition_dimension_labels_check_order(self):

        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel((8,4))\
            .set_channels(2)\
            .set_labels(['angle', 'horizontal', 'channel', 'vertical'])
        
        with self.assertRaises(ValueError):
            AcquisitionDimensionLabels.check_order_for_engine("cil", ag)

        with self.assertRaises(ValueError):
            AcquisitionDimensionLabels.check_order_for_engine("tigre", ag)

        with self.assertRaises(ValueError):
            AcquisitionDimensionLabels.check_order_for_engine("astra", ag)

        ag.set_labels(['channel', 'angle', 'vertical', 'horizontal'])
        self.assertTrue( AcquisitionDimensionLabels.check_order_for_engine("cil", ag))
        self.assertTrue( AcquisitionDimensionLabels.check_order_for_engine("tigre", ag))

        ag.set_labels(['channel', 'vertical', 'angle', 'horizontal'])
        self.assertTrue( AcquisitionDimensionLabels.check_order_for_engine("astra", ag))
