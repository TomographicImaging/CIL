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

from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.framework.labels import FillType, AngleUnit, AcquisitionType, ImageDimension, AcquisitionDimension, Backend


class Test_Lables(unittest.TestCase):
    def test_labels_strenum(self):
        for item in (AngleUnit.DEGREE, "DEGREE", "degree"):
            out = AngleUnit(item)
            self.assertEqual(out, AngleUnit.DEGREE)
            self.assertTrue(isinstance(out, AngleUnit))
        for item in ("bad_str", FillType.RANDOM):
            with self.assertRaises(ValueError):
                AngleUnit(item)

    def test_labels_strenum_eq(self):
        for i in (AngleUnit.RADIAN, "RADIAN", "radian"):
            self.assertEqual(AngleUnit.RADIAN, i)
            self.assertEqual(i, AngleUnit.RADIAN)
        for i in ("DEGREE", AngleUnit.DEGREE, AngleUnit):
            self.assertNotEqual(AngleUnit.RADIAN, i)

    def test_labels_contains(self):
        for i in ("RADIAN", "degree", AngleUnit.RADIAN, AngleUnit.DEGREE):
            self.assertIn(i, AngleUnit)
        for i in ("bad_str", AngleUnit):
            self.assertNotIn(i, AngleUnit)

    def test_backends(self):
        for i in ('ASTRA', 'CIL', 'TIGRE'):
            self.assertIn(i, Backend)
            self.assertIn(i.lower(), Backend)
            self.assertIn(getattr(Backend, i), Backend)

    def test_fill_types(self):
        for i in ('RANDOM', 'RANDOM_INT', 'RANDOM_DEPRECATED', 'RANDOM_INT_DEPRECATED'):
            self.assertIn(i, FillType)
            self.assertIn(i.lower(), FillType)
            self.assertIn(getattr(FillType, i), FillType)

    def test_units_angles(self):
        for i in ('DEGREE', 'RADIAN'):
            self.assertIn(i, AngleUnit)
            self.assertIn(i.lower(), AngleUnit)
            self.assertIn(getattr(AngleUnit, i), AngleUnit)

    def test_acquisition_type(self):
        for i in ('PARALLEL', 'CONE', 'DIM2', 'DIM3', 'CONE_SOUV'):
            self.assertIn(i, AcquisitionType)
            self.assertIn(i.lower(), AcquisitionType)
            self.assertIn(getattr(AcquisitionType, i), AcquisitionType)
        combo = AcquisitionType.DIM2 | AcquisitionType.CONE
        for i in (AcquisitionType.DIM2, AcquisitionType.CONE):
            self.assertIn(i, combo)
        for i in (AcquisitionType.DIM3, AcquisitionType.PARALLEL):
            self.assertNotIn(i, combo)
        for i in ('2D', 'DIM2', AcquisitionType.DIM2):
            self.assertEqual(combo.dimension, i)
        for i in ('CONE', 'cone', AcquisitionType.CONE):
            self.assertEqual(combo.geometry, i)

    def test_image_dimension_labels(self):
        for i in ('CHANNEL', 'VERTICAL', 'HORIZONTAL_X', 'HORIZONTAL_Y'):
            self.assertIn(i, ImageDimension)
            self.assertIn(i.lower(), ImageDimension)
            self.assertIn(getattr(ImageDimension, i), ImageDimension)

    def test_image_dimension_labels_default_order(self):
        order_gold = ImageDimension.CHANNEL, 'VERTICAL', 'horizontal_y', 'HORIZONTAL_X'
        for i in ('CIL', 'TIGRE', 'ASTRA'):
            self.assertSequenceEqual(ImageDimension.get_order_for_engine(i), order_gold)

        with self.assertRaises((KeyError, ValueError)):
            AcquisitionDimension.get_order_for_engine("bad_engine")

    def test_image_dimension_labels_get_order(self):
        ig = ImageGeometry(4, 8, 1, channels=2)
        ig.set_labels(['channel', 'horizontal_y', 'horizontal_x'])

        # for 2D all engines have the same order
        order_gold = ImageDimension.CHANNEL, 'HORIZONTAL_Y', 'horizontal_x'
        self.assertSequenceEqual(ImageDimension.get_order_for_engine('cil', ig), order_gold)
        self.assertSequenceEqual(ImageDimension.get_order_for_engine('tigre', ig), order_gold)
        self.assertSequenceEqual(ImageDimension.get_order_for_engine('astra', ig), order_gold)

    def test_image_dimension_labels_check_order(self):
        ig = ImageGeometry(4, 8, 1, channels=2)
        ig.set_labels(['horizontal_x', 'horizontal_y', 'channel'])

        for i in ('cil', 'tigre', 'astra'):
            with self.assertRaises(ValueError):
                ImageDimension.check_order_for_engine(i, ig)

        ig.set_labels(['channel', 'horizontal_y', 'horizontal_x'])
        self.assertTrue(ImageDimension.check_order_for_engine("cil", ig))
        self.assertTrue(ImageDimension.check_order_for_engine("tigre", ig))
        self.assertTrue(ImageDimension.check_order_for_engine("astra", ig))

    def test_acquisition_dimension_labels(self):
        for i in ('CHANNEL', 'ANGLE', 'VERTICAL', 'HORIZONTAL'):
            self.assertIn(i, AcquisitionDimension)
            self.assertIn(i.lower(), AcquisitionDimension)
            self.assertIn(getattr(AcquisitionDimension, i), AcquisitionDimension)

    def test_acquisition_dimension_labels_default_order(self):
        gold = AcquisitionDimension.CHANNEL, 'ANGLE', 'vertical', 'HORIZONTAL'
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine('CIL'), gold)
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine(Backend.TIGRE), gold)
        gold = 'CHANNEL', 'VERTICAL', 'ANGLE', 'HORIZONTAL'
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine('astra'), gold)

        with self.assertRaises((KeyError, ValueError)):
            AcquisitionDimension.get_order_for_engine("bad_engine")

    def test_acquisition_dimension_labels_get_order(self):
        ag = AcquisitionGeometry.create_Parallel2D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel(4)\
            .set_channels(8)\
            .set_labels(['angle', 'horizontal', 'channel'])

        # for 2D all engines have the same order
        order_gold = AcquisitionDimension.CHANNEL, 'ANGLE', 'horizontal'
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine('CIL', ag), order_gold)
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine('TIGRE', ag), order_gold)
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine('ASTRA', ag), order_gold)

        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel((4,2))\
            .set_labels(['angle', 'horizontal', 'vertical'])

        order_gold = AcquisitionDimension.ANGLE, 'VERTICAL', 'horizontal'
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine("cil", ag), order_gold)
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine("tigre", ag), order_gold)
        order_gold = AcquisitionDimension.VERTICAL, 'ANGLE', 'horizontal'
        self.assertSequenceEqual(AcquisitionDimension.get_order_for_engine("astra", ag), order_gold)

    def test_acquisition_dimension_labels_check_order(self):
        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel((8,4))\
            .set_channels(2)\
            .set_labels(['angle', 'horizontal', 'channel', 'vertical'])

        for i in ('cil', 'tigre', 'astra'):
            with self.assertRaises(ValueError):
                AcquisitionDimension.check_order_for_engine(i, ag)

        ag.set_labels(['channel', 'angle', 'vertical', 'horizontal'])
        self.assertTrue(AcquisitionDimension.check_order_for_engine("cil", ag))
        self.assertTrue(AcquisitionDimension.check_order_for_engine("tigre", ag))

        ag.set_labels(['channel', 'vertical', 'angle', 'horizontal'])
        self.assertTrue(AcquisitionDimension.check_order_for_engine("astra", ag))
