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
from cil.framework.labels import (StrEnum,
                                  FillTypes, UnitsAngles,
                                  AcquisitionTypes, AcquisitionDimensions,
                                  ImageDimensionLabels, AcquisitionDimensionLabels, Backends)


class Test_Lables(unittest.TestCase):
    def test_labels_strenum(self):
        for item in ("3D", "DIM3", AcquisitionDimensions.DIM3):
            out = AcquisitionDimensions(item)
            self.assertEqual(out, AcquisitionDimensions.DIM3)
            self.assertTrue(isinstance(out, AcquisitionDimensions))
        for item in ("bad_str", "4D", "DIM4", UnitsAngles.DEGREE):
            with self.assertRaises(ValueError):
                AcquisitionDimensions(item)

    def test_labels_strenum_eq(self):
        for i in ("3D", "DIM3", AcquisitionDimensions.DIM3):
            self.assertEqual(AcquisitionDimensions.DIM3, i)
            self.assertEqual(i, AcquisitionDimensions.DIM3)
        for i in ("2D", "DIM2", AcquisitionDimensions.DIM2, AcquisitionDimensions):
            self.assertNotEqual(AcquisitionDimensions.DIM3, i)

    def test_labels_contains(self):
        for i in ("3D", "DIM3", AcquisitionDimensions.DIM3, AcquisitionDimensions.DIM2):
            self.assertIn(i, AcquisitionDimensions)
        for i in ("4D", "DIM4", AcquisitionDimensions):
            self.assertNotIn(i, AcquisitionDimensions)

    def test_backends(self):
        for i in ('ASTRA', 'CIL', 'TIGRE'):
            self.assertIn(i, Backends)
            self.assertIn(i.lower(), Backends)
            self.assertIn(getattr(Backends, i), Backends)

    def test_fill_types(self):
        for i in ('RANDOM', 'RANDOM_INT'):
            self.assertIn(i, FillTypes)
            self.assertIn(i.lower(), FillTypes)
            self.assertIn(getattr(FillTypes, i), FillTypes)

    def test_units_angles(self):
        for i in ('DEGREE', 'RADIAN'):
            self.assertIn(i, UnitsAngles)
            self.assertIn(i.lower(), UnitsAngles)
            self.assertIn(getattr(UnitsAngles, i), UnitsAngles)

    def test_acquisition_type(self):
        for i in ('PARALLEL', 'CONE'):
            self.assertIn(i, AcquisitionTypes)
            self.assertIn(i.lower(), AcquisitionTypes)
            self.assertIn(getattr(AcquisitionTypes, i), AcquisitionTypes)

    def test_acquisition_dimension(self):
        for i in ('2D', '3D'):
            self.assertIn(i, AcquisitionDimensions)
        for i in ('DIM2', 'DIM3'):
            self.assertIn(i, AcquisitionDimensions)
            self.assertIn(i.lower(), AcquisitionDimensions)
            self.assertIn(getattr(AcquisitionDimensions, i), AcquisitionDimensions)

    def test_image_dimension_labels(self):
        for i in ('CHANNEL', 'VERTICAL', 'HORIZONTAL_X', 'HORIZONTAL_Y'):
            self.assertIn(i, ImageDimensionLabels)
            self.assertIn(i.lower(), ImageDimensionLabels)
            self.assertIn(getattr(ImageDimensionLabels, i), ImageDimensionLabels)

    def test_image_dimension_labels_default_order(self):
        order_gold = ImageDimensionLabels.CHANNEL, 'VERTICAL', 'horizontal_y', 'HORIZONTAL_X'
        for i in ('CIL', 'TIGRE', 'ASTRA'):
            self.assertSequenceEqual(ImageDimensionLabels.get_order_for_engine(i), order_gold)

        with self.assertRaises((KeyError, ValueError)):
            AcquisitionDimensionLabels.get_order_for_engine("bad_engine")

    def test_image_dimension_labels_get_order(self):
        ig = ImageGeometry(4, 8, 1, channels=2)
        ig.set_labels(['channel', 'horizontal_y', 'horizontal_x'])

        # for 2D all engines have the same order
        order_gold = ImageDimensionLabels.CHANNEL, 'HORIZONTAL_Y', 'horizontal_x'
        self.assertSequenceEqual(ImageDimensionLabels.get_order_for_engine('cil', ig), order_gold)
        self.assertSequenceEqual(ImageDimensionLabels.get_order_for_engine('tigre', ig), order_gold)
        self.assertSequenceEqual(ImageDimensionLabels.get_order_for_engine('astra', ig), order_gold)

    def test_image_dimension_labels_check_order(self):
        ig = ImageGeometry(4, 8, 1, channels=2)
        ig.set_labels(['horizontal_x', 'horizontal_y', 'channel'])

        for i in ('cil', 'tigre', 'astra'):
            with self.assertRaises(ValueError):
                ImageDimensionLabels.check_order_for_engine(i, ig)

        ig.set_labels(['channel', 'horizontal_y', 'horizontal_x'])
        self.assertTrue(ImageDimensionLabels.check_order_for_engine("cil", ig))
        self.assertTrue(ImageDimensionLabels.check_order_for_engine("tigre", ig))
        self.assertTrue(ImageDimensionLabels.check_order_for_engine("astra", ig))

    def test_acquisition_dimension_labels(self):
        for i in ('CHANNEL', 'ANGLE', 'VERTICAL', 'HORIZONTAL'):
            self.assertIn(i, AcquisitionDimensionLabels)
            self.assertIn(i.lower(), AcquisitionDimensionLabels)
            self.assertIn(getattr(AcquisitionDimensionLabels, i), AcquisitionDimensionLabels)

    def test_acquisition_dimension_labels_default_order(self):
        gold = AcquisitionDimensionLabels.CHANNEL, 'ANGLE', 'vertical', 'HORIZONTAL'
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine('CIL'), gold)
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine(Backends.TIGRE), gold)
        gold = 'CHANNEL', 'VERTICAL', 'ANGLE', 'HORIZONTAL'
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine('astra'), gold)

        with self.assertRaises((KeyError, ValueError)):
            AcquisitionDimensionLabels.get_order_for_engine("bad_engine")

    def test_acquisition_dimension_labels_get_order(self):
        ag = AcquisitionGeometry.create_Parallel2D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel(4)\
            .set_channels(8)\
            .set_labels(['angle', 'horizontal', 'channel'])

        # for 2D all engines have the same order
        order_gold = AcquisitionDimensionLabels.CHANNEL, 'ANGLE', 'horizontal'
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine('CIL', ag), order_gold)
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine('TIGRE', ag), order_gold)
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine('ASTRA', ag), order_gold)

        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel((4,2))\
            .set_labels(['angle', 'horizontal', 'vertical'])

        order_gold = AcquisitionDimensionLabels.ANGLE, 'VERTICAL', 'horizontal'
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine("cil", ag), order_gold)
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine("tigre", ag), order_gold)
        order_gold = AcquisitionDimensionLabels.VERTICAL, 'ANGLE', 'horizontal'
        self.assertSequenceEqual(AcquisitionDimensionLabels.get_order_for_engine("astra", ag), order_gold)

    def test_acquisition_dimension_labels_check_order(self):
        ag = AcquisitionGeometry.create_Parallel3D()\
            .set_angles(np.arange(0,16 , 1), angle_unit="degree")\
            .set_panel((8,4))\
            .set_channels(2)\
            .set_labels(['angle', 'horizontal', 'channel', 'vertical'])

        for i in ('cil', 'tigre', 'astra'):
            with self.assertRaises(ValueError):
                AcquisitionDimensionLabels.check_order_for_engine(i, ag)

        ag.set_labels(['channel', 'angle', 'vertical', 'horizontal'])
        self.assertTrue(AcquisitionDimensionLabels.check_order_for_engine("cil", ag))
        self.assertTrue(AcquisitionDimensionLabels.check_order_for_engine("tigre", ag))

        ag.set_labels(['channel', 'vertical', 'angle', 'horizontal'])
        self.assertTrue(AcquisitionDimensionLabels.check_order_for_engine("astra", ag))
