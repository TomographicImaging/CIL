# -*- coding: utf-8 -*-
#  Copyright 2022 United Kingdom Research and Innovation
#  Copyright 2022 The University of Manchester
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
from cil.framework import AcquisitionGeometry
import numpy as np

from utils import has_astra, initialise_tests
initialise_tests()

if has_astra:
    from cil.plugins.astra.utilities import convert_geometry_to_astra
    from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D

class TestGeometry(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        pixels_x = 128
        pixels_y = 3
        
        angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        angles_rad = angles_deg * np.pi /180.0

        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles_rad, angle_unit='radian')\
                                .set_labels(['angle','horizontal'])\
                                .set_panel(pixels_x, 0.1)
        
        ig = ag.get_ImageGeometry()


        ag_deg = AcquisitionGeometry.create_Parallel2D()\
                                    .set_angles(angles_deg, angle_unit='degree')\
                                    .set_labels(['angle','horizontal'])\
                                    .set_panel(pixels_x, 0.1)

        ag_cone = AcquisitionGeometry.create_Cone2D([0,-2], [0,1])\
                                     .set_angles(angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(pixels_x, 0.1)


        
        ag3 = AcquisitionGeometry.create_Parallel3D()\
                                 .set_angles(angles_rad, angle_unit='radian')\
                                 .set_labels(['vertical', 'angle','horizontal'])\
                                 .set_panel((pixels_x,pixels_y), (0.1,0.1))

        ig3 = ag3.get_ImageGeometry()

        ag3_cone = AcquisitionGeometry.create_Cone3D([0,-2,0], [0,1,0])\
                                      .set_angles(angles_rad, angle_unit='radian')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((pixels_x,pixels_y), (0.1,0.1))
        self.ig = ig
        self.ig3 = ig3
        self.ag = ag
        self.ag_deg = ag_deg
        self.ag_cone = ag_cone
        self.ag3 = ag3
        self.ag3_cone = ag3_cone

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_to_astra(self):
        
        #2D parallel radians
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag)

        self.assertEqual(astra_sino['type'],  'parallel')
        self.assertEqual(astra_sino['DetectorCount'], self.ag.pixel_num_h)
        self.assertEqual(astra_sino['DetectorWidth'], self.ag.pixel_size_h)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        #2D cone
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag_cone)

        self.assertEqual(astra_sino['type'], 'fanflat')
        self.assertEqual(astra_sino['DistanceOriginSource'], self.ag_cone.dist_source_center)
        self.assertTrue(astra_sino['DistanceOriginDetector'], self.ag_cone.dist_center_detector)


        #3D parallel
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig3, self.ag3)
        self.assertEqual(astra_sino['type'],  'parallel3d')
        self.assertEqual(astra_sino['DetectorColCount'], self.ag3.pixel_num_h)
        self.assertEqual(astra_sino['DetectorRowCount'], self.ag3.pixel_num_v)
        self.assertEqual(astra_sino['DetectorSpacingX'], self.ag3.pixel_size_h)
        self.assertEqual(astra_sino['DetectorSpacingY'], self.ag3.pixel_size_h)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag3.angles) 

        self.assertEqual(astra_vol['GridColCount'], self.ig3.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig3.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig3.voxel_num_z)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig3.voxel_num_x * self.ig3.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig3.voxel_num_x * self.ig3.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig3.voxel_num_y * self.ig3.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig3.voxel_num_y * self.ig3.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig3.voxel_num_z * self.ig3.voxel_size_z * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig3.voxel_num_z * self.ig3.voxel_size_z * 0.5)

        #3D cone
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig3, self.ag3_cone)
        self.assertEqual(astra_sino['type'], 'cone')
        self.assertEqual(astra_sino['DistanceOriginSource'], self.ag_cone.dist_source_center)
        self.assertEqual(astra_sino['DistanceOriginDetector'], self.ag_cone.dist_center_detector)
        self.assertEqual(astra_sino['DetectorColCount'], self.ag3.pixel_num_h)
        self.assertEqual(astra_sino['DetectorRowCount'], self.ag3.pixel_num_v)
        self.assertEqual(astra_sino['DetectorSpacingX'], self.ag3.pixel_size_h)
        self.assertEqual(astra_sino['DetectorSpacingY'], self.ag3.pixel_size_h)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag3.angles)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_to_astra_vec_3D(self):
        #2D parallel radians
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)

        self.assertEqual(astra_sino['type'],  'parallel3d_vec')
        self.assertEqual(astra_sino['DetectorRowCount'], 1.0)
        self.assertEqual(astra_sino['DetectorColCount'], self.ag.pixel_num_h)

        vectors = np.zeros((3,12),dtype='float64')

        vectors[0][1] = 1.0
        vectors[0][6] = self.ag.pixel_size_h
        vectors[0][11] = self.ag.pixel_size_h

        vectors[1][0] = 1.0
        vectors[1][7] = -self.ag.pixel_size_h
        vectors[1][11] = self.ag.pixel_size_h
    
        vectors[2][1] = -1.0
        vectors[2][6] = -self.ag.pixel_size_h
        vectors[2][11] = self.ag.pixel_size_h

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)


        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], 1)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], - self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], + self.ig.voxel_size_x * 0.5)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #2D cone
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_cone)

        self.assertEqual(astra_sino['type'], 'cone_vec')

        vectors = np.zeros((3,12),dtype='float64')

        pixel_size_v = self.ig.voxel_size_x * self.ag_cone.magnification
        vectors[0][1] = -1 * self.ag_cone.dist_source_center
        vectors[0][4] = self.ag_cone.dist_center_detector
        vectors[0][6] = self.ag_cone.pixel_size_h
        vectors[0][11] = pixel_size_v

        vectors[1][0] = -1 * self.ag_cone.dist_source_center
        vectors[1][3] = self.ag_cone.dist_center_detector
        vectors[1][7] = -self.ag_cone.pixel_size_h
        vectors[1][11] = pixel_size_v

        vectors[2][1] = self.ag_cone.dist_source_center
        vectors[2][4] = -1 * self.ag_cone.dist_center_detector
        vectors[2][6] = -1 * self.ag_cone.pixel_size_h
        vectors[2][11] = pixel_size_v     

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #3D cone
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig3, self.ag3_cone)

        self.assertEqual(astra_sino['type'], 'cone_vec')

        vectors = np.zeros((3,12),dtype='float64')

        vectors[0][1] = -1 * self.ag_cone.dist_source_center
        vectors[0][4] = self.ag_cone.dist_center_detector
        vectors[0][6] = self.ag_cone.pixel_size_h
        vectors[0][11] = self.ag_cone.pixel_size_h

        vectors[1][0] = -1 * self.ag_cone.dist_source_center
        vectors[1][3] = self.ag_cone.dist_center_detector
        vectors[1][7] = -1 * self.ag_cone.pixel_size_h
        vectors[1][11] = self.ag_cone.pixel_size_h

        vectors[2][1] = self.ag_cone.dist_source_center
        vectors[2][4] = -1 * self.ag_cone.dist_center_detector
        vectors[2][6] = -1 * self.ag_cone.pixel_size_h
        vectors[2][11] = self.ag_cone.pixel_size_h       

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)
        