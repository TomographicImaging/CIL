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
import math

from utils import has_astra, initialise_tests
initialise_tests()

if has_astra:
    from cil.plugins.astra.utilities import convert_geometry_to_astra
    from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_2D
    from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D

from utils_projectors import create_cone_flex_default_ig


class TestGeometry_Parallel2D(unittest.TestCase):
    def setUp(self):
        self.pixels_x = 128

        angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        angles_rad = angles_deg * np.pi /180.0

        self.ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles_rad, angle_unit='radian')\
                                .set_labels(['angle','horizontal'])\
                                .set_panel(self.pixels_x, 0.1)

        self.ig = self.ag.get_ImageGeometry()

        self.ag_deg = AcquisitionGeometry.create_Parallel2D()\
                                    .set_angles(angles_deg, angle_unit='degree')\
                                    .set_labels(['angle','horizontal'])\
                                    .set_panel(self.pixels_x, 0.1)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_simple(self):

        #2D parallel radians
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag)

        self.assertEqual(astra_sino['type'],  'parallel')
        self.assertEqual(astra_sino['DetectorCount'], self.ag.pixel_num_h)
        self.assertEqual(astra_sino['DetectorWidth'], self.ag.pixel_size_h)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        # check the image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector2D(self):

        # 2D parallel radians
        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag)

        self.assertEqual(astra_sino['type'],  'parallel_vec')
        self.assertEqual(astra_sino['DetectorCount'], self.ag.pixel_num_h)

        # note the sign of the y values differs from the astra 3D projector case
        vectors = np.zeros((3,6),dtype='float64')

        vectors[0][1] = -1.0
        vectors[0][4] = self.ag.pixel_size_h

        vectors[1][0] = 1.0
        vectors[1][5] = self.ag.pixel_size_h

        vectors[2][1] = 1.0
        vectors[2][4] = -self.ag.pixel_size_h

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # 2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector3D(self):

        # 2D parallel radians
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

        # 2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], 1)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], - self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], + self.ig.voxel_size_x * 0.5)

class TestGeometry_Parallel3D(unittest.TestCase):
    def setUp(self):
        # Define image geometry.
        pixels_x = 128
        pixels_y = 3

        angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        angles_rad = angles_deg * np.pi /180.0

        self.ag = AcquisitionGeometry.create_Parallel3D()\
                                 .set_angles(angles_rad, angle_unit='radian')\
                                 .set_labels(['vertical', 'angle','horizontal'])\
                                 .set_panel((pixels_x,pixels_y), (0.1,0.2))
        
        self.ag_deg = AcquisitionGeometry.create_Parallel3D()\
                                    .set_angles(angles_deg, angle_unit='degree')\
                                    .set_labels(['vertical', 'angle','horizontal'])\
                                    .set_panel((pixels_x,pixels_y), (0.1,0.2))
        
        self.ig = self.ag.get_ImageGeometry()


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_simple(self):

        #3D parallel
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag)
        self.assertEqual(astra_sino['type'],  'parallel3d')
        self.assertEqual(astra_sino['DetectorColCount'], self.ag.pixel_num_h)
        self.assertEqual(astra_sino['DetectorRowCount'], self.ag.pixel_num_v)
        self.assertEqual(astra_sino['DetectorSpacingX'], self.ag.pixel_size_h)
        self.assertEqual(astra_sino['DetectorSpacingY'], self.ag.pixel_size_v)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector2D(self):

        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag)
        self.assertEqual(astra_sino['type'],  'parallel_vec')
        self.assertEqual(astra_sino['DetectorCount'], self.ag.pixel_num_h)

        vectors = np.zeros((3,6),dtype='float64')

        vectors[0][1] = -1.0
        vectors[0][4] = self.ag.pixel_size_h

        vectors[1][0] = 1.0
        vectors[1][5] = self.ag.pixel_size_h

        vectors[2][1] = 1.0
        vectors[2][4] = -self.ag.pixel_size_h

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector3D(self):

        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)
        self.assertEqual(astra_sino['type'],  'parallel3d_vec')
        self.assertEqual(astra_sino['DetectorRowCount'], self.ag.pixel_num_v)
        self.assertEqual(astra_sino['DetectorColCount'], self.ag.pixel_num_h)


        #ray : the ray direction
        #d : the center of the detector
        #u : the vector from detector pixel (0,0) to (0,1)
        #v : the vector from detector pixel (0,0) to (1,0)

        vectors = np.zeros((3,12),dtype='float64')

        vectors[0][1] = 1.0
        vectors[0][6] = self.ag.pixel_size_h
        vectors[0][11] = self.ag.pixel_size_v

        vectors[1][0] = 1.0
        vectors[1][7] = -self.ag.pixel_size_h
        vectors[1][11] = self.ag.pixel_size_v

        vectors[2][1] = -1.0
        vectors[2][6] = -self.ag.pixel_size_h
        vectors[2][11] = self.ag.pixel_size_v

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)

class TestGeometry_Cone2D(unittest.TestCase):
    def setUp(self):
        pixels_x = 128

        angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        angles_rad = angles_deg * np.pi /180.0


        self.ag = AcquisitionGeometry.create_Cone2D([0,-2], [0,1])\
                                     .set_angles(angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(pixels_x, 0.1)

        self.ig = self.ag.get_ImageGeometry()

        self.ag_deg = AcquisitionGeometry.create_Cone2D([0,-2], [0,1])\
                                     .set_angles(angles_deg, angle_unit='degree')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(pixels_x, 0.1)
        


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_simple(self):
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag)

        self.assertEqual(astra_sino['type'], 'fanflat')
        self.assertEqual(astra_sino['DistanceOriginSource'], self.ag.dist_source_center)
        self.assertTrue(astra_sino['DistanceOriginDetector'], self.ag.dist_center_detector)

        self.assertEqual(astra_sino['DetectorCount'], self.ag.pixel_num_h)
        self.assertEqual(astra_sino['DetectorWidth'], self.ag.pixel_size_h)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        # check the image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector2D(self):
        #2D cone
        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag)

        self.assertEqual(astra_sino['type'], 'fanflat_vec')

        vectors = np.zeros((3,6),dtype='float64')

        # note the sign of the y values differs from the astra 3D projector case

        pixel_size_v = self.ig.voxel_size_x * self.ag.magnification
        vectors[0][1] = 1 * self.ag.dist_source_center
        vectors[0][3] = -self.ag.dist_center_detector
        vectors[0][4] = self.ag.pixel_size_h

        vectors[1][0] = -1 * self.ag.dist_source_center
        vectors[1][2] = self.ag.dist_center_detector
        vectors[1][5] = self.ag.pixel_size_h

        vectors[2][1] = -self.ag.dist_source_center
        vectors[2][3] = 1 * self.ag.dist_center_detector
        vectors[2][4] = -1 * self.ag.pixel_size_h

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # check image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector3D(self):
        #2D cone
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)

        self.assertEqual(astra_sino['type'], 'cone_vec')

        vectors = np.zeros((3,12),dtype='float64')

        pixel_size_v = self.ig.voxel_size_x * self.ag.magnification
        vectors[0][1] = -1 * self.ag.dist_source_center
        vectors[0][4] = self.ag.dist_center_detector
        vectors[0][6] = self.ag.pixel_size_h
        vectors[0][11] = pixel_size_v

        vectors[1][0] = -1 * self.ag.dist_source_center
        vectors[1][3] = self.ag.dist_center_detector
        vectors[1][7] = -self.ag.pixel_size_h
        vectors[1][11] = pixel_size_v

        vectors[2][1] = self.ag.dist_source_center
        vectors[2][4] = -1 * self.ag.dist_center_detector
        vectors[2][6] = -1 * self.ag.pixel_size_h
        vectors[2][11] = pixel_size_v

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #2D parallel degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # check image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)


class TestGeometry_Cone3D(unittest.TestCase):
    def setUp(self):
        pixels_x = 128
        pixels_y = 3

        angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        angles_rad = angles_deg * np.pi /180.0
        source_position = [0,-2,0]
        detector_position = [0,1,0]

        self.ag = AcquisitionGeometry.create_Cone3D(source_position, detector_position)\
                                      .set_angles(angles_rad, angle_unit='radian')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((pixels_x,pixels_y), (0.1,0.2))
        
        self.ag_deg = AcquisitionGeometry.create_Cone3D(source_position, detector_position)\
                                      .set_angles(angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((pixels_x,pixels_y), (0.1,0.2))
        
        self.ig = self.ag.get_ImageGeometry()

        source_position_set = []
        detector_position_set = []
        detector_direction_x_set = []
        detector_direction_y_set = []

        for angle in angles_rad:
            rotation_matrix = np.eye(3)
            rotation_matrix[0,0] = rotation_matrix[1,1] = math.cos(-angle)
            rotation_matrix[0,1] = -math.sin(-angle)
            rotation_matrix[1,0] = math.sin(-angle)

            source_position_set.append(rotation_matrix.dot(source_position))
            detector_position_set.append(rotation_matrix.dot(detector_position))
            detector_direction_x_set.append(rotation_matrix.dot([1,0,0]))
            detector_direction_y_set.append(rotation_matrix.dot([0,0,1]))


        self.ag_Flex = AcquisitionGeometry.create_Cone3D_Flex(source_position_set=source_position_set,\
                                                         detector_position_set=detector_position_set,\
                                                         detector_direction_x_set=detector_direction_x_set,\
                                                         detector_direction_y_set=detector_direction_y_set)\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((pixels_x,pixels_y), (0.1,0.2))
        

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_simple(self):
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag)
        self.assertEqual(astra_sino['type'], 'cone')
        self.assertEqual(astra_sino['DistanceOriginSource'], self.ag.dist_source_center)
        self.assertEqual(astra_sino['DistanceOriginDetector'], self.ag.dist_center_detector)
        self.assertEqual(astra_sino['DetectorColCount'], self.ag.pixel_num_h)
        self.assertEqual(astra_sino['DetectorRowCount'], self.ag.pixel_num_v)
        self.assertEqual(astra_sino['DetectorSpacingX'], self.ag.pixel_size_h)
        self.assertEqual(astra_sino['DetectorSpacingY'], self.ag.pixel_size_v)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        # degrees
        astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['ProjectionAngles'], -self.ag.angles)

        # check the image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector2D(self):

        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag)
        self.assertEqual(astra_sino['type'], 'fanflat_vec')

        # note the sign of the y values differs from the astra 3D projector case
        vectors = np.zeros((3,6),dtype='float64')
        vectors[0][1] = 1 * self.ag.dist_source_center
        vectors[0][3] = -self.ag.dist_center_detector
        vectors[0][4] = self.ag.pixel_size_h

        vectors[1][0] = -1 * self.ag.dist_source_center
        vectors[1][2] = self.ag.dist_center_detector
        vectors[1][5] = 1 * self.ag.pixel_size_h

        vectors[2][1] = -self.ag.dist_source_center
        vectors[2][3] = 1 * self.ag.dist_center_detector
        vectors[2][4] = -1 * self.ag.pixel_size_h

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector3D(self):

        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)
        self.assertEqual(astra_sino['type'], 'cone_vec')

        vectors = np.zeros((3,12),dtype='float64')
        vectors[0][1] = -1 * self.ag.dist_source_center
        vectors[0][4] = self.ag.dist_center_detector
        vectors[0][6] = self.ag.pixel_size_h
        vectors[0][11] = self.ag.pixel_size_v

        vectors[1][0] = -1 * self.ag.dist_source_center
        vectors[1][3] = self.ag.dist_center_detector
        vectors[1][7] = -1 * self.ag.pixel_size_h
        vectors[1][11] = self.ag.pixel_size_v

        vectors[2][1] = self.ag.dist_source_center
        vectors[2][4] = -1 * self.ag.dist_center_detector
        vectors[2][6] = -1 * self.ag.pixel_size_h
        vectors[2][11] = self.ag.pixel_size_v

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        #degrees
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_deg)
        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_Flex(self):
        """
        Checks the Flex convention agrees with the standard cone geometry 3D
        """

        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag_Flex)
        self.assertEqual(astra_sino['type'], 'cone_vec')

        vectors = np.zeros((3,12),dtype='float64')
        vectors[0][1] = -1 * self.ag.dist_source_center
        vectors[0][4] = self.ag.dist_center_detector
        vectors[0][6] = self.ag.pixel_size_h
        vectors[0][11] = self.ag.pixel_size_v

        vectors[1][0] = -1 * self.ag.dist_source_center
        vectors[1][3] = self.ag.dist_center_detector
        vectors[1][7] = -1 * self.ag.pixel_size_h
        vectors[1][11] = self.ag.pixel_size_v

        vectors[2][1] = self.ag.dist_source_center
        vectors[2][4] = -1 * self.ag.dist_center_detector
        vectors[2][6] = -1 * self.ag.pixel_size_h
        vectors[2][11] = self.ag.pixel_size_v

        np.testing.assert_allclose(astra_sino['Vectors'], vectors, atol=1e-6)

        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5)
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5)



class TestGeometry_Cone3D_Flex(unittest.TestCase):
    def setUp(self):
        self.pixels_x = 128
        self.pixels_y = 3

        # generate a set of source and detector positions
        self.source_position_set = [
            [60.39,	90.29,	63.12],
            [34.64,	24.1,	-66.48],
            [1.24,	66.74,	-98.15],
        ]

        self.detector_position_set = [
            [6579.59,	6069.16,	-9907.19],
            [4984.46,	5533.88,	1186.63],
            [-7528.65,	2598.33,	-7634.18],
        ]

        # detector direction x and y must be orthogonal
        self.detector_direction_x_set = [
            [1.2,	7.91,	-4.95],
            [4.66,	6.75,	-0.77],
            [-7.12,	4.69,	0.59]
        ]

        vec = [
            [-4.97,	5.62,	-8.26],
            [-0.97,	-6.93,	7.65],
            [-5.5,	7.43,	-5.14]
        ]

        self.detector_direction_y_set = []
        for i in range(3):
            self.detector_direction_x_set[i] = self.detector_direction_x_set[i] / np.linalg.norm(self.detector_direction_x_set[i])
            detector_direction_y = np.cross(self.detector_direction_x_set[i], vec[i])
            self.detector_direction_y_set.append(detector_direction_y / np.linalg.norm(detector_direction_y))

        self.ag = AcquisitionGeometry.create_Cone3D_Flex(source_position_set=self.source_position_set,\
                                                         detector_position_set=self.detector_position_set,\
                                                         detector_direction_x_set=self.detector_direction_x_set,\
                                                         detector_direction_y_set=self.detector_direction_y_set) \
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.pixels_x, self.pixels_y), (0.1,0.2))
        
        self.ig = create_cone_flex_default_ig(self.ag)

  

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_simple(self):

        with self.assertRaises(ValueError):
            astra_vol, astra_sino = convert_geometry_to_astra(self.ig, self.ag)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector2D(self):
        with self.assertRaises(ValueError):
            astra_vol, astra_sino = convert_geometry_to_astra_vec_2D(self.ig, self.ag)

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector3D(self):

        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)
        self.assertEqual(astra_sino['type'], 'cone_vec')

        for i in range(3):
            np.testing.assert_allclose(astra_sino['Vectors'][i][0:3], self.source_position_set[i], atol=1e-6)
            np.testing.assert_allclose(astra_sino['Vectors'][i][3:6], self.detector_position_set[i], atol=1e-6)
            np.testing.assert_allclose(astra_sino['Vectors'][i][6:9], self.detector_direction_x_set[i] * self.ag.pixel_size_h, atol=1e-6)
            np.testing.assert_allclose(astra_sino['Vectors'][i][9:12], self.detector_direction_y_set[i] * self.ag.pixel_size_v, atol=1e-6)
            
        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        origin = self.ag.config.system.volume_centre.position
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5 + origin[0])
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5 + origin[0])
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5 + origin[1])
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5 + origin[1])
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5 + origin[2])
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5 + origin[2])


        # with manual centre
        origin = self.ag.config.system.volume_centre.position = [1,-2,3.5]
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)

        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5 + origin[0])
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5 + origin[0])
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5 + origin[1])
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5 + origin[1])
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5 + origin[2])
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5 + origin[2])


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_convert_geometry_vector_data_origin(self):

        self.ag.set_panel((self.pixels_x, self.pixels_y), (0.1,0.2),origin="top-right")
        astra_vol, astra_sino = convert_geometry_to_astra_vec_3D(self.ig, self.ag)
        self.assertEqual(astra_sino['type'], 'cone_vec')

        for i in range(3):
            np.testing.assert_allclose(astra_sino['Vectors'][i][0:3], self.source_position_set[i], atol=1e-6)
            np.testing.assert_allclose(astra_sino['Vectors'][i][3:6], self.detector_position_set[i], atol=1e-6)
            np.testing.assert_allclose(astra_sino['Vectors'][i][6:9], -self.detector_direction_x_set[i] * self.ag.pixel_size_h, atol=1e-6)
            np.testing.assert_allclose(astra_sino['Vectors'][i][9:12], -self.detector_direction_y_set[i] * self.ag.pixel_size_v, atol=1e-6)
            
        # image geometry
        self.assertEqual(astra_vol['GridColCount'], self.ig.voxel_num_x)
        self.assertEqual(astra_vol['GridRowCount'], self.ig.voxel_num_y)
        self.assertEqual(astra_vol['GridSliceCount'], self.ig.voxel_num_z)

        origin = self.ag.config.system.volume_centre.position
        self.assertEqual(astra_vol['option']['WindowMinX'], -self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5 + origin[0])
        self.assertEqual(astra_vol['option']['WindowMaxX'], self.ig.voxel_num_x * self.ig.voxel_size_x * 0.5 + origin[0])
        self.assertEqual(astra_vol['option']['WindowMinY'], -self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5 + origin[1])
        self.assertEqual(astra_vol['option']['WindowMaxY'], self.ig.voxel_num_y * self.ig.voxel_size_y * 0.5 + origin[1])
        self.assertEqual(astra_vol['option']['WindowMinZ'], -self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5 + origin[2])
        self.assertEqual(astra_vol['option']['WindowMaxZ'], self.ig.voxel_num_z * self.ig.voxel_size_z * 0.5 + origin[2])

