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
from utils import has_astra, has_nvidia, initialise_tests
from utils_projectors import TestCommon_ProjectionOperatorBlockOperator
from cil.utilities import dataexample

initialise_tests()

if has_astra:
    from cil.plugins.astra.operators import AstraProjector2D, AstraProjector3D
    from cil.plugins.astra.operators import ProjectionOperator

class TestAstraProjectors(unittest.TestCase):
    def setUp(self):

        N = 128
        angles = np.linspace(0, np.pi, 180, dtype='float32')

        self.ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['angle', 'horizontal'])

        self.ig = self.ag.get_ImageGeometry()


        self.ag3 = AcquisitionGeometry.create_Parallel3D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N, N), (0.1, 0.1))\
                                .set_labels(['vertical', 'angle', 'horizontal'])

        self.ig3 = self.ag3.get_ImageGeometry()

        self.ag_channel = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['channel','angle', 'horizontal'])\
                                .set_channels(5)


        self.ig_channel = self.ag_channel.get_ImageGeometry()


        self.ag3_channel = AcquisitionGeometry.create_Parallel3D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N, N), (0.1, 0.1))\
                                .set_labels(['channel','vertical', 'angle', 'horizontal'])\
                                .set_channels(5)

        self.ig3_channel = self.ag3_channel.get_ImageGeometry()

        self.norm = 14.85


    def foward_projection(self, A, ig, ag):
        image_data = ig.allocate(None)
        image_data.fill(1)

        acq_data_0 = A.direct(image_data)
        acq_data_1 = ag.allocate(0)
        id_1 = id(acq_data_1)
        A.direct(image_data, out=acq_data_1)
        id_2 = id(acq_data_1)

        self.assertEqual(id_1,id_2)
        self.assertAlmostEqual(acq_data_0.array.item(0), 12.800, places=3) #check not zeros
        np.testing.assert_allclose(acq_data_0.as_array(),acq_data_1.as_array())


    def backward_projection(self, A, ig, ag):
        acq_data = ag.allocate(None)
        acq_data.fill(1)

        image_data_0 = A.adjoint(acq_data)
        image_data_1 = ig.allocate(0)
        id_1 = id(image_data_1)
        A.adjoint(acq_data, out=image_data_1)
        id_2 = id(image_data_1)

        self.assertEqual(id_1,id_2)
        self.assertAlmostEqual(image_data_0.array.item(0), 9.14, places=2) #check not zeros
        np.testing.assert_allclose(image_data_0.as_array(),image_data_1.as_array())

    def projector_norm(self, A):
        n = A.norm()
        self.assertAlmostEqual(n, self.norm, places=2)


    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_AstraProjector2D_cpu(self):

        A = AstraProjector2D(self.ig, self.ag, device = 'cpu')

        self.foward_projection(A,self.ig, self.ag)
        self.backward_projection(A,self.ig, self.ag)
        self.projector_norm(A)


    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def test_AstraProjector2D_gpu(self):

        A = AstraProjector2D(self.ig, self.ag, device = 'gpu')

        self.foward_projection(A,self.ig, self.ag)
        self.backward_projection(A,self.ig, self.ag)
        self.projector_norm(A)


    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def test_AstraProjector3D_2Ddata(self):

        A = AstraProjector3D(self.ig, self.ag)

        self.foward_projection(A,self.ig, self.ag)
        self.backward_projection(A,self.ig, self.ag)
        self.projector_norm(A)

        ag_2 = self.ag.copy()
        ag_2.dimension_labels = ['horizontal','angle']
        with self.assertRaises(ValueError):
            A = AstraProjector3D(self.ig, ag_2)

        ig_2 = self.ig3.copy()
        ig_2.dimension_labels = ['horizontal_x','horizontal_y']
        with self.assertRaises(ValueError):
            A = AstraProjector3D(ig_2, self.ag)


    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def test_AstraProjector3D_3Ddata(self):
        # test exists
        A = AstraProjector3D(self.ig3, self.ag3)

        self.foward_projection(A,self.ig3, self.ag3)
        self.backward_projection(A,self.ig3, self.ag3)
        self.projector_norm(A)

        ag3_2 = self.ag3.copy()
        ag3_2.dimension_labels = ['angle','vertical','horizontal']
        with self.assertRaises(ValueError):
            A3 = AstraProjector3D(self.ig3, ag3_2)

        ig3_2 = self.ig3.copy()
        ig3_2.dimension_labels = ['horizontal_y','vertical','horizontal_x']
        with self.assertRaises(ValueError):
            A3 = AstraProjector3D(ig3_2, self.ag3)

class TestASTRA_BlockOperator(unittest.TestCase, TestCommon_ProjectionOperatorBlockOperator):
    def setUp(self):
        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data = data.get_slice(vertical='centre')
        ig = self.data.geometry.get_ImageGeometry()
        self.datasplit = self.data.partition(10, 'sequential')


        K = ProjectionOperator(image_geometry=ig, acquisition_geometry=self.datasplit.geometry)
        A = ProjectionOperator(image_geometry=ig, acquisition_geometry=self.data.geometry)
        self.projectionOperator = (A, K)

    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA and a GPU")
    def test_partition(self):
        self.partition_test()
