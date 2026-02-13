#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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
from cil.utilities.dataexample import SIMULATED_PARALLEL_BEAM_DATA, SIMULATED_CONE_BEAM_DATA, SIMULATED_SPHERE_VOLUME
from skimage.transform.radon_transform import _get_fourier_filter as skimage_get_fourier_filter
import numpy as np
from utils import has_tigre, has_ipp, has_astra, has_nvidia, has_matplotlib, has_skimage, initialise_tests

from cil.recon.Reconstructor import Reconstructor # checks on baseclass
from cil.recon.FBP import GenericFilteredBackProjection # checks on baseclass
from cil.recon import FDK, FBP
from cil.processors import Slicer



initialise_tests()

if has_tigre:
    from cil.plugins.tigre import FBP as FBP_tigre

if has_astra:
    from cil.plugins.astra import ProjectionOperator as ProjectionOperator_astra
    from cil.plugins.astra import FBP as FBP_astra


from cil.recon.weighted_fbp import get_weights_for_FBP

class Test_get_weights_for_FBP(unittest.TestCase):

    def setUp(self):
        self.acq_data_360 = SIMULATED_PARALLEL_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()

        

        self.acq_data_360=np.log(self.acq_data_360)
        self.acq_data_360*=-1.0

        self.acq_data_180 = Slicer(roi={'angle': (0,150,1)})(self.acq_data_360)

        self.ig = self.img_data.geometry
        self.ag = self.acq_data_360.geometry

    def test_weights_with_no_duplicates_and_equal_angular_spread(self):
        for data in [self.acq_data_360, self.acq_data_180]:
            weights = get_weights_for_FBP(data)
            # TODO: look into tolerance etc
            np.testing.assert_allclose(weights, np.ones_like(weights), atol=1e-4)
            # Check normalisation of weights:
            np.testing.assert_almost_equal(sum(weights), len(weights), decimal=1e-6)

    def test_weights_with_duplicates(self):
        for data in [self.acq_data_360, self.acq_data_180]:
            # Create one duplicate angle:
            acq_data_with_duplicate = data.copy()
            angles = np.insert(acq_data_with_duplicate.geometry.angles, 1, acq_data_with_duplicate.geometry.angles[0])
            acq_data_with_duplicate.array = np.insert(acq_data_with_duplicate.array, 1, acq_data_with_duplicate.array[0], axis=0)
            
            # Create an angle which has 3 projections at the same angle:
            angles = np.insert(angles, 30, angles[30])
            angles = np.insert(angles, 30, angles[30])
            acq_data_with_duplicate.geometry.set_angles(angles)
            acq_data_with_duplicate.array = np.insert(acq_data_with_duplicate.array, 30, acq_data_with_duplicate.array[30], axis=0)
            acq_data_with_duplicate.array = np.insert(acq_data_with_duplicate.array, 30, acq_data_with_duplicate.array[30], axis=0)

            weights = get_weights_for_FBP(acq_data_with_duplicate)

            np.testing.assert_allclose(weights[0], weights[1], atol=1e-4)
            np.testing.assert_allclose(weights[0], weights[2]/2.0, atol=1e-4)
            np.testing.assert_allclose([weights[32], weights[31]], weights[30], atol=1e-4)
            np.testing.assert_allclose(weights[30], weights[2]/3.0, atol=1e-4)
            np.testing.assert_allclose(weights[2:30], weights[2], atol=1e-4)
            np.testing.assert_allclose(weights[33:], weights[2], atol=1e-4)
            # Check normalisation of weights:
            np.testing.assert_almost_equal(sum(weights), len(weights)-3, decimal=1e-6) # subtract 3 here as we have 3 duplicate angles

    def test_weights_with_non_uniform_angular_spread(self):
        angular_range=[360,180]
        for i, data in enumerate([self.acq_data_360, self.acq_data_180]):
            # create data with final 25% of angles missing:
            final_angle_index = int(angular_range[i]*0.75)
            acq_data_non_uniform = Slicer(roi={'angle': (0,final_angle_index+1,1)})(data.copy())
            
            standard_gap = acq_data_non_uniform.geometry.angles[1] - acq_data_non_uniform.geometry.angles[0]
            large_gap = (acq_data_non_uniform.geometry.angles[0] - acq_data_non_uniform.geometry.angles[final_angle_index] )%angular_range[i]
            
            weights = get_weights_for_FBP(acq_data_non_uniform)
            np.testing.assert_allclose(weights[1:final_angle_index], weights[1], atol=1e-4)
            np.testing.assert_almost_equal(weights[0], weights[final_angle_index], decimal=1e-6)

            np.testing.assert_almost_equal(weights[1],(standard_gap)/(angular_range[i]/len(acq_data_non_uniform.geometry.angles)), decimal=1e-4)
            np.testing.assert_almost_equal(weights[0], (large_gap+standard_gap)/2.0/(angular_range[i]/len(acq_data_non_uniform.geometry.angles)) , decimal=1e-4)

            # Check normalisation of weights:
            np.testing.assert_almost_equal(sum(weights), len(weights), decimal=1e-6)


if __name__ == '__main__':
    unittest.main()