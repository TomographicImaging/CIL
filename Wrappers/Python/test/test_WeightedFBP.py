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


from cil.recon.weighted_fbp import get_weights_for_FBP, calculate_angular_sampling_weights

class Test_get_weights_for_FBP(unittest.TestCase):

    def setUp(self):
        self.acq_data_360 = SIMULATED_PARALLEL_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()
        self.acq_data_360=np.log(self.acq_data_360)
        self.acq_data_360*=-1.0

        self.acq_data_180 = Slicer(roi={'angle': (0,150,1)})(self.acq_data_360)

        self.ig = self.img_data.geometry
        self.ag = self.acq_data_360.geometry

    def test_weights_with_equal_angular_spread(self):
        for data in [self.acq_data_180, self.acq_data_360]:
            weights = calculate_angular_sampling_weights(data)
            np.testing.assert_allclose(weights, np.ones_like(weights), atol=1e-4)
            
            # Check normalisation of weights:
            np.testing.assert_almost_equal(sum(weights), len(weights), decimal=6)


    def test_weights_with_two_sets_of_duplicates_180(self):
        data = self.acq_data_180 
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

        weights = calculate_angular_sampling_weights(acq_data_with_duplicate)

        np.testing.assert_allclose(weights[0], weights[1], atol=1e-4)
        np.testing.assert_allclose(weights[0], weights[2]/2.0, atol=1e-4)
        np.testing.assert_allclose([weights[32], weights[31]], weights[30], atol=1e-4)
        np.testing.assert_allclose(weights[30], weights[2]/3.0, atol=1e-4)
        np.testing.assert_allclose(weights[2:30], weights[2], atol=1e-4)
        np.testing.assert_allclose(weights[33:], weights[2], atol=1e-4)
    # def test_weights_with_duplicates_through_even_proj_around_360(self):
    #     data = self.acq_data_360
    #     weights = calculate_angular_sampling_weights(data)



    # def test_weights_with_non_uniform_angular_spread(self):
    #     for data in [self.acq_data_180, self.acq_data_360]:
    #         data = Slicer(roi={'angle':(None, None, 3)})(data)
    #         weights = calculate_angular_sampling_weights(data)
            
    #         # Check normalisation of weights:
    #         np.testing.assert_almost_equal(sum(weights), len(weights), decimal=6)
    #     pass
        
    def test_weights_with_missing_wedge_at_end(self):
        # 180 degree domain -----------------------------------------------------
        # As data is parallel beam, should get same result whether 
        # angular_domain is None or 180:
        for angular_domain in [180, None]:

            # 180 degrees of data:

            # create data with final 25% of angles missing:
            data = self.acq_data_180
            final_angle_index = int(150*0.75)
            acq_data_non_uniform = Slicer(roi={'angle': (0,final_angle_index,1)})(data.copy())
            
            weights = calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain=angular_domain)
            # print(weights[-10:-1])
            # print(weights[0:10])
            # print(weights[weights!=weights[0]])
            np.testing.assert_allclose(weights, weights[0], atol=1e-4)
            np.testing.assert_almost_equal(weights[0], 1, decimal=6)
        # 360 degree domain --------------------------------------------------------
        angular_domain = 360
        data = self.acq_data_360
        final_angle_index = int(300*0.75)
        acq_data_non_uniform = Slicer(roi={'angle': (0,final_angle_index+1,1)})(data.copy())
        
        weights = calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain=angular_domain)
        # print(weights[-10:-1])
        # print(weights[0:10])
        # print(weights[weights!=weights[0]])
        #np.testing.assert_allclose(weights, weights[0], atol=1e-4)
        np.testing.assert_almost_equal(weights[0], 1, decimal=6)

    def test_weights_with_missing_wedge_at_start(self):
        angular_range=[360,180]
        for i, data in enumerate([self.acq_data_360, self.acq_data_180]):
            # create data with final 25% of angles missing:
            first_angle_index = int(len(data.geometry.angles)*0.75)
            acq_data_non_uniform = Slicer(roi={'angle': (first_angle_index,-1,1)})(data.copy())
            
            weights = calculate_angular_sampling_weights(acq_data_non_uniform)
            np.testing.assert_allclose(weights, weights[0], atol=1e-4)
            np.testing.assert_almost_equal(weights[0], 1, decimal=1e-6)

    def test_weights_with_missing_wedges_at_centre(self):
        angular_range = [360, 180]
        for i, data in enumerate([self.acq_data_360, self.acq_data_180]):
            # create data with middle sections removed (keep first 25% and last 25%):
            total_angles = len(data.geometry.angles)
            quarter_point = total_angles // 4
            three_quarter_point = 3 * total_angles // 4
            
            # Keep first quarter and last quarter
            indices_to_keep = list(range(quarter_point)) + list(range(three_quarter_point, total_angles))
            
            acq_data_non_uniform = data.copy()
            acq_data_non_uniform.array = acq_data_non_uniform.array[indices_to_keep]
            acq_data_non_uniform.geometry.set_angles(data.geometry.angles[indices_to_keep])
            
            weights = calculate_angular_sampling_weights(acq_data_non_uniform)
            
            # Check that weights are approximately equal (more lenient tolerance due to gap)
            np.testing.assert_allclose(weights, weights[0], atol=1e-4)
            
            # Check normalization
            np.testing.assert_almost_equal(sum(weights), len(weights))

        # np.testing.assert_almost_equal(weights[1],(standard_gap)/(angular_range[i]/len(acq_data_non_uniform.geometry.angles)), decimal=1e-4)
        # np.testing.assert_almost_equal(weights[0], (large_gap+standard_gap)/2.0/(angular_range[i]/len(acq_data_non_uniform.geometry.angles)) , decimal=1e-4)

        # # Check normalisation of weights:

    def test_weights_with_non_uniform_angular_spread(self):
        # Every fourth angle is missing
        for i, data in enumerate([ self.acq_data_180]):
            # create data with every fourth angle missing:
            total_angles = len(data.geometry.angles)
            indices_to_keep = [idx for idx in range(total_angles) if idx % 4 != 3]
            
            acq_data_non_uniform = data.copy()
            acq_data_non_uniform.array = acq_data_non_uniform.array[indices_to_keep]
            acq_data_non_uniform.geometry.set_angles(data.geometry.angles[indices_to_keep])
            
            weights = calculate_angular_sampling_weights(acq_data_non_uniform)

            # expect weights to be in relative pattern:
            # [x, 2,3,3,2,3,3,2,3,3], etc
            # exclude end weights
            # np.testing.assert_allclose(weights[3::3], weights[1]*3/2, atol=1e-4)
            # # make copy of weights with all third indices removed:
            # #weights_third_removed = 

            np.testing.assert_allclose(weights[1::3], weights[2]*2/3, atol=1e-4)
            np.testing.assert_allclose(weights[2::3], weights[1]*3/2, atol=1e-4)
            np.testing.assert_allclose(weights[3::3], weights[1]*3/2, atol=1e-4)


if __name__ == '__main__':
    unittest.main()