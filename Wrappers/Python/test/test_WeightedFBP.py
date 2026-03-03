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

from unittest.mock import patch


from cil.recon.weighted_fbp import _calculate_angular_sampling_weights, normalise_weights_for_FBP, calculate_angular_sampling_weights_parallel, calculate_angular_sampling_weights_cone

class Test_calculate_angular_sampling_weights(unittest.TestCase):

    def setUp(self):
        self.acq_data_360 = SIMULATED_PARALLEL_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()
        self.acq_data_360=np.log(self.acq_data_360)
        self.acq_data_360*=-1.0

        self.acq_data_180 = Slicer(roi={'angle': (0,150,1)})(self.acq_data_360)

        self.ig = self.img_data.geometry
        self.ag = self.acq_data_360.geometry

    def test_weights_with_equal_angular_spread(self):
        angular_ranges=[360,180]
        for data, angular_domain in zip([self.acq_data_180, self.acq_data_360], angular_ranges):
            weights = _calculate_angular_sampling_weights(data, angular_domain=angular_domain)
            np.testing.assert_allclose(weights, weights[0], atol=1e-4)

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

        weights = _calculate_angular_sampling_weights(acq_data_with_duplicate, angular_domain=180)

        np.testing.assert_allclose(weights[0], weights[1], atol=1e-4)
        np.testing.assert_allclose(weights[0], weights[2]/2.0, atol=1e-4)
        np.testing.assert_allclose([weights[32], weights[31]], weights[30], atol=1e-4)
        np.testing.assert_allclose(weights[30], weights[2]/3.0, atol=1e-4)
        np.testing.assert_allclose(weights[2:30], weights[2], atol=1e-4)
        np.testing.assert_allclose(weights[33:], weights[2], atol=1e-4)

        
    def test_weights_with_missing_wedge_at_end(self):
        # 180 degree domain -----------------------------------------------------
        # create data with final 25% of angles missing:
        data = self.acq_data_180
        final_angle_index = int(150*0.75)
        acq_data_non_uniform = Slicer(roi={'angle': (0,final_angle_index,1)})(data.copy())
        # As data is parallel beam, should get same result whether 

        angular_domain=180
        # as 'forward/back' is default behaviour, should get same result whether we specify it or not:
        wedge_behaviour = 'forward/back'
        weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain=angular_domain, wedge_behaviour=wedge_behaviour)
        np.testing.assert_allclose(weights, weights[0], atol=1e-4)
        wedge_behaviour = 'max_gap'
        max_gap = 2
        weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain=angular_domain, wedge_behaviour=wedge_behaviour, max_gap=max_gap)
        np.testing.assert_allclose(weights[1:-1], weights[1], atol=1e-4)
        np.testing.assert_allclose(weights[-1], max_gap, atol=1e-4)
        np.testing.assert_allclose(weights[0], max_gap, atol=1e-4)
        # 360 degree domain --------------------------------------------------------
        angular_domain = 360
        data = self.acq_data_360
        final_angle_index = int(300*0.75)
        acq_data_non_uniform = Slicer(roi={'angle': (0,final_angle_index+1,1)})(data.copy())
        
        weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain)
        np.testing.assert_allclose(weights, weights[0], atol=1e-4)

        wedge_behaviour = 'max_gap'
        max_gap = 2
        weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain, wedge_behaviour=wedge_behaviour, max_gap=max_gap)
        np.testing.assert_allclose(weights[1:-1], weights[1], atol=1e-4)
        np.testing.assert_allclose(weights[-1], max_gap, atol=1e-4)
        np.testing.assert_allclose(weights[0], max_gap, atol=1e-4)

    def test_weights_with_missing_wedge_at_start(self):
        angular_range=[360,180]
        for i, data in enumerate([self.acq_data_360, self.acq_data_180]):
            angular_domain = angular_range[i]
            # create data with final 25% of angles missing:
            first_angle_index = int(len(data.geometry.angles)*0.75)
            acq_data_non_uniform = Slicer(roi={'angle': (first_angle_index,-1,1)})(data.copy())
            
            weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain)
            np.testing.assert_allclose(weights, weights[0], atol=1e-4)

            wedge_behaviour = 'max_gap'
            max_gap = 2
            weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain, wedge_behaviour=wedge_behaviour, max_gap=max_gap)
            np.testing.assert_allclose(weights[1:-1], weights[1], atol=1e-4)
            np.testing.assert_allclose(weights[-1], max_gap, atol=1e-4)
            np.testing.assert_allclose(weights[0], max_gap, atol=1e-4)

            # TODO: add test with default value of max_gap

    def test_weights_with_missing_wedge_at_centre(self):
        angular_range = [360, 180]
        for i, data in enumerate([self.acq_data_360, self.acq_data_180]):
            angular_domain = angular_range[i]
            # create data with middle section removed (keep first 25% and last 25%):
            total_angles = len(data.geometry.angles)
            quarter_point = total_angles // 4
            three_quarter_point = 3 * total_angles // 4
            
            # Keep first quarter and last quarter
            indices_to_keep = list(range(quarter_point)) + list(range(three_quarter_point, total_angles))
            
            acq_data_non_uniform = data.copy()
            acq_data_non_uniform.array = acq_data_non_uniform.array[indices_to_keep]
            acq_data_non_uniform.geometry.set_angles(data.geometry.angles[indices_to_keep])
            
            weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain)
            
            # Check that weights are approximately equal (more lenient tolerance due to gap)
            np.testing.assert_allclose(weights, weights[0], atol=1e-4)

            wedge_behaviour = 'max_gap'
            max_gap = 2
            weights = _calculate_angular_sampling_weights(acq_data_non_uniform, angular_domain, wedge_behaviour=wedge_behaviour, max_gap=max_gap)
            np.testing.assert_allclose(weights[0:quarter_point-1], weights[1], atol=1e-4)
            np.testing.assert_allclose(weights[quarter_point-1], max_gap, atol=1e-4)
            np.testing.assert_allclose(weights[quarter_point], max_gap, atol=1e-4)
            np.testing.assert_allclose(weights[three_quarter_point:], max_gap, atol=1e-4)


    def test_weights_with_non_uniform_angular_spread(self):
        # Every fourth angle is missing
        data = self.acq_data_180
        # create data with every fourth angle missing:
        total_angles = len(data.geometry.angles)
        indices_to_keep = [idx for idx in range(total_angles) if idx % 4 != 3]
        
        acq_data_non_uniform = data.copy()
        acq_data_non_uniform.array = acq_data_non_uniform.array[indices_to_keep]
        acq_data_non_uniform.geometry.set_angles(data.geometry.angles[indices_to_keep])



        # With wedge_behaviour = forward/back -------------------------------------------------
        
        weights = _calculate_angular_sampling_weights(acq_data_non_uniform, 180,wedge_behaviour='forward/back')

        # expect weights to be in relative pattern:
        # [x, 2,3,3,2,3,3,2,3,3], etc
        # This is because the gaps are:
        # [1.2, 1.2, 2.4, 1.2, 1.2, 2.4, etc]
        # exclude end weights for now
        np.testing.assert_allclose(weights[1::3], weights[2]*2/3, atol=1e-4) # 1.2
        np.testing.assert_allclose(weights[2::3], weights[1]*3/2, atol=1e-4) # 1.8
        np.testing.assert_allclose(weights[3::3], weights[1]*3/2, atol=1e-4) # 1.8
        # start and end behaviour:
        # we have 150 projections and then we got rid of every 4th one so we 
        # lost 37 projections and have 113 left.
        # the first and last of these are in the same position as the original data so we expect 1.2
        np.testing.assert_allclose(weights[0], 1.2, atol=1e-4)
        np.testing.assert_allclose(weights[-1], 1.2, atol=1e-4)


        # with wedge_behaviour=max_gap -------------------------------------------------

        max_gap=1.5

        weights = _calculate_angular_sampling_weights(acq_data_non_uniform,180, max_gap=max_gap, wedge_behaviour='max_gap')

        # note in this data the standard gap between angles is 1.2
        # And the big gap is 1.8 
        # as 1.8 > max_gap we expect these to be set to the max_gap, 1.5
        np.testing.assert_allclose(weights[1::3], 1.2, atol=1e-4)
        np.testing.assert_allclose(weights[2::3], max_gap, atol=1e-4)
        np.testing.assert_allclose(weights[3::3], max_gap, atol=1e-4)
        np.testing.assert_allclose(weights[0], 1.2, atol=1e-4)
        np.testing.assert_allclose(weights[-1], 1.2, atol=1e-4)


        # TODO: add test with default value of max_gap

    def test_weights_with_non_uniform_angular_spread_and_missing_wedge_at_centre(self):
        # Every fourth angle is missing
        data = self.acq_data_180
        # create data with every fourth angle missing:
        total_angles = len(data.geometry.angles)
        indices_to_keep = [idx for idx in range(total_angles) if idx % 4 != 3]

        total_angles = len(indices_to_keep)
        quarter_point = total_angles // 4
        three_quarter_point = 3 * total_angles // 4
            
        # Keep first quarter and last quarter
        indices_to_keep_2 = list(range(quarter_point)) + list(range(three_quarter_point, total_angles))
        
        acq_data_non_uniform = data.copy()
        acq_data_non_uniform.array = acq_data_non_uniform.array[indices_to_keep][indices_to_keep_2]
        acq_data_non_uniform.geometry.set_angles(data.geometry.angles[indices_to_keep][indices_to_keep_2])

        weights = _calculate_angular_sampling_weights(acq_data_non_uniform, 180, wedge_behaviour='forward/back')
        # expect weights to be in relative pattern:
        # [x, 2,3,3,2,3,3,2,3,3], etc, except where gap is:
        # The gap is at the index 113/4-1 in the new data = 27.
        # so we expect index 27 and 28 to be set to forward/back
        print("the weights[20:30]:", weights[20:30])
        # The gap between index 26 and 27 is 2.8 (double the normal gap) so we expect 27 to be set to 2*1.2 = 2.4
        np.testing.assert_allclose(weights[27], 2.4, atol=1e-4)
        # The gap between index 27 and 28 is equivalent to the gap between index 0.75*113=84 and 85 in the data 
        # before the wedge was removed, which is 1.2 so we expect weight for index 28 to be set to 1.2
        np.testing.assert_allclose(weights[28], 1.2, atol=1e-4)


class Test_calculate_angular_sampling_weights_parallel(unittest.TestCase):
    def setUp(self):
        self.acq_data_360 = SIMULATED_PARALLEL_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()
        self.acq_data_360=np.log(self.acq_data_360)
        self.acq_data_360*=-1.0

        self.acq_data_180 = Slicer(roi={'angle': (0,150,1)})(self.acq_data_360)

        self.ig = self.img_data.geometry
        self.ag = self.acq_data_360.geometry

    def test_calls_calculate_angular_sampling_weights_with_360(self):
        with patch('cil.recon.weighted_fbp._calculate_angular_sampling_weights') as mock_calculate_weights:
            calculate_angular_sampling_weights_parallel(self.acq_data_360)
            mock_calculate_weights.assert_called_with(self.acq_data_360, 180, None, 'forward/back')

    
class Test_calculate_angular_sampling_weights_cone(unittest.TestCase):
    def setUp(self):
        self.acq_data_360 = SIMULATED_CONE_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()
        self.acq_data_360=np.log(self.acq_data_360)
        self.acq_data_360*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data_360.geometry    

    def test_calls_calculate_angular_sampling_weights_with_360(self):
        # use mock to check that calculate_angular_sampling_weights is called with angular_domain=360
        with patch('cil.recon.weighted_fbp._calculate_angular_sampling_weights') as mock_calculate_weights:
            calculate_angular_sampling_weights_cone(self.acq_data_360)
            mock_calculate_weights.assert_called_with(self.acq_data_360, 360, None, 'forward/back')

        with patch('cil.recon.weighted_fbp._calculate_angular_sampling_weights') as mock_calculate_weights:
            calculate_angular_sampling_weights_cone(self.acq_data_360, scan_type='full')
            mock_calculate_weights.assert_called_with(self.acq_data_360, 360, None, 'forward/back')

    def test_half_angle_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            calculate_angular_sampling_weights_cone(self.acq_data_360, scan_type='half')


class Test_normalise_weights_for_FBP(unittest.TestCase):
    def setUp(self):
        self.acq_data_360 = SIMULATED_CONE_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()
        self.acq_data_360=np.log(self.acq_data_360)
        self.acq_data_360*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data_360.geometry

    def normalise_equal_weights(self):
        weights = [5,5,5,5,5,5,5]
        normalised_weights = normalise_weights_for_FBP(weights)
        expected_normalised_value = 1
        np.testing.assert_allclose(normalised_weights, expected_normalised_value, atol=1e-4)
        np.testing.assert_almost_equal(sum(normalised_weights), 7)

    def normalise_different_weights(self):
        weights = [5,6,7,8,9,10,11]
        normalised_weights = normalise_weights_for_FBP(weights)
        expected_normalised_value = 1
        np.testing.assert_allclose(normalised_weights, expected_normalised_value, atol=1e-4)
        np.testing.assert_almost_equal(sum(normalised_weights), 7)


    def normalise_weights_with_duplicates(self):
        weights = [2,2,1,1, 2]
        expected_normalised_weights = [1, 1, 0.5, 0.5, 1]
        normalised_weights = normalise_weights_for_FBP(weights)
        np.testing.assert_allclose(normalised_weights, expected_normalised_weights, atol=1e-4)
        # inside FBP it multiplies by 1/len(weights) i.e. 1/(number of projections) we want the weights to
        # sum to the number of projections:
        expected_sum = 4
        np.testing.assert_almost_equal(sum(normalised_weights), expected_sum)


if __name__ == '__main__':
    unittest.main()