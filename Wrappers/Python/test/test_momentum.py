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
import numpy as np
from cil.optimisation.algorithms.APGD import ConstantMomentum, NesterovMomentum, ScalarMomentumCoefficient

class MockAlgorithm:
    pass  # Placeholder for any required algorithm attributes

class TestMomentumCoefficients(unittest.TestCase):
    
    def test_constant_momentum(self):
        momentum_value = 0.9
        momentum = ConstantMomentum(momentum_value)
        algorithm = MockAlgorithm()
        
        self.assertEqual(momentum(algorithm), momentum_value, msg="ConstantMomentum should return the set momentum value")
    
    def test_nesterov_momentum(self):
        momentum = NesterovMomentum()
        algorithm = MockAlgorithm()
        
        initial_value = momentum(algorithm)
        second_value = momentum(algorithm)
        
        # Check initial momentum value
        expected_initial_t = 1
        expected_next_t = 0.5 * (1 + np.sqrt(1 + 4 * (expected_initial_t ** 2)))
        expected_initial_momentum = (expected_initial_t - 1) / expected_next_t
        
        self.assertAlmostEqual(initial_value, expected_initial_momentum, places=6, msg="Incorrect initial momentum value")
        
        # Check second iteration momentum value
        expected_next_t_old = expected_next_t
        expected_next_t = 0.5 * (1 + np.sqrt(1 + 4 * (expected_next_t_old ** 2)))
        expected_next_momentum = (expected_next_t_old - 1) / expected_next_t
        
        self.assertAlmostEqual(second_value, expected_next_momentum, places=6, msg="Incorrect second iteration momentum value")
        
    def test_momentum_coefficient_abc(self):
        class InvalidMomentum(ScalarMomentumCoefficient):
            pass
        
        with self.assertRaises(TypeError):
            InvalidMomentum()    