#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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
from unittest_parametrize import param, parametrize, ParametrizedTestCase

from cil.utilities import dtype_like
from utils import initialise_tests

initialise_tests()


class TestUtilities(ParametrizedTestCase, unittest.TestCase):

    @parametrize("input_value, reference_dtype",
        [param(3, np.float32, id="int_to_float32"),
         param(3.5, np.float32, id="float_to_float32"),
         param(3, np.float64, id="int_to_float64"),
         param(3.5, np.float64, id="float_to_float64"),
         param(3, np.complex64, id="int_to_complex64"),
         param(3.5, np.complex64, id="float_to_complex64"),
         param(1 + 2j, np.complex64, id="complex_to_complex64")])
    def test_dtype_like_scalar_input_casts(self, input_value, reference_dtype):
        reference_array = np.zeros(3, dtype=reference_dtype)
        result = dtype_like(input_value, reference_array)
        self.assertEqual(type(result), reference_dtype)
        self.assertEqual(result, input_value)

    def test_dtype_like_array_input_casts(self):
        a = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        reference_array = np.zeros(3, dtype=np.float32)
        result = dtype_like(a, reference_array)
        self.assertEqual(result.dtype, reference_array.dtype)
        np.testing.assert_array_equal(result, a)

    @parametrize("input_value, reference_array",
        [param(np.array([1, 2, 3]), [1, 2, 3], id="list_reference_has_no_dtype"),
         param(5, 10, id="int_reference_has_no_dtype"),
         param(2.5, "not-an-array", id="str_reference_has_no_dtype")])
    def test_dtype_like_without_dtype_reference(self, input_value, reference_array):
        result = dtype_like(input_value, reference_array)
        self.assertIs(result, input_value)

    def test_dtype_like_no_copy(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        reference_array = np.zeros(3, dtype=np.float32)
        result = dtype_like(a, reference_array)
        self.assertIs(result, a)

