#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
from cil.framework import DataContainer, BlockDataContainer
import numpy as np
from utils import initialise_tests

initialise_tests()

def dt(steps):
    return steps[-1] - steps[-2]

class CCPiTestClass(unittest.TestCase):
    def assertBlockDataContainerEqual(self, container1, container2):
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if issubclass(container1.get_item(col).__class__, DataContainer):
                self.assertNumpyArrayEqual(
                    container1.get_item(col).as_array(),
                    container2.get_item(col).as_array()
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),container2.get_item(col))


    def assertBlockDataContainerAlmostEqual(self, container1, container2, decimal=7):
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if issubclass(container1.get_item(col).__class__, DataContainer):
                self.assertNumpyArrayAlmostEqual(
                    container1.get_item(col).as_array(),
                    container2.get_item(col).as_array(),
                    decimal=decimal
                    )
            else:
                self.assertBlockDataContainerAlmostEqual(container1.get_item(col),container2.get_item(col), decimal=decimal)


    def assertNumpyArrayEqual(self, first, second):
        np.testing.assert_array_equal(first, second)


    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        np.testing.assert_array_almost_equal(first, second, decimal)



    def assertDataArraysInContainerAllClose(self, container1, container2, rtol=1e-07, msg=None):
        self.assertTrue(issubclass(container1.__class__, container2.__class__), msg=msg)
        if isinstance(container1, BlockDataContainer):
            for col in range(container1.shape[0]):
                if issubclass(container1.get_item(col).__class__, DataContainer):
                    np.testing.assert_allclose(
                        container1.get_item(col).as_array(),
                        container2.get_item(col).as_array(),
                        rtol=rtol,
                        err_msg=msg
                        )
                else:
                    self.assertDataArraysInContainerAllClose(container1.get_item(col),container2.get_item(col), rtol=rtol,  msg=msg)
        else:
            np.testing.assert_allclose(container1.as_array(), container2.as_array(), rtol=rtol, err_msg=msg)

    def assertDataContainerAllClose(self, container1, container2, rtol=1e-07, msg=None, strict=False):
        '''
        Test to check if two DataContainers are close, by checking
        - they are the same class
        - they have arrays that are all close
        - if they have geometry, the geometries are equal
        - and if strict=True, their data type is the same
        
        Parameters
        ----------
        container1 : DataContainer
            The first DataContainer to compare.
        container2 : DataContainer
            The second DataContainer to compare.
        rtol : float, optional
            The relative tolerance for the array comparison
        msg : string, optional
            The error message to be printed in case of failure
        strict : bool, optional
            If True, raises an error if the data type in the DataContainers
            is not equal
        '''
        
        if not isinstance(container1, container2.__class__):
            raise TypeError("container2 is not the same class as container1")
        
        np.testing.assert_allclose(container1.array, container2.array, rtol=rtol, err_msg=msg)

        if hasattr(container1, "geometry"):
            np.testing.assert_equal(container1.geometry, container2.geometry, err_msg=msg)
        
        if strict:
            np.testing.assert_equal(container1.dtype, container2.dtype, err_msg=msg)