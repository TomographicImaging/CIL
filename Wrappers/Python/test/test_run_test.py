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
import numpy
import numpy as np
from cil.framework import ImageData, ImageGeometry

import numpy.testing

from testclass import CCPiTestClass

from utils import has_cvxpy, initialise_tests

initialise_tests()

if has_cvxpy:
    import cvxpy

def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]


def dt(steps):
    return steps[-1] - steps[-2]


class TestFunction(CCPiTestClass):


    def create_simple_ImageData(self):
        N = 64
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        Phantom = ImageData(geometry=ig)

        x = Phantom.as_array()

        x[int(round(N/4)):int(round(3*N/4)),
          int(round(N/4)):int(round(3*N/4))] = 0.5
        x[int(round(N/8)):int(round(7*N/8)),
          int(round(3*N/8)):int(round(5*N/8))] = 1

        return (ig, Phantom)

    def _test_Norm2(self):
        ig, Phantom = self.create_simple_ImageData()
        x = Phantom.as_array()

        norm2 = cvxpy.Norm2()
        v1 = norm2(x)
        v2 = norm2(Phantom)
        self.assertEqual(v1, v2)

        p1 = norm2.prox(Phantom, 1)
        print(p1)
        p2 = norm2.prox(x, 1)
        self.assertNumpyArrayEqual(p1.as_array(), p2)

        p3 = norm2.proximal(Phantom, 1)
        p4 = norm2.proximal(x, 1)
        self.assertNumpyArrayEqual(p3.as_array(), p2)
        self.assertNumpyArrayEqual(p3.as_array(), p4)

        out = Phantom.copy()
        p5 = norm2.proximal(Phantom, 1, out=out)
        self.assertEqual(id(p5), id(out))
        self.assertNumpyArrayEqual(p5.as_array(), p3.as_array())
# =============================================================================
#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')
#
#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())
#
#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)
# =============================================================================



if __name__ == '__main__':
    unittest.main()
