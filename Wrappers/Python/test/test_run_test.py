# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest
import numpy
import numpy as np
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData, VectorData
from cil.framework import ImageGeometry,VectorGeometry
from cil.framework import AcquisitionGeometry
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.functions import L1Norm

from cil.optimisation.operators import MatrixOperator
from cil.optimisation.operators import LinearOperator

import numpy.testing

try:
    from cvxpy import *
    cvx_not_installable = False
except ImportError:
    cvx_not_installable = True


def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]


def dt(steps):
    return steps[-1] - steps[-2]




class TestAlgorithms(unittest.TestCase):
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    
class TestFunction(unittest.TestCase):
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

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
        print("test Norm2")
        opt = {'memopt': True}
        ig, Phantom = self.create_simple_ImageData()
        x = Phantom.as_array()
        print(Phantom)
        print(Phantom.as_array())

        norm2 = Norm2()
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
    
