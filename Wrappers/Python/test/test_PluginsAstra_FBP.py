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
from utils_projectors import TestCommon_FBP_SIM
from utils import has_astra, has_nvidia, initialise_tests

initialise_tests()

if has_astra:
    from cil.plugins.astra import FBP

def setup_parameters(self):

    self.backend = 'astra'
    self.FBP = FBP
    self.FBP_args={'device':'gpu'}


class Test_Cone3D_FBP_GPU(unittest.TestCase, TestCommon_FBP_SIM):

    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_fbp = 1e-3
        self.tolerance_fbp_roi = 1e-3


class Test_Cone2D_FBP_GPU(unittest.TestCase, TestCommon_FBP_SIM):

    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fbp = 1e-3
        self.tolerance_fbp_roi = 1e-3


class Test_Parallel3D_FBP_GPU(unittest.TestCase, TestCommon_FBP_SIM):

    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_fbp = 1e-3
        self.tolerance_fbp_roi = 1e-3


class Test_Parallel2D_FBP_GPU(unittest.TestCase, TestCommon_FBP_SIM):

    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fbp = 1e-3
        self.tolerance_fbp_roi = 1e-3


class Test_Parallel2D_FBP_CPU(unittest.TestCase, TestCommon_FBP_SIM):

    @unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA")
    def setUp(self):
        setup_parameters(self)
        self.FBP_args['device'] = 'cpu'
        self.Parallel2D()
        self.tolerance_fbp = 1e-3
        self.tolerance_fbp_roi = 1e-3
