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
from utils_projectors import TestCommon_ProjectionOperator_SIM
from utils_projectors import TestCommon_ProjectionOperator_TOY, TestCommon_ProjectionOperator
from utils import disable_print, enable_prints
from utils import has_astra, has_nvidia, initialise_tests

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator
    import astra


def setup_parameters(self):

    self.backend = 'astra'
    self.ProjectionOperator = ProjectionOperator
    self.PO_args={'device':'gpu'}

@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA and GPU")
class Test_basic_astra(unittest.TestCase):
    def test_astra_basic_cuda(self):
        try:
            disable_print()
            astra.test_CUDA()
            enable_prints()
        except:
            self.assertFalse('ASTRA GPU test failed')

@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Cone3D_Projectors_GPU_basic(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_fp=0


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Cone3D_Projectors_GPU_sim(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_fp = 0.16


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Cone3D_Projectors_GPU_toy(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_linearity = 1e-3
        self.tolerance_norm = 0.1


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Cone2D_Projectors_GPU_basic(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fp=0


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Cone2D_Projectors_GPU_sim(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fp = 0.16


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Cone2D_Projectors_GPU_toy(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_linearity = 1e-3
        self.tolerance_norm = 0.1


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Parallel3D_Projectors_GPU_basic(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_fp=0


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Parallel3D_Projectors_GPU_sim(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_fp = 0.16


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Parallel3D_Projectors_GPU_toy(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_linearity = 1e-7
        self.tolerance_norm = 1e-6


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Parallel2D_Projectors_GPU_basic(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fp=0


@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Parallel2D_Projectors_GPU_sim(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fp = 0.16

@unittest.skipUnless(has_astra and has_nvidia, "Requires ASTRA GPU")
class Test_Parallel2D_Projectors_GPU_toy(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_linearity = 1e-6
        self.tolerance_norm = 1e-6
