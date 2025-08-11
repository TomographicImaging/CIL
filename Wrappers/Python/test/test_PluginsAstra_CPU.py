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
from utils import has_astra, initialise_tests

initialise_tests()

if has_astra:
    from cil.plugins.astra import ProjectionOperator
    import astra

def setup_parameters(self):

    self.backend = 'astra'
    self.ProjectionOperator = ProjectionOperator
    self.PO_args={'device':'cpu'}


class Test_basic_astra(unittest.TestCase):

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def test_astra_basic_(self):
        try:
            disable_print()
            astra.test_noCUDA()
            enable_prints()
        except:
            self.assertFalse('ASTRA CPU test failed')


class Test_Cone2D_Projectors_CPU_basic(unittest.TestCase, TestCommon_ProjectionOperator):

    @unittest.skipUnless(has_astra, "Requires ASTRA ")
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fp=0


class Test_Cone2D_Projectors_CPU_sim(unittest.TestCase, TestCommon_ProjectionOperator_SIM):

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fp = 0.2


class Test_Cone2D_Projectors_CPU_toy(unittest.TestCase, TestCommon_ProjectionOperator_TOY):

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_linearity = 2e-5
        self.tolerance_norm = 1e-3


class Test_Parallel2D_Projectors_CPU_basic(unittest.TestCase, TestCommon_ProjectionOperator):

    @unittest.skipUnless(has_astra, "Requires ASTRA ")
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fp=0

class Test_Parallel2D_Projectors_CPU_sim(unittest.TestCase, TestCommon_ProjectionOperator_SIM):

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fp = 0.16


class Test_Parallel2D_Projectors_CPU_toy(unittest.TestCase, TestCommon_ProjectionOperator_TOY):

    @unittest.skipUnless(has_astra, "Requires ASTRA")
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_linearity = 4e-6
        self.tolerance_norm = 1e-6
