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
from utils import has_tigre, has_nvidia, initialise_tests

if has_tigre:
    from cil.plugins.tigre import ProjectionOperator

initialise_tests()

def setup_parameters(self):
    self.backend = 'tigre'
    self.ProjectionOperator = ProjectionOperator
    self.PO_args={'direct_method':'Siddon','adjoint_weights':'matched'}


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Cone3D_Projectors_basic_tigre(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_fp=0.16


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Cone3D_Projectors_sim_tigre(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_fp = 0.35


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Cone3D_Projectors_toy_tigre(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    def setUp(self):
        setup_parameters(self)
        self.Cone3D()
        self.tolerance_linearity = 1e-2
        self.tolerance_norm = 0.1


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Cone2D_Projectors_basic_tigre(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fp=0.16


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Cone2D_Projectors_sim_tigre(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_fp = 0.2


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Cone2D_Projectors_toy_tigre(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    def setUp(self):
        setup_parameters(self)
        self.Cone2D()
        self.tolerance_linearity = 1e-2
        self.tolerance_norm = 0.1


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Parallel3D_Projectors_basic_tigre(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_fp=1e-5


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Parallel3D_Projectors_sim_tigre(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_fp = 0.25


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Parallel3D_Projectors_toy_tigre(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    @unittest.skip("TIGRE backprojector weights bug")
    def setUp(self):
        setup_parameters(self)
        self.Parallel3D()
        self.tolerance_linearity = 1e-6
        self.tolerance_norm = 1e-5


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Parallel2D_Projectors_basic_tigre(unittest.TestCase, TestCommon_ProjectionOperator):
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fp=1e-5


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Parallel2D_Projectors_sim_tigre(unittest.TestCase, TestCommon_ProjectionOperator_SIM):
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_fp = 0.15


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class Test_Parallel2D_Projectors_toy_tigre(unittest.TestCase, TestCommon_ProjectionOperator_TOY):
    @unittest.skip("TIGRE backprojector weights bug")
    def setUp(self):
        setup_parameters(self)
        self.Parallel2D()
        self.tolerance_linearity = 1e-6
        self.tolerance_norm = 1e-5
