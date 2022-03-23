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


from cil.utilities.projector_tests import TestCommon_ProjectionOperator_SIM, TestCommon_FBP_SIM
from cil.utilities.projector_tests import TestCommon_ProjectionOperator_TOY, TestCommon_ProjectionOperator
import unittest
from utils import has_gpu_astra, has_astra

if has_astra:
    from cil.plugins.astra import ProjectionOperator
    from cil.plugins.astra import FBP

has_astra_gpu = has_gpu_astra()
if not has_astra_gpu:
    print("Unable to run ASTRA GPU tests")


def setup_parameters(self):

    self.backend = 'astra'
    self.ProjectionOperator = ProjectionOperator
    self.PO_args={'device':'cpu'}



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
        self.tolerance_linearity = 1e-7
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
        self.tolerance_linearity = 1e-10
        self.tolerance_norm = 1e-6
