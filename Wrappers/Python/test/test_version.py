#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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
from utils import initialise_tests
initialise_tests()

class TestModuleBase(unittest.TestCase):
    def test_version(self):
        try:
            from cil import version
            a = version.version
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))
        try:
            import cil
            a = cil.__version__
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))


    def test_version_major(self):
        try:
            from cil import version
            a = version.major
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))


    def test_version_minor(self):
        try:
            from cil import version
            a = version.minor
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))


    def test_version_patch(self):
        try:
            from cil import version
            a = version.patch
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))


    def test_version_num_commit(self):
        try:
            from cil import version
            a = version.num_commit
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))


    def test_version_commit_hash(self):
        try:
            from cil import version
            a = version.commit_hash
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))
