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
        from cil import version
        self.assertTrue(isinstance(version.version, str))
        import cil
        self.assertEqual(cil.__version__, version.version)

    def test_version_major(self):
        from cil import version
        self.assertTrue(isinstance(version.major, int))

    def test_version_minor(self):
        from cil import version
        self.assertTrue(isinstance(version.minor, int))

    def test_version_patch(self):
        from cil import version
        self.assertTrue(isinstance(version.patch, int))

    def test_version_num_commit(self):
        from cil import version
        self.assertTrue(isinstance(version.num_commit, int))

    def test_version_commit_hash(self):
        from cil import version
        self.assertTrue(isinstance(version.commit_hash, str))
        try: # tagged release
            self.assertEqual((version.commit_hash, version.num_commit), ('None', 0))
        except AssertionError: # dev build
            self.assertEqual(version.commit_hash[0], 'g')
