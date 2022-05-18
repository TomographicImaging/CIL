import unittest

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
        
        