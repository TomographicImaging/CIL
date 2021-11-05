import unittest

class TestModuleBase(unittest.TestCase):
    def test_version(self):
        try:
            from cil import version
            a = version.version
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))
