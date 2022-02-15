import unittest

class TestModuleBase(unittest.TestCase):
    def test_version(self):
        try:
            from cil import version
            a = version.version
            self.assertTrue(isinstance(a, str))
        except ImportError as ie:
            self.assertFalse(True, str(ie))
    
    def test_line_plot(self):
        try:
            from cil.utilities.display import line_plot
            self.assertTrue(True)
        except ImportError as ie:
            self.assertFalse(True, str(ie))
