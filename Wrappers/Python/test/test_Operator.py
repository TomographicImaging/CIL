import unittest
#from ccpi.optimisation.operators import Operator
from ccpi.optimisation.ops import TomoIdentity
from ccpi.framework import ImageGeometry, ImageData
import numpy

class TestOperator(unittest.TestCase):
    def test_ScaledOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        scalar = 0.5
        sid = scalar * TomoIdentity(ig)
        numpy.testing.assert_array_equal(scalar * img.as_array(), sid.direct(img).as_array())
        

    def test_TomoIdentity(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        self.assertTrue(img.shape == (30,20,10))
        self.assertEqual(img.sum(), 0)
        Id = TomoIdentity(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())

