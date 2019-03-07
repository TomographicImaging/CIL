import unittest
from ccpi.optimisation.operators import BlockOperator
from ccpi.framework import BlockDataContainer
from ccpi.optimisation.ops import TomoIdentity
from ccpi.framework import ImageGeometry, ImageData
import numpy

class TestBlockOperator(unittest.TestCase):

    def test_BlockOperator(self):
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(11,21,31) , \
               ImageGeometry(12,22,32) ]
        x = [ g.allocate() for g in ig ]
        ops = [ TomoIdentity(g) for g in ig ]

        K = BlockOperator(*ops)
        #X = BlockDataContainer(*x).T + 1
        X = BlockDataContainer(x[0])
        Y = K.direct(X)
        #self.assertTrue(Y.shape == X.shape)

        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),X.get_item(0).as_array())
        numpy.testing.assert_array_equal(Y.get_item(1).as_array(),X.get_item(0).as_array())
        #numpy.testing.assert_array_equal(Y.get_item(2).as_array(),X.get_item(2).as_array())


    def test_ScaledBlockOperatorSingleScalar(self):
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(11,21,31) , \
               ImageGeometry(12,22,32) ]
        x = [ g.allocate() for g in ig ]
        ops = [ TomoIdentity(g) for g in ig ]

        scalar = 0.5
        K = 0.5 * BlockOperator(*ops)
        X = BlockDataContainer(*x) + 1
        print (X.shape)
        X = BlockDataContainer(*x).T + 1
        print (X.shape, K.shape)
        Y = K.direct(X)
        self.assertTrue(Y.shape == X.shape)

        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),scalar * X.get_item(0).as_array())
        numpy.testing.assert_array_equal(Y.get_item(1).as_array(),scalar * X.get_item(1).as_array())
        numpy.testing.assert_array_equal(Y.get_item(2).as_array(),scalar * X.get_item(2).as_array())

    def test_ScaledBlockOperatorScalarList(self):
        ig = [ImageGeometry(10, 20, 30),
              ImageGeometry(11, 21, 31),
              ImageGeometry(12, 22, 32)]
        x = [g.allocate() for g in ig]
        ops = [TomoIdentity(g) for g in ig]

        scalar = [i*1.2 for i, el in enumerate(ig)]

        K = scalar * BlockOperator(*ops)
        X = BlockDataContainer(*x).T + 1
        Y = K.direct(X)
        self.assertTrue(Y.shape == X.shape)

        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),
                                         scalar[0] * X.get_item(0).as_array())
        numpy.testing.assert_array_equal(Y.get_item(1).as_array(),
                                         scalar[1] * X.get_item(1).as_array())
        numpy.testing.assert_array_equal(Y.get_item(2).as_array(),
                                         scalar[2] * X.get_item(2).as_array())


    def test_TomoIdentity(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        self.assertTrue(img.shape == (30,20,10))
        self.assertEqual(img.sum(), 0)
        Id = TomoIdentity(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())

