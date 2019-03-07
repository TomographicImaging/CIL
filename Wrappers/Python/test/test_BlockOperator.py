import unittest
from ccpi.optimisation.operators import BlockOperator
from ccpi.framework import BlockDataContainer
from ccpi.optimisation.ops import TomoIdentity
from ccpi.framework import ImageGeometry, ImageData
import numpy

class TestBlockOperator(unittest.TestCase):

    def test_BlockOperator(self):
        print ("test_BlockOperator")
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) ]
        x = [ g.allocate() for g in ig ]
        ops = [ TomoIdentity(g) for g in ig ]

        K = BlockOperator(*ops)
        X = BlockDataContainer(x[0])
        Y = K.direct(X)
        self.assertTrue(Y.shape == K.shape)

        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),X.get_item(0).as_array())
        numpy.testing.assert_array_equal(Y.get_item(1).as_array(),X.get_item(0).as_array())
        #numpy.testing.assert_array_equal(Y.get_item(2).as_array(),X.get_item(2).as_array())
        
        X = BlockDataContainer(*x) + 1
        Y = K.T.direct(X)
        # K.T (1,3) X (3,1) => output shape (1,1)
        self.assertTrue(Y.shape == (1,1))
        zero = numpy.zeros(X.get_item(0).shape)
        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),len(x)+zero)
        

    def test_ScaledBlockOperatorSingleScalar(self):
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) ]
        x = [ g.allocate() for g in ig ]
        ops = [ TomoIdentity(g) for g in ig ]

        val = 1
        # test limit as non Scaled
        scalar = 1
        k = BlockOperator(*ops)
        K = scalar * k
        X = BlockDataContainer(*x) + val
        
        Y = K.T.direct(X)
        self.assertTrue(Y.shape == (1,1))
        zero = numpy.zeros(X.get_item(0).shape)
        xx = numpy.asarray([val for _ in x])
        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),((scalar*xx).sum()+zero))
        
        scalar = 0.5
        k = BlockOperator(*ops)
        K = scalar * k
        X = BlockDataContainer(*x) + 1
        
        Y = K.T.direct(X)
        self.assertTrue(Y.shape == (1,1))
        zero = numpy.zeros(X.get_item(0).shape)
        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),scalar*(len(x)+zero))
        
        
    def test_ScaledBlockOperatorScalarList(self):
        ig = [ ImageGeometry(2,3) , \
               #ImageGeometry(10,20,30) , \
               ImageGeometry(2,3    ) ]
        x = [ g.allocate() for g in ig ]
        ops = [ TomoIdentity(g) for g in ig ]


        # test limit as non Scaled
        scalar = numpy.asarray([1 for _ in x])
        k = BlockOperator(*ops)
        K = scalar * k
        val = 1
        X = BlockDataContainer(*x) + val
        
        Y = K.T.direct(X)
        self.assertTrue(Y.shape == (1,1))
        zero = numpy.zeros(X.get_item(0).shape)
        xx = numpy.asarray([val for _ in x])
        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),(scalar*xx).sum()+zero)
        
        scalar = numpy.asarray([i+1 for i,el in enumerate(x)])
        #scalar = numpy.asarray([6,0])
        k = BlockOperator(*ops)
        K = scalar * k
        X = BlockDataContainer(*x) + val
        Y = K.T.direct(X)
        self.assertTrue(Y.shape == (1,1))
        zero = numpy.zeros(X.get_item(0).shape)
        xx = numpy.asarray([val for _ in x])
        

        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),
          (scalar*xx).sum()+zero)
        

    def test_TomoIdentity(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        self.assertTrue(img.shape == (30,20,10))
        self.assertEqual(img.sum(), 0)
        Id = TomoIdentity(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())

