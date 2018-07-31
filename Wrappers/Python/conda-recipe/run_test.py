import unittest
import numpy
from ccpi.framework import DataContainer, ImageData, AcquisitionData, \
  ImageGeometry, AcquisitionGeometry
import sys
from timeit import default_timer as timer
def dt (steps):
    return steps[-1] - steps[-2]

class TestDataContainer(unittest.TestCase):
    
    def test_creation_nocopy(self):
        shape = (2,3,4,5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range( size )])
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z' ,'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a),3)
        self.assertEqual(ds.dimension_labels , {0: 'X', 1: 'Y', 2: 'Z', 3: 'W'})
        
    def testGb_creation_nocopy(self):
        X,Y,Z = 1024,1024,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a),3)
        
        ds += 1
        steps.append(timer())
        t1 = dt(steps)
        print (dt(steps))
        ds = ds + 1
        steps.append(timer())
        t2 = dt(steps)
        print (dt(steps))
        self.assertAlmostEqual(t2,t1+t0,delta=t0*0.1)
        
    def test_creation_copy(self):
        shape = (2,3,4,5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range( size )])
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, True, ['X', 'Y','Z' ,'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a),2)
    
    def test_subset(self):
        shape = (4,3,2)
        a = [i for i in range(2*3*4)]
        a = numpy.asarray(a)
        a = numpy.reshape(a, shape)
        ds = DataContainer(a, True, ['X', 'Y','Z'])
        sub = ds.subset(['X'])
        res = True
        try:
            numpy.testing.assert_array_equal(sub.as_array(),
                                                 numpy.asarray([0,6,12,18]))
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
        
        sub = ds.subset(['X'], Y=2, Z=0)
        res = True
        try:
            numpy.testing.assert_array_equal(sub.as_array(),
                                                 numpy.asarray([4,10,16,22]))
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
        
        
        sub = ds.subset(['Y'])
        try:
            numpy.testing.assert_array_equal(
                        sub.as_array(), numpy.asarray([0,2,4]))
            res = True
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
            
        
        sub = ds.subset(['Z'])
        try:
            numpy.testing.assert_array_equal(
                        sub.as_array(), numpy.asarray([0,1]))
            res = True
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
        sub = ds.subset(['Z'], X=1, Y=2)
        try:
            numpy.testing.assert_array_equal(
                        sub.as_array(), numpy.asarray([10,11]))
            res = True
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
        
        print(a)
        sub = ds.subset(['X', 'Y'] , Z=1)
        res = True
        try:
            numpy.testing.assert_array_equal(sub.as_array(),
                                                 numpy.asarray([[ 1,  3,  5],
       [ 7,  9, 11],
       [13, 15, 17],
       [19, 21, 23]]))
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
        
    def test_ImageData(self):
        # create ImageData from geometry
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        vol = ImageData(geometry=vgeometry)
        self.assertEqual(vol.shape , (2,3,4))
        
        vol1 = vol + 1
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape))
        
        vol1 = vol - 1
        self.assertNumpyArrayEqual(vol1.as_array(), -numpy.ones(vol.shape))
        
        vol1 = 2 * (vol + 1)
        self.assertNumpyArrayEqual(vol1.as_array(), 2 * numpy.ones(vol.shape))
        
        vol1 = (vol + 1) / 2 
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape) / 2 )
        
        vol1 = vol + 1
        self.assertEqual(vol1.sum() , 2*3*4)
        vol1 = ( vol + 2 ) ** 2
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape) * 4 )
        
        
    
    def test_AcquisitionData(self):
        sgeometry = AcquisitionGeometry(dimension=2, angles=numpy.linspace(0, 180, num=10), 
                                           geom_type='parallel', pixel_num_v=3,
                                           pixel_num_h=5 , channels=2)
        sino = AcquisitionData(geometry=sgeometry)
        self.assertEqual(sino.shape , (2,10,3,5))
        
    
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
# =============================================================================
#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')
# 
#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())
# 
#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)
# =============================================================================

if __name__ == '__main__':
    unittest.main()