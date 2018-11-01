import unittest
import numpy
import numpy as np
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
import sys
from timeit import default_timer as timer
from ccpi.optimisation.algs import FISTA
from ccpi.optimisation.algs import FBPD
from ccpi.optimisation.funcs import Norm2sq
from ccpi.optimisation.funcs import ZeroFun
from ccpi.optimisation.funcs import Norm1
from ccpi.optimisation.funcs import TV2D

from ccpi.optimisation.ops import LinearOperatorMatrix
from ccpi.optimisation.ops import TomoIdentity

from cvxpy import *


def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]

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
        X,Y,Z = 512,512,512
        X,Y,Z = 256,512,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print("test clone")
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a),3)
        ds1 = ds.copy()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))
        ds1 = ds.clone()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))
        
        
    def testInlineAlgebra(self):
        print ("Test Inline Algebra")
        X,Y,Z = 1024,512,512
        X,Y,Z = 256,512,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        #ds.__iadd__( 2 )
        ds += 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],3.)
        #ds.__isub__( 2 ) 
        ds -= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],1.)
        #ds.__imul__( 2 )
        ds *= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],2.)
        #ds.__idiv__( 2 )
        ds /= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],1.)
        
        ds1 = ds.copy()
        #ds1.__iadd__( 1 )
        ds1 += 1
        #ds.__iadd__( ds1 )
        ds += ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],3.)
        #ds.__isub__( ds1 )
        ds -= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],1.)
        #ds.__imul__( ds1 )
        ds *= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],2.)
        #ds.__idiv__( ds1 )
        ds /= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],1.)
        
        
    def test_unary_operations(self):
        print ("Test unary operations")
        X,Y,Z = 1024,512,512
        X,Y,Z = 256,512,512
        steps = [timer()]
        a = -numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        
        ds.sign(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],-1.)
        
        ds.abs(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],1.)
        
        ds.__imul__( 2 )
        ds.sqrt(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],numpy.sqrt(2., dtype='float32'))
        
        
    
    def test_binary_operations(self):
        self.binary_add()
        self.binary_subtract()
        self.binary_multiply()
        self.binary_divide()
    
    def binary_add(self):
        print ("Test binary add")
        X,Y,Z = 512,512,512
        X,Y,Z = 256,512,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        ds1 = ds.copy()
        
        steps.append(timer())
        ds.add(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.add(ds1, out=ds)",dt(steps))
        steps.append(timer())
        ds2 = ds.add(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.add(ds1)",dt(steps))
        
        self.assertLess(t1,t2)
        self.assertEqual(ds.as_array()[0][0][0] , 2.)
        
        ds0 = ds
        ds0.add(2,out=ds0)
        steps.append(timer())
        print ("ds0.add(2,out=ds0)", dt(steps), 3 , ds0.as_array()[0][0][0])
        self.assertEqual(4., ds0.as_array()[0][0][0])
        
        dt1 = dt(steps)       
        ds3 = ds0.add(2)
        steps.append(timer())
        print ("ds3 = ds0.add(2)", dt(steps), 5 , ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1,dt2)
    
    def binary_subtract(self):
        print ("Test binary subtract")
        X,Y,Z = 512,512,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        ds1 = ds.copy()
        
        steps.append(timer())
        ds.subtract(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.subtract(ds1, out=ds)",dt(steps))
        self.assertEqual(0., ds.as_array()[0][0][0])
        
        steps.append(timer())
        ds2 = ds.subtract(ds1)
        self.assertEqual(-1., ds2.as_array()[0][0][0])
        
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.subtract(ds1)",dt(steps))
        
        self.assertLess(t1,t2)
        
        del ds1
        ds0 = ds.copy()
        steps.append(timer())
        ds0.subtract(2,out=ds0)
        #ds0.__isub__( 2 )
        steps.append(timer())
        print ("ds0.subtract(2,out=ds0)", dt(steps), -2. , ds0.as_array()[0][0][0])
        self.assertEqual(-2., ds0.as_array()[0][0][0])
        
        dt1 = dt(steps)       
        ds3 = ds0.subtract(2)
        steps.append(timer())
        print ("ds3 = ds0.subtract(2)", dt(steps), 0. , ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1,dt2)
        self.assertEqual(-2., ds0.as_array()[0][0][0])
        self.assertEqual(-4., ds3.as_array()[0][0][0])
       
    def binary_multiply(self):
        print ("Test binary multiply")
        X,Y,Z = 1024,512,512
        X,Y,Z = 256,512,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        ds1 = ds.copy()
        
        steps.append(timer())
        ds.multiply(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.multiply(ds1, out=ds)",dt(steps))
        steps.append(timer())
        ds2 = ds.multiply(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.multiply(ds1)",dt(steps))
        
        self.assertLess(t1,t2)
        
        ds0 = ds
        ds0.multiply(2,out=ds0)
        steps.append(timer())
        print ("ds0.multiply(2,out=ds0)", dt(steps), 2. , ds0.as_array()[0][0][0])
        self.assertEqual(2., ds0.as_array()[0][0][0])
        
        dt1 = dt(steps)       
        ds3 = ds0.multiply(2)
        steps.append(timer())
        print ("ds3 = ds0.multiply(2)", dt(steps), 4. , ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1,dt2)
        self.assertEqual(4., ds3.as_array()[0][0][0])
        self.assertEqual(2., ds.as_array()[0][0][0])
    
    def binary_divide(self):
        print ("Test binary divide")
        X,Y,Z = 1024,512,512
        X,Y,Z = 256,512,512
        steps = [timer()]
        a = numpy.ones((X,Y,Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y','Z'])
        ds1 = ds.copy()
        
        steps.append(timer())
        ds.divide(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.divide(ds1, out=ds)",dt(steps))
        steps.append(timer())
        ds2 = ds.divide(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.divide(ds1)",dt(steps))
        
        self.assertLess(t1,t2)
        self.assertEqual(ds.as_array()[0][0][0] , 1.)
        
        ds0 = ds
        ds0.divide(2,out=ds0)
        steps.append(timer())
        print ("ds0.divide(2,out=ds0)", dt(steps), 0.5 , ds0.as_array()[0][0][0])
        self.assertEqual(0.5, ds0.as_array()[0][0][0])
        
        dt1 = dt(steps)       
        ds3 = ds0.divide(2)
        steps.append(timer())
        print ("ds3 = ds0.divide(2)", dt(steps), 0.25 , ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1,dt2)
        self.assertEqual(.25, ds3.as_array()[0][0][0])
        self.assertEqual(.5, ds.as_array()[0][0][0])
        
    
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

    
    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)

class TestAlgorithms(unittest.TestCase):
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)

    
    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)
    def test_FISTA(self):
        # Problem data.
        m = 30
        n = 20
        np.random.seed(1)
        Amat = np.random.randn(m, n)
        A = LinearOperatorMatrix(Amat)
        bmat = np.random.randn(m)
        bmat.shape = (bmat.shape[0],1)
        
        # A = Identity()
        # Change n to equal to m.
        
        b = DataContainer(bmat)
        
        # Regularization parameter
        lam = 10
        opt = {'memopt':True}
        # Create object instances with the test data A and b.
        f = Norm2sq(A,b,c=0.5, memopt=True)
        g0 = ZeroFun()
        
        # Initial guess
        x_init = DataContainer(np.zeros((n,1)))
        
        f.grad(x_init)
        
        # Run FISTA for least squares plus zero function.
        x_fista0, it0, timing0, criter0 = FISTA(x_init, f, g0 , opt=opt)
        
        # Print solution and final objective/criterion value for comparison
        print("FISTA least squares plus zero function solution and objective value:")
        print(x_fista0.array)
        print(criter0[-1])
        
        # Compare to CVXPY
          
        # Construct the problem.
        x0 = Variable(n)
        objective0 = Minimize(0.5*sum_squares(Amat*x0 - bmat.T[0]) )
        prob0 = Problem(objective0)
          
        # The optimal objective is returned by prob.solve().
        result0 = prob0.solve(verbose=False,solver=SCS,eps=1e-9)
            
        # The optimal solution for x is stored in x.value and optimal objective value 
        # is in result as well as in objective.value
        print("CVXPY least squares plus zero function solution and objective value:")
        print(x0.value)
        print(objective0.value)
        self.assertNumpyArrayAlmostEqual(
                 numpy.squeeze(x_fista0.array),x0.value,6)
    def test_FISTA_Norm1(self):

        opt = {'memopt':True}
        # Problem data.
        m = 30
        n = 20
        np.random.seed(1)
        Amat = np.random.randn(m, n)
        A = LinearOperatorMatrix(Amat)
        bmat = np.random.randn(m)
        bmat.shape = (bmat.shape[0],1)
        
        # A = Identity()
        # Change n to equal to m.
        
        b = DataContainer(bmat)
        
        # Regularization parameter
        lam = 10
        opt = {'memopt':True}
        # Create object instances with the test data A and b.
        f = Norm2sq(A,b,c=0.5, memopt=True)
        g0 = ZeroFun()
        
        # Initial guess
        x_init = DataContainer(np.zeros((n,1)))
        
        # Create 1-norm object instance
        g1 = Norm1(lam)
        
        g1(x_init)
        g1.prox(x_init,0.02)
        
        # Combine with least squares and solve using generic FISTA implementation
        x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1,opt=opt)
        
        # Print for comparison
        print("FISTA least squares plus 1-norm solution and objective value:")
        print(x_fista1.as_array().squeeze())
        print(criter1[-1])
        
        # Compare to CVXPY
            
        # Construct the problem.
        x1 = Variable(n)
        objective1 = Minimize(0.5*sum_squares(Amat*x1 - bmat.T[0]) + lam*norm(x1,1) )
        prob1 = Problem(objective1)
            
        # The optimal objective is returned by prob.solve().
        result1 = prob1.solve(verbose=False,solver=SCS,eps=1e-9)
            
        # The optimal solution for x is stored in x.value and optimal objective value 
        # is in result as well as in objective.value
        print("CVXPY least squares plus 1-norm solution and objective value:")
        print(x1.value)
        print(objective1.value)
            
        self.assertNumpyArrayAlmostEqual(
                 numpy.squeeze(x_fista1.array),x1.value,6)

    def test_FBPD_Norm1(self):

        opt = {'memopt':True}
        # Problem data.
        m = 30
        n = 20
        np.random.seed(1)
        Amat = np.random.randn(m, n)
        A = LinearOperatorMatrix(Amat)
        bmat = np.random.randn(m)
        bmat.shape = (bmat.shape[0],1)
        
        # A = Identity()
        # Change n to equal to m.
        
        b = DataContainer(bmat)
        
        # Regularization parameter
        lam = 10
        opt = {'memopt':True}
        # Create object instances with the test data A and b.
        f = Norm2sq(A,b,c=0.5, memopt=True)
        g0 = ZeroFun()
        
        # Initial guess
        x_init = DataContainer(np.zeros((n,1)))
        
        # Create 1-norm object instance
        g1 = Norm1(lam)
        
        
        # Compare to CVXPY
            
        # Construct the problem.
        x1 = Variable(n)
        objective1 = Minimize(0.5*sum_squares(Amat*x1 - bmat.T[0]) + lam*norm(x1,1) )
        prob1 = Problem(objective1)
            
        # The optimal objective is returned by prob.solve().
        result1 = prob1.solve(verbose=False,solver=SCS,eps=1e-9)
            
        # The optimal solution for x is stored in x.value and optimal objective value 
        # is in result as well as in objective.value
        print("CVXPY least squares plus 1-norm solution and objective value:")
        print(x1.value)
        print(objective1.value)
            
        # Now try another algorithm FBPD for same problem:
        x_fbpd1, itfbpd1, timingfbpd1, criterfbpd1 = FBPD(x_init, None, f, g1)
        print(x_fbpd1)
        print(criterfbpd1[-1])
        
        self.assertNumpyArrayAlmostEqual(
                 numpy.squeeze(x_fbpd1.array),x1.value,6)
        # Plot criterion curve to see both FISTA and FBPD converge to same value.
        # Note that FISTA is very efficient for 1-norm minimization so it beats
        # FBPD in this test by a lot. But FBPD can handle a larger class of problems 
        # than FISTA can.
        
        
        # Now try 1-norm and TV denoising with FBPD, first 1-norm.
        
        # Set up phantom size NxN by creating ImageGeometry, initialising the 
        # ImageData object with this geometry and empty array and finally put some
        # data into its array, and display as image.
    def test_FISTA_denoise(self):
        opt = {'memopt':True}
        N = 64
        ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
        Phantom = ImageData(geometry=ig)
        
        x = Phantom.as_array()
        x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
        x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
        
        
        # Identity operator for denoising
        I = TomoIdentity(ig)
        
        # Data and add noise
        y = I.direct(Phantom)
        y.array = y.array + 0.1*np.random.randn(N, N)
        
        
        # Data fidelity term
        f_denoise = Norm2sq(I,y,c=0.5,memopt=True)
        
        # 1-norm regulariser
        lam1_denoise = 1.0
        g1_denoise = Norm1(lam1_denoise)
        
        # Initial guess
        x_init_denoise = ImageData(np.zeros((N,N)))
        
        # Combine with least squares and solve using generic FISTA implementation
        x_fista1_denoise, it1_denoise, timing1_denoise, criter1_denoise = FISTA(x_init_denoise, f_denoise, g1_denoise, opt=opt)
        
        print(x_fista1_denoise)
        print(criter1_denoise[-1])
        
        
        # Now denoise LS + 1-norm with FBPD
        x_fbpd1_denoise, itfbpd1_denoise, timingfbpd1_denoise, criterfbpd1_denoise = FBPD(x_init_denoise, None, f_denoise, g1_denoise)
        print(x_fbpd1_denoise)
        print(criterfbpd1_denoise[-1])
        
        
        # Compare to CVXPY
            
        # Construct the problem.
        x1_denoise = Variable(N**2,1)
        objective1_denoise = Minimize(0.5*sum_squares(x1_denoise - y.array.flatten()) + lam1_denoise*norm(x1_denoise,1) )
        prob1_denoise = Problem(objective1_denoise)
            
        # The optimal objective is returned by prob.solve().
        result1_denoise = prob1_denoise.solve(verbose=False,solver=SCS,eps=1e-12)
            
        # The optimal solution for x is stored in x.value and optimal objective value 
        # is in result as well as in objective.value
        print("CVXPY least squares plus 1-norm solution and objective value:")
        print(x1_denoise.value)
        print(objective1_denoise.value)
        self.assertNumpyArrayAlmostEqual(
                 x_fista1_denoise.array.flatten(),x1_denoise.value,5)
        
        self.assertNumpyArrayAlmostEqual(
                 x_fbpd1_denoise.array.flatten(),x1_denoise.value,5)
        x1_cvx = x1_denoise.value
        x1_cvx.shape = (N,N)
        
        
        # Now TV with FBPD
        lam_tv = 0.1
        gtv = TV2D(lam_tv)
        gtv(gtv.op.direct(x_init_denoise))
        
        opt_tv = {'tol': 1e-4, 'iter': 10000}
        
        x_fbpdtv_denoise, itfbpdtv_denoise, timingfbpdtv_denoise, criterfbpdtv_denoise = FBPD(x_init_denoise, None, f_denoise, gtv,opt=opt_tv)
        print(x_fbpdtv_denoise)
        print(criterfbpdtv_denoise[-1])
        
        
        
        # Compare to CVXPY
        
        # Construct the problem.
        xtv_denoise = Variable((N,N))
        objectivetv_denoise = Minimize(0.5*sum_squares(xtv_denoise - y.array) + lam_tv*tv(xtv_denoise) )
        probtv_denoise = Problem(objectivetv_denoise)
            
        # The optimal objective is returned by prob.solve().
        resulttv_denoise = probtv_denoise.solve(verbose=False,solver=SCS,eps=1e-12)
            
        # The optimal solution for x is stored in x.value and optimal objective value 
        # is in result as well as in objective.value
        print("CVXPY least squares plus 1-norm solution and objective value:")
        print(xtv_denoise.value)
        print(objectivetv_denoise.value)
            
        self.assertNumpyArrayAlmostEqual(
                 x_fbpdtv_denoise.as_array(),xtv_denoise.value,1)
        
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
