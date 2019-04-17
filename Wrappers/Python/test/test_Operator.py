import unittest
#from ccpi.optimisation.operators import Operator
#from ccpi.optimisation.ops import TomoIdentity
from ccpi.framework import ImageGeometry, ImageData, BlockDataContainer, DataContainer
from ccpi.optimisation.operators import BlockOperator, BlockScaledOperator,\
    FiniteDiff
import numpy
from timeit import default_timer as timer
from ccpi.framework import ImageGeometry
from ccpi.optimisation.operators import Gradient, Identity, SparseFiniteDiff
from ccpi.optimisation.operators import LinearOperator

def dt(steps):
    return steps[-1] - steps[-2]

class CCPiTestClass(unittest.TestCase):
    def assertBlockDataContainerEqual(self, container1, container2):
        print ("assert Block Data Container Equal")
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if issubclass(container1.get_item(col).__class__, DataContainer):
                print ("Checking col ", col)
                self.assertNumpyArrayEqual(
                    container1.get_item(col).as_array(), 
                    container2.get_item(col).as_array()
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),container2.get_item(col))
    
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
            print("expected " , second)
            print("actual " , first)

        self.assertTrue(res)


class TestOperator(CCPiTestClass):
    def test_ScaledOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        scalar = 0.5
        sid = scalar * Identity(ig)
        numpy.testing.assert_array_equal(scalar * img.as_array(), sid.direct(img).as_array())
        

    def test_TomoIdentity(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        self.assertTrue(img.shape == (30,20,10))
        self.assertEqual(img.sum(), 0)
        Id = Identity(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())

    def test_FiniteDifference(self):
        print ("test FiniteDifference")
        ##
        N, M = 2, 3

        ig = ImageGeometry(N, M)
        Id = Identity(ig)

        FD = FiniteDiff(ig, direction = 0, bnd_cond = 'Neumann')
        u = FD.domain_geometry().allocate('random_int')
        
        
        res = FD.domain_geometry().allocate(ImageGeometry.RANDOM_INT)
        FD.adjoint(u, out=res)
        w = FD.adjoint(u)

        self.assertNumpyArrayEqual(res.as_array(), w.as_array())
        
        res = Id.domain_geometry().allocate(ImageGeometry.RANDOM_INT)
        Id.adjoint(u, out=res)
        w = Id.adjoint(u)

        self.assertNumpyArrayEqual(res.as_array(), w.as_array())
        self.assertNumpyArrayEqual(u.as_array(), w.as_array())

        G = Gradient(ig)

        u = G.range_geometry().allocate(ImageGeometry.RANDOM_INT)
        res = G.domain_geometry().allocate()
        G.adjoint(u, out=res)
        w = G.adjoint(u)
        self.assertNumpyArrayEqual(res.as_array(), w.as_array())
        
        u = G.domain_geometry().allocate(ImageGeometry.RANDOM_INT)
        res = G.range_geometry().allocate()
        G.direct(u, out=res)
        w = G.direct(u)
        self.assertBlockDataContainerEqual(res, w)
        
    def test_PowerMethod(self):
        
        N, M = 200, 300
        niter = 1000
        ig = ImageGeometry(N, M)
        Id = Identity(ig)
        
        G = Gradient(ig)
        
        uid = Id.domain_geometry().allocate(ImageGeometry.RANDOM_INT, seed=1)
        
        a = LinearOperator.PowerMethod(Id, niter, uid)
        b = LinearOperator.PowerMethodNonsquare(Id, niter, uid)
        
        print ("Edo impl", a[0])
        print ("old impl", b[0])
        
        #self.assertAlmostEqual(a[0], b[0])
        self.assertNumpyArrayAlmostEqual(a[0],b[0],decimal=6)
        
        a = LinearOperator.PowerMethod(G, niter, uid)
        b = LinearOperator.PowerMethodNonsquare(G, niter, uid)
        
        print ("Edo impl", a[0])
        print ("old impl", b[0])
        self.assertNumpyArrayAlmostEqual(a[0],b[0],decimal=2)
        #self.assertAlmostEqual(a[0], b[0])
        





class TestBlockOperator(unittest.TestCase):
    def assertBlockDataContainerEqual(self, container1, container2):
        print ("assert Block Data Container Equal")
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if issubclass(container1.get_item(col).__class__, DataContainer):
                print ("Checking col ", col)
                self.assertNumpyArrayEqual(
                    container1.get_item(col).as_array(), 
                    container2.get_item(col).as_array()
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),container2.get_item(col))
    
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
            print("expected " , second)
            print("actual " , first)

        self.assertTrue(res)

    def test_BlockOperator(self):
        
        
        M, N  = 3, 4
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random_int')  
        
        G = Gradient(ig)
        Id = Identity(ig)
        
        B = BlockOperator(G, Id)
        # Nx1 case
        u = ig.allocate('random_int')
        z1 = B.direct(u)
        
        res = B.range_geometry().allocate()
        #res = z1.copy()
        B.direct(u, out=res)
        
        
        print (type(z1), type(res))
        print (z1.shape)
        print(z1[0][0].as_array())
        print(res[0][0].as_array())   
        self.assertBlockDataContainerEqual(z1, res)
        # for col in range(z1.shape[0]):
        #     a = z1.get_item(col)
        #     b = res.get_item(col)
        #     if isinstance(a, BlockDataContainer):
        #         for col2 in range(a.shape[0]):
        #             self.assertNumpyArrayEqual(
        #                 a.get_item(col2).as_array(), 
        #                 b.get_item(col2).as_array()
        #                 )        
        #     else:
        #         self.assertNumpyArrayEqual(
        #             a.as_array(), 
        #             b.as_array()
        #             )
        z1 = B.range_geometry().allocate(ImageGeometry.RANDOM_INT)

        res1 = B.adjoint(z1)
        res2 = B.domain_geometry().allocate()
        B.adjoint(z1, out=res2)
        
        self.assertNumpyArrayEqual(res1.as_array(), res2.as_array())

        BB = BlockOperator( Id, 2 * Id)
        B = BlockOperator( BB, Id )
        v = B.domain_geometry().allocate()
        B.adjoint(res,out=v)
        vv = B.adjoint(res)
        el1 = B.get_item(0,0).adjoint(z1.get_item(0)) +\
              B.get_item(1,0).adjoint(z1.get_item(1)) 
        print ("el1" , el1.as_array())
        print ("vv" , vv.as_array())
        print ("v" , v.as_array())
        
        self.assertNumpyArrayEqual(v.as_array(),vv.as_array())
        # test adjoint
        print ("############ 2x1 #############")

        BB = BlockOperator( Id, 2 * Id)
        u = ig.allocate(1)
        z1 = BB.direct(u)
        print ("z1 shape {} one\n{} two\n{}".format(z1.shape, 
            z1.get_item(0).as_array(),
            z1.get_item(1).as_array()))
        res = BB.range_geometry().allocate(0)
        BB.direct(u, out=res)
        print ("res shape {} one\n{} two\n{}".format(res.shape, 
            res.get_item(0).as_array(),
            res.get_item(1).as_array()))
        
        
        self.assertNumpyArrayEqual(z1.get_item(0).as_array(),
                                   u.as_array())
        self.assertNumpyArrayEqual(z1.get_item(1).as_array(),
                                   2 * u.as_array())
        self.assertNumpyArrayEqual(res.get_item(0).as_array(),
                                   u.as_array())
        self.assertNumpyArrayEqual(res.get_item(1).as_array(),
                                   2 * u.as_array())

        x1 = BB.adjoint(z1)
        print("adjoint x1\n",x1.as_array())

        res1 = BB.domain_geometry().allocate()
        BB.adjoint(z1, out=res1)
        print("res1\n",res1.as_array())
        self.assertNumpyArrayEqual(x1.as_array(),
                                   res1.as_array())
        
        self.assertNumpyArrayEqual(x1.as_array(),
                                   5 * u.as_array())
        self.assertNumpyArrayEqual(res1.as_array(),
                                   5 * u.as_array())
        #################################################
    
        print ("############ 2x2 #############")
        BB = BlockOperator( Id, 2 * Id, 3 * Id,  Id, shape=(2,2))
        B = BB
        u = ig.allocate(1)
        U = BlockDataContainer(u,u)
        z1 = B.direct(U)


        print ("z1 shape {} one\n{} two\n{}".format(z1.shape, 
            z1.get_item(0).as_array(),
            z1.get_item(1).as_array()))
        self.assertNumpyArrayEqual(z1.get_item(0).as_array(),
                                   3 * u.as_array())
        self.assertNumpyArrayEqual(z1.get_item(1).as_array(),
                                   4 * u.as_array())
        res = B.range_geometry().allocate()
        B.direct(U, out=res)
        self.assertNumpyArrayEqual(res.get_item(0).as_array(),
                                   3 * u.as_array())
        self.assertNumpyArrayEqual(res.get_item(1).as_array(),
                                   4 * u.as_array())
        

        x1 = B.adjoint(z1)
        # this should be [15 u, 10 u]
        el1 = B.get_item(0,0).adjoint(z1.get_item(0)) + B.get_item(1,0).adjoint(z1.get_item(1)) 
        el2 = B.get_item(0,1).adjoint(z1.get_item(0)) + B.get_item(1,1).adjoint(z1.get_item(1)) 

        shape = B.get_output_shape(z1.shape, adjoint=True)
        print ("shape ", shape)
        out = B.domain_geometry().allocate()
        
        for col in range(B.shape[1]):
            for row in range(B.shape[0]):
                if row == 0:
                    el = B.get_item(row,col).adjoint(z1.get_item(row))
                else:
                    el += B.get_item(row,col).adjoint(z1.get_item(row))
            out.get_item(col).fill(el)        

        print ("el1 " , el1.as_array())
        print ("el2 " , el2.as_array())
        print ("out shape {} one\n{} two\n{}".format(out.shape,
            out.get_item(0).as_array(), 
            out.get_item(1).as_array()))
        
        self.assertNumpyArrayEqual(out.get_item(0).as_array(),
                                   15 * u.as_array())
        self.assertNumpyArrayEqual(out.get_item(1).as_array(),
                                   10 * u.as_array())
        
        res2 = B.domain_geometry().allocate()  
        #print (res2, res2.as_array())  
        B.adjoint(z1, out = res2)
        
        #print ("adjoint",x1.as_array(),"\n",res2.as_array())
        self.assertNumpyArrayEqual(
            out.get_item(0).as_array(), 
            res2.get_item(0).as_array()
            )
        self.assertNumpyArrayEqual(
            out.get_item(1).as_array(), 
            res2.get_item(1).as_array()
            )
    
        if True:
            #B1 = BlockOperator(Id, Id, Id, Id, shape=(2,2))
            B1 = BlockOperator(G, Id)
            U = ig.allocate(ImageGeometry.RANDOM_INT)
            #U = BlockDataContainer(u,u)
            RES1 = B1.range_geometry().allocate()
            
            Z1 = B1.direct(U)
            B1.direct(U, out = RES1)
            
            self.assertBlockDataContainerEqual(Z1,RES1)
            
                        
            
            print("U", U.as_array())
            print("Z1", Z1[0][0].as_array())
            print("RES1", RES1[0][0].as_array())
            print("Z1", Z1[0][1].as_array())
            print("RES1", RES1[0][1].as_array())
    def test_timedifference(self):
        
        M, N ,W = 100, 512, 512
        ig = ImageGeometry(M, N, W)
        arr = ig.allocate('random_int')  
        
        G = Gradient(ig)
        Id = Identity(ig)
        
        B = BlockOperator(G, Id)
        
    
        # Nx1 case
        u = ig.allocate('random_int')
        steps = [timer()]
        i = 0
        n = 2.
        t1 = t2 = 0
        res = B.range_geometry().allocate()
            
        while (i < n):
            print ("i ", i)
            steps.append(timer())
            z1 = B.direct(u)
            steps.append(timer())
            t = dt(steps)
            #print ("B.direct(u) " ,t)
            t1 += t/n
            
            steps.append(timer())
            B.direct(u, out = res)
            steps.append(timer())
            t = dt(steps)
            #print ("B.direct(u, out=res) " ,t)
            t2 += t/n
            i += 1

        print ("Time difference ", t1,t2)
        self.assertGreater(t1,t2)

        steps = [timer()]
        i = 0
        #n = 50.
        t1 = t2 = 0
        resd = B.domain_geometry().allocate()
        z1 = B.direct(u)
        #B.adjoint(z1, out=resd)
        
        print (type(res))
        while (i < n):
            print ("i ", i)
            steps.append(timer())
            w1 = B.adjoint(z1)
            steps.append(timer())
            t = dt(steps)
            #print ("B.adjoint(z1) " ,t)
            t1 += t/n
            
            steps.append(timer())
            B.adjoint(z1, out=resd)
            steps.append(timer())
            t = dt(steps)
            #print ("B.adjoint(z1, out=res) " ,t)
            t2 += t/n
            i += 1

        print ("Time difference ", t1,t2)
        self.assertGreater(t1,t2)
    
    def test_BlockOperatorLinearValidity(self):
        
        
        M, N  = 3, 4
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random_int')  
        
        G = Gradient(ig)
        Id = Identity(ig)
        
        B = BlockOperator(G, Id)
        # Nx1 case
        u = ig.allocate('random_int')
        w = B.range_geometry().allocate(ImageGeometry.RANDOM_INT)
        w1 = B.direct(u)
        u1 = B.adjoint(w)
        self.assertEqual((w * w1).sum() , (u1*u).sum())

    


