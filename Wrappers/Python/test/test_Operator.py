import unittest
#from ccpi.optimisation.operators import Operator
from ccpi.optimisation.ops import TomoIdentity
from ccpi.framework import ImageGeometry, ImageData, BlockDataContainer, DataContainer
from ccpi.optimisation.operators import BlockOperator, BlockScaledOperator
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
        from ccpi.framework import ImageGeometry
        from ccpi.optimisation.operators import Gradient, Identity, SparseFiniteDiff

        
        M, N = 4, 3
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random_int')  
        
        G = Gradient(ig)
        Id = Identity(ig)
        
        B = BlockOperator(G, Id)
        
    #     Gx = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    #     Gy = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
        
    #     d1 = abs(Gx.matrix()).toarray().sum(axis=0)
    #     d2 = abs(Gy.matrix()).toarray().sum(axis=0)
    #     d3 = abs(Id.matrix()).toarray().sum(axis=0)
        
        
    #     d_res = numpy.reshape(d1 + d2 + d3, ig.shape, 'F')
        
    #     print(d_res)
    # #    
    #     z1 = abs(Gx.matrix()).toarray().sum(axis=1)
    #     z2 = abs(Gy.matrix()).toarray().sum(axis=1)
    #     z3 = abs(Id.matrix()).toarray().sum(axis=1)
    # #
    #     z_res = BlockDataContainer(BlockDataContainer(ImageData(numpy.reshape(z2, ig.shape, 'F')),\
    #                                                 ImageData(numpy.reshape(z1, ig.shape, 'F'))),\
    #                                                 ImageData(numpy.reshape(z3, ig.shape, 'F')))
    #
        # ttt = B.sum_abs_col()
    #    
        #TODO this is not working
    #    numpy.testing.assert_array_almost_equal(z_res[0][0].as_array(), ttt[0][0].as_array(), decimal=4)    
    #    numpy.testing.assert_array_almost_equal(z_res[0][1].as_array(), ttt[0][1].as_array(), decimal=4)    
    #    numpy.testing.assert_array_almost_equal(z_res[1].as_array(), ttt[1].as_array(), decimal=4)    

        # Nx1 case
        u = ig.allocate('random_int')
        
        z1 = B.direct(u)
        
        res = B.range_geometry().allocate()    
        B.direct(u, out = res)
        print (type(z1), type(res))
        print (z1.shape)
        print(z1[0][0].as_array())
        print(res[0][0].as_array())   

        for col in range(z1.shape[0]):
            a = z1.get_item(col)
            b = res.get_item(col)
            if isinstance(a, BlockDataContainer):
                for col2 in range(a.shape[0]):
                    self.assertNumpyArrayEqual(
                        a.get_item(col2).as_array(), 
                        b.get_item(col2).as_array()
                        )        
            else:
                self.assertNumpyArrayEqual(
                    a.as_array(), 
                    b.as_array()
                    )
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

