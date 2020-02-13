# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import unittest
from ccpi.framework import ImageGeometry, VectorGeometry, ImageData, BlockDataContainer, DataContainer
from ccpi.optimisation.operators import BlockOperator, BlockScaledOperator,\
    FiniteDiff, SymmetrizedGradient
import numpy
from timeit import default_timer as timer
from ccpi.optimisation.operators import Gradient, Identity, SparseFiniteDiff
from ccpi.optimisation.operators import LinearOperator, LinearOperatorMatrix
import numpy   
from ccpi.optimisation.operators import SumOperator, Gradient,\
            ZeroOperator, SymmetrizedGradient, CompositionOperator

from ccpi.framework import TestData
import os

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
    
    def test_LinearOperatorMatrix(self):
        
        print('Check LinearOperatorMatrix')
                
        m = 30
        n = 20
        
        vg = VectorGeometry(n)
        
        Amat = numpy.random.randn(m, n)
        A = LinearOperatorMatrix(Amat)
        
        b = vg.allocate('random')
        
        out1 = A.range_geometry().allocate()
        out2 = A.domain_geometry().allocate()
        
        res1 = A.direct(b)
        res2 = numpy.dot(A.A, b.as_array())
        self.assertNumpyArrayAlmostEqual(res1.as_array(), res2)
            
        A.direct(b, out = out1)
        self.assertNumpyArrayAlmostEqual(res1.as_array(), out1.as_array(), decimal=4)
        
        res3 = A.adjoint(res1)
        res4 = numpy.dot(A.A.transpose(),res1.as_array())
        self.assertNumpyArrayAlmostEqual(res3.as_array(), res4, decimal=4)   
        
        A.adjoint(res1, out = out2)
        self.assertNumpyArrayAlmostEqual(res3.as_array(), out2.as_array(), decimal=4)        
    
    
    def test_ScaledOperator(self):
        print ("test_ScaledOperator")
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        scalar = 0.5
        sid = scalar * Identity(ig)
        numpy.testing.assert_array_equal(scalar * img.as_array(), sid.direct(img).as_array())
        

    def test_Identity(self):
        print ("test_Identity")
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        # img.fill(numpy.ones((30,20,10)))
        self.assertTrue(img.shape == (30,20,10))
        #self.assertEqual(img.sum(), 2*float(10*20*30))
        self.assertEqual(img.sum(), 0.)
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
        print ("test_BlockOperator")
        
        N, M = 200, 300
        niter = 10
        ig = ImageGeometry(N, M)
        Id = Identity(ig)
        
        G = Gradient(ig)
        
        uid = Id.domain_geometry().allocate(ImageGeometry.RANDOM_INT, seed=1)
        
        a = LinearOperator.PowerMethod(Id, niter, uid)
        #b = LinearOperator.PowerMethodNonsquare(Id, niter, uid)
        b = LinearOperator.PowerMethod(Id, niter)
        print ("Edo impl", a[0])
        print ("None impl", b[0])
        
        #self.assertAlmostEqual(a[0], b[0])
        self.assertNumpyArrayAlmostEqual(a[0],b[0],decimal=6)
        
        a = LinearOperator.PowerMethod(G, niter, uid)
        b = LinearOperator.PowerMethod(G, niter)
        #b = LinearOperator.PowerMethodNonsquare(G, niter, uid)
        
        print ("Edo impl", a[0])
        #print ("old impl", b[0])
        self.assertNumpyArrayAlmostEqual(a[0],b[0],decimal=2)
        #self.assertAlmostEqual(a[0], b[0])
        
    def test_Norm(self):
        print ("test_BlockOperator")
        ##
        numpy.random.seed(1)
        N, M = 200, 300

        ig = ImageGeometry(N, M)
        G = Gradient(ig)
        t0 = timer()
        norm = G.norm()
        t1 = timer()
        norm2 = G.norm()
        t2 = timer()
        norm3 = G.norm(force=True)
        t3 = timer()
        print ("Norm dT1 {} dT2 {} dT3 {}".format(t1-t0,t2-t1, t3-t2))
        self.assertLess(t2-t1, t1-t0)
        self.assertLess(t2-t1, t3-t2)

        numpy.random.seed(1)
        t4 = timer()
        norm4 = G.norm(iterations=50, force=True)
        t5 = timer()
        self.assertLess(t2-t1, t5-t4)

        numpy.random.seed(1)
        t4 = timer()
        norm5 = G.norm(x_init=ig.allocate('random'), iterations=50, force=True)
        t5 = timer()
        self.assertLess(t2-t1, t5-t4)
        for n in [norm, norm2, norm3, norm4, norm5]:
            print ("norm {}", format(n))

    

        


class TestGradients(CCPiTestClass): 
    def setUp(self):
        N, M = 20, 30
        K = 20
        C = 20
        self.decimal = 1
        self.iterations = 50
        ###########################################################################
        # 2D geometry no channels
        self.ig = ImageGeometry(N, M)
        self.ig2 = ImageGeometry(N, M, channels = C)
        self.ig3 = ImageGeometry(N, M, K)

    def test_SymmetrizedGradient1a(self):
        ###########################################################################  
        ## Symmetrized Gradient Tests
        print ("Test SymmetrizedGradient")
        ###########################################################################
        # 2D geometry no channels
        # ig = ImageGeometry(N, M)
        Grad = Gradient(self.ig)
        
        E1 = SymmetrizedGradient(Grad.range_geometry())
        numpy.testing.assert_almost_equal(E1.norm(iterations=self.iterations), numpy.sqrt(8), decimal = self.decimal)
        
    def test_SymmetrizedGradient1b(self):
        ###########################################################################  
        ## Symmetrized Gradient Tests
        print ("Test SymmetrizedGradient")
        ###########################################################################
        # 2D geometry no channels
        # ig = ImageGeometry(N, M)
        Grad = Gradient(self.ig)
        
        E1 = SymmetrizedGradient(Grad.range_geometry())
        u1 = E1.domain_geometry().allocate('random_int')
        w1 = E1.range_geometry().allocate('random_int', symmetry = True)
        
        
        lhs = E1.direct(u1).dot(w1)
        rhs = u1.dot(E1.adjoint(w1))
        # self.assertAlmostEqual(lhs, rhs)
        numpy.testing.assert_almost_equal(lhs, rhs)
            
    def test_SymmetrizedGradient2(self):        
        ###########################################################################
        # 2D geometry with channels
        # ig2 = ImageGeometry(N, M, channels = C)
        Grad2 = Gradient(self.ig2, correlation = 'Space')
        
        E2 = SymmetrizedGradient(Grad2.range_geometry())
        
        u2 = E2.domain_geometry().allocate('random_int')
        w2 = E2.range_geometry().allocate('random_int', symmetry = True)
    #    
        lhs2 = E2.direct(u2).dot(w2)
        rhs2 = u2.dot(E2.adjoint(w2))
            
        numpy.testing.assert_almost_equal(lhs2, rhs2)
        
    def test_SymmetrizedGradient2a(self):        
        ###########################################################################
        # 2D geometry with channels
        # ig2 = ImageGeometry(N, M, channels = C)
        Grad2 = Gradient(self.ig2, correlation = 'Space')
        
        E2 = SymmetrizedGradient(Grad2.range_geometry())
        numpy.testing.assert_almost_equal(E2.norm(iterations=self.iterations), 
           numpy.sqrt(8), decimal = self.decimal)
        
    
    def test_SymmetrizedGradient3a(self):
        ###########################################################################
        # 3D geometry no channels
        #ig3 = ImageGeometry(N, M, K)
        Grad3 = Gradient(self.ig3, correlation = 'Space')
        
        E3 = SymmetrizedGradient(Grad3.range_geometry())

        norm1 = E3.norm()
        norm2 = E3.calculate_norm(iterations=100)
        print (norm1,norm2)
        numpy.testing.assert_almost_equal(norm2, numpy.sqrt(12), decimal = self.decimal)
        
    def test_SymmetrizedGradient3b(self):
        ###########################################################################
        # 3D geometry no channels
        #ig3 = ImageGeometry(N, M, K)
        Grad3 = Gradient(self.ig3, correlation = 'Space')
        
        E3 = SymmetrizedGradient(Grad3.range_geometry())
        
        u3 = E3.domain_geometry().allocate('random_int')
        w3 = E3.range_geometry().allocate('random_int', symmetry = True)
    #    
        lhs3 = E3.direct(u3).dot(w3)
        rhs3 = u3.dot(E3.adjoint(w3))
        numpy.testing.assert_almost_equal(lhs3, rhs3)  
        self.assertAlmostEqual(lhs3, rhs3)
        print (lhs3, rhs3, abs((rhs3-lhs3)/rhs3) , 1.5 * 10**(-4), abs((rhs3-lhs3)/rhs3) < 1.5 * 10**(-4))
        self.assertTrue( LinearOperator.dot_test(E3, range_init = w3, domain_init=u3) )
    def test_dot_test(self):
        Grad3 = Gradient(self.ig3, correlation = 'Space', backend='numpy')
             
        # self.assertAlmostEqual(lhs3, rhs3)
        self.assertTrue( LinearOperator.dot_test(Grad3 , verbose=True))
        self.assertTrue( LinearOperator.dot_test(Grad3 , decimal=6, verbose=True))

    def test_dot_test2(self):
        Grad3 = Gradient(self.ig3, correlation = 'SpaceChannel', backend='c')
             
        # self.assertAlmostEqual(lhs3, rhs3)
        self.assertTrue( LinearOperator.dot_test(Grad3 , verbose=True))
        self.assertTrue( LinearOperator.dot_test(Grad3 , decimal=6, verbose=True))




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
        print ("test_BlockOperator")
        
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

        print ("test_timedifference")
        M, N ,W = 100, 512, 512
        ig = ImageGeometry(M, N, W)
        arr = ig.allocate('random_int')  
        
        G = Gradient(ig, backend='numpy')
        Id = Identity(ig)
        
        B = BlockOperator(G, Id)
        
    
        # Nx1 case
        u = ig.allocate('random_int')
        steps = [timer()]
        i = 0
        n = 10.
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
        # self.assertGreater(t1,t2)
    
    def test_BlockOperatorLinearValidity(self):
        print ("test_BlockOperatorLinearValidity")
        
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

class TestOperatorCompositionSum(unittest.TestCase):
    def setUp(self):
        
        self.data = TestData().load(TestData.BOAT, size=(128,128))
        self.ig = self.data.geometry

    def test_SumOperator(self):

        # numpy.random.seed(1)
        ig = self.ig
        data = self.data

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        # z = ZeroOperator(domain_geometry=ig)
        # sym = SymmetrizedGradient(domain_geometry=ig)

        c = SumOperator(Id1,Id2)
        out = c.direct(data)

        numpy.testing.assert_array_almost_equal(out.as_array(),3 * data.as_array())


    def test_CompositionOperator_direct1(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        
        d = CompositionOperator(G, Id2)

        out1 = G.direct(data)
        out2 = d.direct(data)


        numpy.testing.assert_array_almost_equal(out2.get_item(0).as_array(),  out1.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(out2.get_item(1).as_array(),  out1.get_item(1).as_array())

    def test_CompositionOperator_direct2(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        
        d = CompositionOperator(G, Id2)

        out1 = G.direct(data)
        
        d_out = d.direct(data)

        d1 = Id2.direct(data)
        d2 = G.direct(d1)

        numpy.testing.assert_array_almost_equal(d2.get_item(0).as_array(),
                                                d_out.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(d2.get_item(1).as_array(),
                                                d_out.get_item(1).as_array())

    def test_CompositionOperator_direct3(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        
        d = CompositionOperator(G, Id2)

        out1 = G.direct(data)
        
        d_out = d.direct(data)

        d1 = Id2.direct(data)
        d2 = G.direct(d1)

        numpy.testing.assert_array_almost_equal(d2.get_item(0).as_array(),
                                                d_out.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(d2.get_item(1).as_array(),
                                                d_out.get_item(1).as_array())

        G2Id = G.compose(2*Id2)
        d2g = G2Id.direct(data)

        numpy.testing.assert_array_almost_equal(d2g.get_item(0).as_array(),
                                                2 * d_out.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(d2g.get_item(1).as_array(),
                                                2 * d_out.get_item(1).as_array())



    def test_CompositionOperator_adjoint1(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        
        d = CompositionOperator(G, Id2)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  out1.as_array())
        
    def test_CompositionOperator_adjoint2(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        
        d = CompositionOperator(G, Id1)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  2 * out1.as_array())
    
    def test_CompositionOperator_adjoint3(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        Id2 = Identity(ig)
        
        d = G.compose(Id1)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  2 * out1.as_array())


    def test_CompositionOperator_adjoint4(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 2 * Identity(ig)
        
        d = G.compose(-Id1)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  - 2 * out1.as_array())

    def test_CompositionOperator_adjoint5(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 3 * Identity(ig)
        Id = Id1 - Identity(ig)
        d = G.compose(Id)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  2 * out1.as_array())
    
    def test_CompositionOperator_adjoint6(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        Id1 = 3 * Identity(ig)
        Id = ZeroOperator(ig)
        d = G.compose(Id)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  0 * out1.as_array())

    def stest_CompositionOperator_direct4(self):
        ig = self.ig
        data = self.data
        G = Gradient(domain_geometry=ig)
        

        sym = SymmetrizedGradient(domain_geometry=ig)
        Id2 = Identity(ig)
        
        d = CompositionOperator(sym, Id2)

        out1 = G.direct(data)
        out2 = d.direct(data)


        numpy.testing.assert_array_almost_equal(out2.get_item(0).as_array(),  out1.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(out2.get_item(1).as_array(),  out1.get_item(1).as_array())


    

