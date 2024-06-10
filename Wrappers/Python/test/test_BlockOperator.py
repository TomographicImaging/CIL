#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest
from utils import initialise_tests
import logging
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.framework import BlockDataContainer
from cil.optimisation.operators import IdentityOperator
from cil.framework import ImageGeometry, ImageData, BlockGeometry
import numpy
from cil.optimisation.operators import FiniteDifferenceOperator
from testclass import CCPiTestClass
from timeit import default_timer as timer

log = logging.getLogger(__name__)
initialise_tests()

def dt(steps):
    return steps[-1] - steps[-2]

class TestBlockOperator(CCPiTestClass):
    
    def setUp(self):
        numpy.random.seed(1)


    def test_BlockOperator(self):
        M, N  = 3, 4
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random')

        G = GradientOperator(ig)
        Id = IdentityOperator(ig)

        B = BlockOperator(G, Id)
        # Nx1 case
        u = ig.allocate('random')
        z1 = B.direct(u)

        res = B.range_geometry().allocate()
        #res = z1.copy()
        B.direct(u, out=res)

        self.assertBlockDataContainerEqual(z1, res)

        z1 = B.range_geometry().allocate(ImageGeometry.RANDOM)

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

        self.assertNumpyArrayEqual(v.as_array(),vv.as_array())
        # test adjoint

        BB = BlockOperator( Id, 2 * Id)
        u = ig.allocate(1)
        z1 = BB.direct(u)
        res = BB.range_geometry().allocate(0)
        BB.direct(u, out=res)

        self.assertNumpyArrayEqual(z1.get_item(0).as_array(),
                                   u.as_array())
        self.assertNumpyArrayEqual(z1.get_item(1).as_array(),
                                   2 * u.as_array())
        self.assertNumpyArrayEqual(res.get_item(0).as_array(),
                                   u.as_array())
        self.assertNumpyArrayEqual(res.get_item(1).as_array(),
                                   2 * u.as_array())

        x1 = BB.adjoint(z1)

        res1 = BB.domain_geometry().allocate()
        BB.adjoint(z1, out=res1)
        self.assertNumpyArrayEqual(x1.as_array(),
                                   res1.as_array())

        self.assertNumpyArrayEqual(x1.as_array(),
                                   5 * u.as_array())
        self.assertNumpyArrayEqual(res1.as_array(),
                                   5 * u.as_array())
        #################################################

        BB = BlockOperator( Id, 2 * Id, 3 * Id,  Id, shape=(2,2))
        B = BB
        u = ig.allocate(1)
        U = BlockDataContainer(u,u)
        z1 = B.direct(U)

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
        out = B.domain_geometry().allocate()

        for col in range(B.shape[1]):
            for row in range(B.shape[0]):
                if row == 0:
                    el = B.get_item(row,col).adjoint(z1.get_item(row))
                else:
                    el += B.get_item(row,col).adjoint(z1.get_item(row))
            out.get_item(col).fill(el)

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

        B1 = BlockOperator(G, Id)
        U = ig.allocate(ImageGeometry.RANDOM)
        #U = BlockDataContainer(u,u)
        RES1 = B1.range_geometry().allocate()

        Z1 = B1.direct(U)
        B1.direct(U, out = RES1)

        self.assertBlockDataContainerEqual(Z1,RES1)
        
    def test_block_operator_1_1(self):
        M, N ,W = 3, 4, 5
        ig = ImageGeometry(M, N, W)
        operator0=IdentityOperator(ig)
        operator1=-IdentityOperator(ig)
        K = BlockOperator(operator0, operator1, shape = (1,2))
        bg=BlockGeometry(ig, ig)
        data=bg.allocate('random', seed=2)
        ans = K.direct(data)
        #self.assertNumpyArrayEqual( ans.shape, ig.allocate(0).as_array())
        self.assertNumpyArrayEqual( ans.as_array(), ig.allocate(0).as_array())
        self.assertFalse(isinstance(ans, BlockDataContainer))
        
        self.assertEqual(K.range_geometry(), ig)

        
        ans2 = K.adjoint(ans)
        self.assertTrue(isinstance(ans2, BlockDataContainer))
        self.assertNumpyArrayEqual(ans2.shape, (2,1))
        
        range_data=ans.geometry.allocate('random', seed=2)
        ans3=K.adjoint(range_data)
        self.assertNumpyArrayEqual(ans3.shape, (2,1))
        self.assertNumpyArrayEqual(ans3.get_item(0).as_array(), range_data.as_array())
        self.assertNumpyArrayEqual(ans3.get_item(1).as_array(), -range_data.as_array())
        
        M, N ,W = 3, 4, 5
        ig = ImageGeometry(M, N, W)
        operator0=IdentityOperator(ig)
        operator1=-IdentityOperator(ig)
        K = BlockOperator(operator0, operator1, shape = (2,1))
        bg=BlockGeometry(ig, ig)
        data=ig.allocate('random', seed=2)
        ans = K.direct(data)
        #self.assertNumpyArrayEqual( ans.shape, ig.allocate(0).as_array())
        self.assertNumpyArrayEqual( ans.get_item(0).as_array(), data.as_array())
        self.assertNumpyArrayEqual( ans.get_item(1).as_array(), -data.as_array())
        
        self.assertEqual(K.domain_geometry(), ig)
        
        

    @unittest.skipIf(True, 'Skipping time tests')
    def test_timedifference(self):
        M, N ,W = 100, 512, 512
        ig = ImageGeometry(M, N, W)
        arr = ig.allocate('random')

        G = GradientOperator(ig, backend='numpy')
        Id = IdentityOperator(ig)

        B = BlockOperator(G, Id)


        # Nx1 case
        u = ig.allocate('random')
        steps = [timer()]
        i = 0
        n = 10.
        t1 = t2 = 0
        res = B.range_geometry().allocate()

        while (i < n):
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

        self.assertGreater(t1,t2)

        steps = [timer()]
        i = 0
        #n = 50.
        t1 = t2 = 0
        resd = B.domain_geometry().allocate()
        z1 = B.direct(u)
        #B.adjoint(z1, out=resd)

        while (i < n):
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


    def test_BlockOperatorLinearValidity(self):
        M, N  = 3, 4
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random', seed=1)

        G = GradientOperator(ig)
        Id = IdentityOperator(ig)

        B = BlockOperator(G, Id)
        # Nx1 case
        u = ig.allocate('random', seed=2)
        w = B.range_geometry().allocate(ImageGeometry.RANDOM, seed=3)
        w1 = B.direct(u)
        u1 = B.adjoint(w)
        self.assertAlmostEqual((w * w1).sum() , (u1*u).sum(), places=5)



    def test_norms(self):
        numpy.random.seed(1)
        N, M = 200, 300

        ig = ImageGeometry(N, M)
        G = GradientOperator(ig)
        G2 = GradientOperator(ig)

        A=BlockOperator(G,G2)


        #calculates norm
        self.assertAlmostEqual(G.norm(), numpy.sqrt(8), 2)
        self.assertAlmostEqual(G2.norm(), numpy.sqrt(8), 2)
        self.assertAlmostEqual(A.norm(), numpy.sqrt(16), 2)
        self.assertAlmostEqual(A.get_norms_as_list()[0], numpy.sqrt(8), 2)
        self.assertAlmostEqual(A.get_norms_as_list()[1], numpy.sqrt(8), 2)


        #sets_norm
        A.set_norms([2,3])
        #gets cached norm
        self.assertListEqual(A.get_norms_as_list(), [2,3], 2)
        self.assertEqual(A.norm(), numpy.sqrt(13))


        #Check that it changes the underlying operators
        self.assertEqual(A.operators[0]._norm, 2)
        self.assertEqual(A.operators[1]._norm, 3)

        #sets cache to None
        A.set_norms([None, None])
        #recalculates norm
        self.assertAlmostEqual(A.norm(), numpy.sqrt(16), 2)
        self.assertAlmostEqual(A.get_norms_as_list()[0], numpy.sqrt(8), 2)
        self.assertAlmostEqual(A.get_norms_as_list()[1], numpy.sqrt(8), 2)

        #Check the warnings on set_norms
        #Check the length of list that is passed
        with self.assertRaises(ValueError):
            A.set_norms([1])
        #Check that elements in the list are numbers or None
        with self.assertRaises(TypeError):
            A.set_norms(['Banana', 'Apple'])
        #Check that numbers in the list are positive
        with self.assertRaises(ValueError):
            A.set_norms([-1,-3])

    


    def test_BlockOperator(self):
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) ]
        x = [ g.allocate() for g in ig ]
        ops = [ IdentityOperator(g) for g in ig ]

        K = BlockOperator(*ops)
        X = BlockDataContainer(x[0])
        Y = K.direct(X)
        self.assertNumpyArrayEqual(Y.shape ,K.shape)

        numpy.testing.assert_array_equal(Y.get_item(0).as_array(),X.get_item(0).as_array())
        numpy.testing.assert_array_equal(Y.get_item(1).as_array(),X.get_item(0).as_array())
        #numpy.testing.assert_array_equal(Y.get_item(2).as_array(),X.get_item(2).as_array())

        X = BlockDataContainer(*x) + 1
        Y = K.T.direct(X)
        # K.T (1,3) X (3,1) => output shape (1,1)
        self.assertFalse(isinstance(Y, BlockDataContainer))
        zero = numpy.zeros(X.get_item(0).shape)
        numpy.testing.assert_array_equal(Y.as_array(),len(x)+zero)

        K2 = BlockOperator(*(ops+ops), shape=(3,2))
        Y = K2.T.direct(X)
        # K.T (2,3) X (3,1) => output shape (2,1)
        self.assertNumpyArrayEqual(Y.shape , (2,1))

        try:
            # this should fail as the domain is not compatible
            ig = [ ImageGeometry(10,20,31) , \
                ImageGeometry(10,20,30) , \
                ImageGeometry(10,20,30) ]
            x = [ g.allocate() for g in ig ]
            ops = [ IdentityOperator(g) for g in ig ]

            K = BlockOperator(*ops)
            self.assertFalse(K.column_wise_compatible())
        except ValueError as ve:
            log.info(str(ve))
            self.assertTrue(True)

        try:
            # this should fail as the range is not compatible
            ig = [ ImageGeometry(10,20,30) , \
                ImageGeometry(10,20,30) , \
                ImageGeometry(10,20,30) ]
            rg0 = [ ImageGeometry(10,20,31) , \
                ImageGeometry(10,20,31) , \
                ImageGeometry(10,20,31) ]
            rg1 = [ ImageGeometry(10,22,31) , \
                   ImageGeometry(10,22,31) , \
                   ImageGeometry(10,20,31) ]
            x = [ g.allocate() for g in ig ]
            ops = [ IdentityOperator(g, range_geometry=r) for g,r in zip(ig, rg0) ]
            ops += [ IdentityOperator(g, range_geometry=r) for g,r in zip(ig, rg1) ]

            K = BlockOperator(*ops, shape=(2,3))
            log.info("K col comp? %r", K.column_wise_compatible())
            log.info("K row comp? %r", K.row_wise_compatible())
            for op in ops:
                log.info("range %r", op.range_geometry().shape)
            for op in ops:
                log.info("domain %r", op.domain_geometry().shape)
            self.assertFalse(K.row_wise_compatible())
        except ValueError as ve:
            log.info(str(ve))
            self.assertTrue(True)


    def test_ScaledBlockOperatorSingleScalar(self):
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) ]
        x = [ g.allocate() for g in ig ]
        ops = [ IdentityOperator(g) for g in ig ]

        val = 1
        # test limit as non Scaled
        scalar = 1
        k = BlockOperator(*ops)
        K = scalar * k
        X = BlockDataContainer(*x) + val

        Y = K.T.direct(X)
        self.assertFalse(isinstance(Y, BlockDataContainer))
        zero = numpy.zeros(X.get_item(0).shape)
        xx = numpy.asarray([val for _ in x])
        numpy.testing.assert_array_equal(Y.as_array(),((scalar*xx).sum()+zero))

        scalar = 0.5
        k = BlockOperator(*ops)
        K = scalar * k
        X = BlockDataContainer(*x) + 1

        Y = K.T.direct(X)
        self.assertFalse(isinstance(Y, BlockDataContainer))
        zero = numpy.zeros(X.get_item(0).shape)
        numpy.testing.assert_array_equal(Y.as_array(),scalar*(len(x)+zero))


    def test_ScaledBlockOperatorScalarList(self):
        ig = [ ImageGeometry(2,3) , \
               #ImageGeometry(10,20,30) , \
               ImageGeometry(2,3    ) ]
        x = [ g.allocate(0) for g in ig ]
        ops = [ IdentityOperator(g) for g in ig ]


        # test limit as non Scaled
        scalar = numpy.asarray([1 for _ in x])
        k = BlockOperator(*ops)
        K = scalar * k
        val = 1
        X = BlockDataContainer(*x) + val

        Y = K.T.direct(X)
        self.assertFalse(isinstance(Y, BlockDataContainer))
        zero = numpy.zeros(ig[0].shape)
        xx = numpy.asarray([val for _ in x])
        numpy.testing.assert_array_equal(Y.as_array(),(scalar*xx).sum()+zero)

        scalar = numpy.asarray([i+1 for i,el in enumerate(x)])
        #scalar = numpy.asarray([6,0])
        k = BlockOperator(*ops)
        K = scalar * k
        X = BlockDataContainer(*x) + val
        Y = K.T.direct(X)
        self.assertFalse(isinstance(Y, BlockDataContainer))
        zero = numpy.zeros(ig[0].shape)
        xx = numpy.asarray([val for _ in x])


        numpy.testing.assert_array_equal(Y.as_array(),
          (scalar*xx).sum()+zero)


    def test_IdentityOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        log.info("%r %r", img.shape, ig.shape)
        self.assertNumpyArrayEqual(img.shape , (30,20,10))
        self.assertEqual(img.sum(), 0)
        Id = IdentityOperator(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())


    def test_FiniteDiffOperator(self):
        N, M = 200, 300

        ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N)
        u = ig.allocate('random_int')
        G = FiniteDifferenceOperator(ig, direction=0, bnd_cond = 'Neumann')
        log.info("%s %s", type(u), u.as_array())
        log.info("%s", G.direct(u).as_array())
        # Gradient Operator norm, for one direction should be close to 2
        numpy.testing.assert_allclose(G.norm(), numpy.sqrt(4), atol=0.1)

        M1, N1, K1 = 200, 300, 2
        ig1 = ImageGeometry(voxel_num_x = M1, voxel_num_y = N1, channels = K1)
        u1 = ig1.allocate('random_int')
        G1 = FiniteDifferenceOperator(ig1, direction=2, bnd_cond = 'Periodic')
        log.info(ig1.shape==u1.shape)
        log.info("%s", G1.norm())
        numpy.testing.assert_allclose(G1.norm(), numpy.sqrt(4), atol=0.1)
