# -*- coding: utf-8 -*-
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
from cil.optimisation.operators import BlockOperator
from cil.framework import BlockDataContainer
from cil.optimisation.operators import IdentityOperator
from cil.framework import ImageGeometry, ImageData
import numpy
from cil.optimisation.operators import FiniteDifferenceOperator
from cil.optimisation.operators import LinearOperator
initialise_tests()

class TestBlockOperator(unittest.TestCase):

    def test_BlockOperator(self):
        ig = [ ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) , \
               ImageGeometry(10,20,30) ]
        x = [ g.allocate() for g in ig ]
        ops = [ IdentityOperator(g) for g in ig ]

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
        
        K2 = BlockOperator(*(ops+ops), shape=(3,2))
        Y = K2.T.direct(X)
        # K.T (2,3) X (3,1) => output shape (2,1)
        self.assertTrue(Y.shape == (2,1))

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
            logging.info(str(ve))
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
            logging.info ("K col comp? {}".format(K.column_wise_compatible()))
            logging.info ("K row comp? {}".format(K.row_wise_compatible()))
            for op in ops:
                logging.info ("range {}".format(op.range_geometry().shape))
            for op in ops:
                logging.info ("domain {}".format(op.domain_geometry().shape))
            self.assertFalse(K.row_wise_compatible())
        except ValueError as ve:
            logging.info (str(ve))
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
        ops = [ IdentityOperator(g) for g in ig ]


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
        

    def test_IdentityOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        logging.info ("{} {}".format(img.shape, ig.shape))
        self.assertTrue(img.shape == (30,20,10))
        self.assertEqual(img.sum(), 0)
        Id = IdentityOperator(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())

    
    def test_FiniteDiffOperator(self):
        N, M = 200, 300
        
        ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N)    
        u = ig.allocate('random_int')
        G = FiniteDifferenceOperator(ig, direction=0, bnd_cond = 'Neumann')
        logging.info("{} {}".format(type(u), str(u.as_array())))    
        logging.info(str(G.direct(u).as_array()))
        LinearOperator.PowerMethod(G, range_is_domain=False)
        # Gradient Operator norm, for one direction should be close to 2
        numpy.testing.assert_allclose(LinearOperator.PowerMethod(G, range_is_domain=False), numpy.sqrt(4), atol=0.1)

        M1, N1, K1 = 200, 300, 2
        ig1 = ImageGeometry(voxel_num_x = M1, voxel_num_y = N1, channels = K1)
        u1 = ig1.allocate('random_int')
        G1 = FiniteDifferenceOperator(ig1, direction=2, bnd_cond = 'Periodic')
        logging.info(ig1.shape==u1.shape)
        logging.info(str(G1.norm()))
        numpy.testing.assert_allclose(G1.norm(), numpy.sqrt(4), atol=0.1)
