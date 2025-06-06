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
from unittest.mock import Mock, patch
import logging
import numpy
from cil.framework import ImageGeometry
from cil.optimisation.operators import GradientOperator
from cil.optimisation.operators import LinearOperator
from utils import initialise_tests

log = logging.getLogger(__name__)
initialise_tests()


class TestGradientOperator(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        N, M, K = 10, 11, 12
        channels = 13

        self.voxel_size_x, self.voxel_size_y, self.voxel_size_z = 0.1, 0.3, 0.6

        self.ig_2D = ImageGeometry(voxel_num_x = M, voxel_num_y = N)
        self.ig_2D_chan = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels)
        self.ig_3D = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z= K)
        self.ig_3D_chan = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K)

        self.ig_2D_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_size_x=self.voxel_size_x, voxel_size_y=self.voxel_size_y)
        self.ig_2D_chan_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_size_x=self.voxel_size_x, voxel_size_y=self.voxel_size_y)
        self.ig_3D_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z= K, voxel_size_x=self.voxel_size_x, voxel_size_y=self.voxel_size_y, voxel_size_z=self.voxel_size_z)
        self.ig_3D_chan_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K, voxel_size_y=self.voxel_size_y, voxel_size_x=self.voxel_size_x, voxel_size_z=self.voxel_size_z)

        self.list_geometries = [self.ig_2D, self.ig_2D_chan, self.ig_3D, self.ig_3D_chan,
                                self.ig_2D_voxel, self.ig_2D_chan_voxel, self.ig_3D_voxel, self.ig_3D_chan_voxel]

        test_matrix_backend_numpy = {  'backend':'numpy',
                                            'bconditions': ['Neumann', 'Periodic'],
                                            'correlation':['Space','SpaceChannels'],
                                            'method':['forward', 'backward', 'centered']}

        test_matrix_backend_c ={   'backend':'c',
                                        'bconditions': ['Neumann', 'Periodic'],
                                        'correlation':['SpaceChannels'],
                                        'method':['forward']}

        self.test_backend_configurations = [test_matrix_backend_numpy, test_matrix_backend_c]

    def print_assertion_info(self, geom = None, bnd = None, backend = None, method = None, corr = None, split = None):

        log.info("Test Failed")
        log.info("ImageGeometry %s", geom)
        log.info("Bnd Cond %s", bnd)
        log.info("Backend %s", backend)
        log.info("Method %s", method)
        log.info("Correlation %s", corr)
        if split is not None:
            log.info("Split %s", split)


    def test_GradientOperator_linearity(self):

        for config in self.test_backend_configurations:
            backend = config.get('backend')

            for geom in self.list_geometries:
                for bnd in config.get('bconditions'):
                    for corr in config.get('correlation'):
                        for method in config.get('method'):

                            Grad = GradientOperator(geom,
                                                    bnd_cond = bnd,
                                                    backend = backend,
                                                    correlation = corr, method = method)
                            try:
                                for sd in [5, 10, 15]:
                                    self.assertTrue(LinearOperator.dot_test(Grad, seed=sd))
                            except:
                                self.print_assertion_info(geom,bnd,backend,corr,method,None)
                                raise


    def test_GradientOperator_norm(self):

        for config in self.test_backend_configurations:
            backend = config.get('backend')

            for geom in self.list_geometries:
                for bnd in config.get('bconditions'):
                    for corr in config.get('correlation'):
                        for method in config.get('method'):

                            if geom.channels == 1:

                                if geom.length == 2:
                                    norm = numpy.sqrt(
                                        (2/geom.voxel_size_y)**2 + (2/geom.voxel_size_x)**2)
                                elif geom.length == 3:
                                    norm = numpy.sqrt(
                                        (2/geom.voxel_size_z)**2 + (2/geom.voxel_size_y)**2 + (2/geom.voxel_size_x)**2)

                            else:

                                if corr == 'Space':
                                    if geom.length ==3:
                                        norm = numpy.sqrt(
                                            (2/geom.voxel_size_y)**2 + (2/geom.voxel_size_x)**2)
                                    else:
                                        norm = numpy.sqrt((2/geom.voxel_size_z)**2 + (2/geom.voxel_size_y)**2 + (2/geom.voxel_size_x)**2)
                                else:

                                    if geom.length ==3:
                                        norm = numpy.sqrt(
                                            (2/geom.channel_spacing)**2 + (2/geom.voxel_size_y)**2 + (2/geom.voxel_size_x)**2)
                                    else:
                                        norm = numpy.sqrt((2/geom.channel_spacing)**2 + (2/geom.voxel_size_z)**2 + (2/geom.voxel_size_y)**2 + (2/geom.voxel_size_x)**2)


                            Grad = GradientOperator(geom,
                                                    bnd_cond= bnd,
                                                    backend = backend,
                                                    correlation= corr, method=method)
                            try:
                                self.assertAlmostEqual(Grad.norm(), norm, 6)
                            except AssertionError:
                                self.print_assertion_info(geom, bnd,backend,corr,method,None)
                                raise


    def test_GradientOperator_in_place_vs_allocate_direct(self):

        for config in self.test_backend_configurations:
            backend = config.get('backend')

            for geom in self.list_geometries:
                for bnd in config.get('bconditions'):
                    for corr in config.get('correlation'):
                        for method in config.get('method'):

                            Grad = GradientOperator(geom,
                                                    bnd_cond = bnd,
                                                    backend = backend,
                                                    correlation = corr, method=method)
                            tmp_x = Grad.domain.allocate('random', seed=5)
                            res_in_place = Grad.direct(tmp_x)
                            res_allocate = Grad.range.allocate()
                            Grad.direct(tmp_x, out = res_allocate)

                            for m in range(len(Grad.range.geometries)):

                                try:
                                    numpy.testing.assert_array_almost_equal(res_in_place[m].array,
                                                                    res_allocate[m].array)
                                except:
                                    self.print_assertion_info(geom,bnd,backend,corr,method,None)
                                    raise


    def test_GradientOperator_in_place_vs_allocate_adjoint(self):

        for config in self.test_backend_configurations:
            backend = config.get('backend')

            for geom in self.list_geometries:
                for bnd in config.get('bconditions'):
                    for corr in config.get('correlation'):
                        for method in config.get('method'):

                            Grad = GradientOperator(geom,
                                                    bnd_cond = bnd,
                                                    backend = backend,
                                                    correlation = corr, method=method)
                            tmp_x = Grad.range.allocate('random', seed=5)
                            res_in_place = Grad.adjoint(tmp_x)
                            res_allocate = Grad.domain.allocate()
                            Grad.adjoint(tmp_x, out = res_allocate)

                            try:
                                numpy.testing.assert_array_almost_equal(res_in_place.array,
                                                                res_allocate.array)
                            except:
                                self.print_assertion_info(geom,bnd,backend,corr,method, None)
                                raise


    def test_GradientOperator_range_shape(self):
        Grad2D = GradientOperator(self.ig_2D, backend='numpy')
        numpy.testing.assert_equal(Grad2D.range.shape, (2,1))

        Grad2D = GradientOperator(self.ig_2D_chan, correlation="Space", backend='numpy')
        numpy.testing.assert_equal(Grad2D.range.shape, (2,1))

        Grad2D_chan = GradientOperator(self.ig_2D_chan, correlation="SpaceChannels", backend='numpy')
        numpy.testing.assert_equal(Grad2D_chan.range.shape, (3,1))

        Grad3D = GradientOperator(self.ig_3D, backend='numpy')
        numpy.testing.assert_equal(Grad3D.range.shape, (3,1))

        Grad3D_chan = GradientOperator(self.ig_3D_chan, correlation="Space", backend='numpy')
        numpy.testing.assert_equal(Grad3D_chan.range.shape, (3,1))

        Grad3D_chan = GradientOperator(self.ig_3D_chan, correlation="SpaceChannels", backend='numpy')
        numpy.testing.assert_equal(Grad3D_chan.range.shape, (4,1))

        Grad2D = GradientOperator(self.ig_2D, backend='c')
        numpy.testing.assert_equal(Grad2D.range.shape, (2,1))

        Grad2D_chan = GradientOperator(self.ig_2D_chan, correlation="SpaceChannels", backend='c')
        numpy.testing.assert_equal(Grad2D_chan.range.shape, (3,1))

        Grad3D = GradientOperator(self.ig_3D, backend='c')
        numpy.testing.assert_equal(Grad3D.range.shape, (3,1))

        Grad3D_chan = GradientOperator(self.ig_3D_chan, correlation="SpaceChannels", backend='c')
        numpy.testing.assert_equal(Grad3D_chan.range.shape, (4,1))


    def test_GradientOperator_split_shape(self):
        for geom in [self.ig_2D, self.ig_2D_chan, self.ig_3D, self.ig_3D_chan]:

            for split in [True, False]:

                Grad = GradientOperator(geom, split = split, correlation="SpaceChannels", backend='c')

                if geom == self.ig_2D:
                    shape = (2,1)
                elif geom == self.ig_3D:
                    shape = (3,1)
                elif geom == self.ig_2D_chan:
                    if split:
                        shape = (2,1)
                    else:
                        shape = (3,1)
                elif geom == self.ig_3D_chan:
                    if split:
                        shape = (2,1)
                    else:
                        shape = (4,1)

                try:
                    numpy.testing.assert_equal(Grad.range.shape, shape)
                except:
                    self.print_assertion_info(geom, None, None, None, split=split)
                    raise


    def test_GradientOperator_split_direct_adjoint(self):
        # Test split for direct and adjoint in 2D + channels geometry
        geom = self.ig_2D_chan
        Grad2D_split_false = GradientOperator(geom, split = False, correlation="SpaceChannels", backend='c')
        Grad2D_split_true = GradientOperator(geom, split = True, correlation="SpaceChannels", backend='c')

        tmp_x = geom.allocate('random', seed=5)
        res1 = Grad2D_split_false.direct(tmp_x)
        res2 = Grad2D_split_true.direct(tmp_x)
        res1_adj = Grad2D_split_false.adjoint(res1)
        res2_adj = Grad2D_split_true.adjoint(res2)

        numpy.testing.assert_array_almost_equal(res1[0].array, res2[0].array)
        numpy.testing.assert_array_almost_equal(res1[1].array, res2[1][0].array)
        numpy.testing.assert_array_almost_equal(res1[2].array, res2[1][1].array)
        numpy.testing.assert_array_almost_equal(res1_adj.array, res2_adj.array)

        # Test split for direct and adjoint in 3D + channels geometry
        geom = self.ig_3D_chan
        Grad3D_split_false = GradientOperator(geom, split = False, correlation="SpaceChannels", backend='c')
        Grad3D_split_true = GradientOperator(geom, split = True, correlation="SpaceChannels", backend='c')

        tmp_x = geom.allocate('random', seed=6)
        res1 = Grad3D_split_false.direct(tmp_x)
        res2 = Grad3D_split_true.direct(tmp_x)
        res1_adj = Grad3D_split_false.adjoint(res1)
        res2_adj = Grad3D_split_true.adjoint(res2)

        numpy.testing.assert_array_almost_equal(res1[0].array, res2[0].array)
        numpy.testing.assert_array_almost_equal(res1[1].array, res2[1][0].array)
        numpy.testing.assert_array_almost_equal(res1[2].array, res2[1][1].array)
        numpy.testing.assert_array_almost_equal(res1[3].array, res2[1][2].array)
        numpy.testing.assert_array_almost_equal(res1_adj.array, res2_adj.array)


    def test_Gradient_operator_numpy_vs_c(self):

        #only use configurations supported by c backend
        config = self.test_backend_configurations[1]

        for geom in self.list_geometries:
            for bnd in config.get('bconditions'):
                for corr in config.get('correlation'):
                    for method in config.get('method'):

                        Grad_c = GradientOperator(geom, bnd_cond = bnd, method=method, correlation= corr, backend = 'c')
                        Grad_numpy =  GradientOperator(geom, bnd_cond = bnd, method=method, correlation= corr, backend = 'numpy')

                        tmp_x = geom.allocate('random', seed=5)
                        res1_c = Grad_c.direct(tmp_x)
                        res1_np = Grad_numpy.direct(tmp_x)

                        # Check direct of numpy vs c in place
                        for m in range(len(Grad_c.range.geometries)):
                            try:
                                numpy.testing.assert_array_almost_equal(res1_c[m].array,
                                                                res1_np[m].array)
                            except:
                                self.print_assertion_info(geom, bnd, None, None, corr, None)
                                raise

                        res1_c_out = Grad_c.range.allocate()
                        Grad_c.direct(tmp_x, out = res1_c_out)

                        res1_np_out = Grad_numpy.range.allocate()
                        Grad_numpy.direct(tmp_x, out = res1_np_out)

                        # Check direct of numpy vs c allocate
                        for m in range(len(Grad_c.range.geometries)):
                            try:
                                numpy.testing.assert_array_almost_equal(res1_c_out[m].array,
                                                                res1_np_out[m].array)
                            except:
                                self.print_assertion_info(geom, bnd, None, None, corr, None)
                                raise

                        tmp_x = Grad_c.range.allocate('random', seed=6)
                        res1_c = Grad_c.adjoint(tmp_x)
                        res1_np = Grad_numpy.adjoint(tmp_x)

                        # Check adjoint of numpy vs c in place
                        try:
                            numpy.testing.assert_array_almost_equal(res1_c.array,
                                                            res1_np.array, decimal=5)
                        except:
                            self.print_assertion_info(geom, bnd, None, None, corr, None)
                            raise

                        res1_c_out = Grad_c.domain.allocate()
                        Grad_c.adjoint(tmp_x, out = res1_c_out)

                        res1_np_out = Grad_numpy.domain.allocate()
                        Grad_numpy.adjoint(tmp_x, out = res1_np_out)

                        # Check adjoint of numpy vs c allocate
                        try:
                            numpy.testing.assert_array_almost_equal(res1_c_out.array,
                                                            res1_np_out.array, decimal=5)
                        except:
                            self.print_assertion_info(geom, bnd, None, None, corr, None)
                            raise


    def test_GradientOperator_for_pseudo_2D_geometries(self):
            numpy.random.seed(1)
            # ImageGeometry shape (5,5,1)
            ig1 = ImageGeometry(voxel_num_x = 1, voxel_num_y = 5, voxel_num_z=5,
                                voxel_size_x = 0.4, voxel_size_y = 0.2, voxel_size_z=0.6)
            # ImageGeometry shape (1,5,5)
            ig2 = ImageGeometry(voxel_num_x = 5, voxel_num_y = 5, voxel_num_z=1,
                                voxel_size_x = 0.1, voxel_size_y = 0.2, voxel_size_z=0.4)
            # ImageGeometry shape (5,1,5)
            ig3 = ImageGeometry(voxel_num_x = 5, voxel_num_y = 1, voxel_num_z=5,
                                voxel_size_x = 0.6, voxel_size_y = 0.4, voxel_size_z=0.3)

            data1 = ig1.allocate('random', seed=5)
            data2 = ig2.allocate('random', seed=6)
            data3 = ig3.allocate('random', seed=7)

            data = [data1, data2, data3]
            ig = [ig1, ig2, ig3]

            for i in range(3):

                ########################################
                ##### Test Gradient numpy backend  #####
                ########################################
                Grad_numpy = GradientOperator(ig[i], backend='numpy')
                res1 = Grad_numpy.direct(data[i])
                res2 = Grad_numpy.range_geometry().allocate()
                Grad_numpy.direct(data[i], out=res2)

                # test direct with and without out
                numpy.testing.assert_array_almost_equal(res1[0].as_array(), res2[0].as_array())
                numpy.testing.assert_array_almost_equal(res1[1].as_array(), res2[1].as_array())

                # test adjoint with and without out
                res3 = Grad_numpy.adjoint(res1)
                res4 = Grad_numpy.domain_geometry().allocate()
                Grad_numpy.adjoint(res2, out=res4)
                numpy.testing.assert_array_almost_equal(res3.as_array(), res4.as_array())

                # test dot_test
                for sd in [5, 10, 15]:
                    self.assertTrue(LinearOperator.dot_test(Grad_numpy, seed=sd))

                # test shape of output of direct
                self.assertEqual(res1[0].shape, ig[i].shape)
                self.assertEqual(res1.shape, (2,1))

                ########################################
                ##### Test Gradient c backend  #####
                ########################################
                Grad_c = GradientOperator(ig[i], backend='c')

                # test direct with and without out
                res5 = Grad_c.direct(data[i])
                res6 = Grad_c.range_geometry().allocate()*0.
                Grad_c.direct(data[i], out=res6)

                numpy.testing.assert_array_almost_equal(res5[0].as_array(), res6[0].as_array())
                numpy.testing.assert_array_almost_equal(res5[1].as_array(), res6[1].as_array())

                # test adjoint
                res7 = Grad_c.adjoint(res5)
                res8 = Grad_c.domain_geometry().allocate()*0.
                Grad_c.adjoint(res5, out=res8)
                numpy.testing.assert_array_almost_equal(res7.as_array(), res8.as_array())

                # test dot_test
                for sd in [5, 10, 15]:
                    self.assertTrue(LinearOperator.dot_test(Grad_c, seed = sd))

                # test direct numpy vs direct c backends (with and without out)
                numpy.testing.assert_array_almost_equal(res5[0].as_array(), res1[0].as_array())
                numpy.testing.assert_array_almost_equal(res6[1].as_array(), res2[1].as_array())


    def test_GradientOperator_complex_data(self):
        # make complex dtype
        self.ig_2D.dtype = numpy.complex64
        x = self.ig_2D.allocate('random', seed=5)

        Grad = GradientOperator(domain_geometry=self.ig_2D, backend='numpy')

        res1 = Grad.direct(x)
        res2 = Grad.range.allocate()
        Grad.direct(x, out=res2)

        numpy.testing.assert_array_almost_equal(res1[0].as_array(), res2[0].as_array())
        numpy.testing.assert_array_almost_equal(res1[1].as_array(), res2[1].as_array())

        # check dot_test
        for sd in [5, 10, 15]:
            self.assertTrue(LinearOperator.dot_test(Grad, seed=sd))


    def test_GradientOperator_cpp_failure_direct(self):
        # Simulate the failure by setting the status to non-zero
            ig = ImageGeometry(voxel_num_x = 2, voxel_num_y = 3, voxel_num_z=4)
            data = ig.allocate('random', seed=5)

            Grad = GradientOperator(ig, backend='c')
        
            # with the call to status = self.fd(args) returning a non-zero value
            Grad.operator.fd = Mock(return_value=-1)
            with self.assertRaises(RuntimeError):
                Grad.direct(data)


    def test_GradientOperator_cpp_failure_adjoint(self):
        # Simulate the failure by setting the status to non-zero
            ig = ImageGeometry(voxel_num_x = 2, voxel_num_y = 3, voxel_num_z=4)
            data = ig.allocate('random', seed=5)

            Grad = GradientOperator(ig, backend='c')
            res_direct = Grad.direct(data)
        
            # with the call to status = self.fd(args) returning a non-zero value
            Grad.operator.fd = Mock(return_value=-1)
            with self.assertRaises(RuntimeError):
                Grad.adjoint(res_direct)
