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
import numpy
from ccpi.framework import ImageGeometry, AcquisitionGeometry
from ccpi.framework import ImageData, AcquisitionData
from ccpi.framework import BlockDataContainer
import functools

from ccpi.optimisation.operators import Gradient, Identity, BlockOperator
from ccpi.optimisation.operators import LinearOperator

class TestGradient(unittest.TestCase):
    def test_Gradient(self): 
        N, M, K = 20, 30, 40
        channels = 10
        
        numpy.random.seed(1)
        
        # check range geometry, examples
        
        ig1 = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        ig2 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z = K) 
        ig3 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels) 
        ig4 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K) 
        
        G1 = Gradient(ig1, correlation = 'Space', backend='numpy') 
        print(G1.range_geometry().shape, '2D no channels')
            
        G4 = Gradient(ig3, correlation = 'SpaceChannels', backend='numpy')
        print(G4.range_geometry().shape, '2D with channels corr')  
        G5 = Gradient(ig3, correlation = 'Space', backend='numpy')
        print(G5.range_geometry().shape, '2D with channels no corr')
        
        G6 = Gradient(ig4, correlation = 'Space', backend='numpy')
        print(G6.range_geometry().shape, '3D with channels no corr')
        G7 = Gradient(ig4, correlation = 'SpaceChannels', backend='numpy')
        print(G7.range_geometry().shape, '3D with channels with corr')
        
        
        u = ig1.allocate(ImageGeometry.RANDOM)
        w = G1.range_geometry().allocate(ImageGeometry.RANDOM_INT)
        
        LHS = (G1.direct(u)*w).sum()
        RHS = (u * G1.adjoint(w)).sum()
        numpy.testing.assert_approx_equal(LHS, RHS, significant = 1)
        numpy.testing.assert_approx_equal(G1.norm(), numpy.sqrt(2*4), significant = 1)
            
        
        u1 = ig3.allocate('random')
        w1 = G4.range_geometry().allocate('random')
        LHS1 = (G4.direct(u1) * w1).sum()
        RHS1 = (u1 * G4.adjoint(w1)).sum() 
        numpy.testing.assert_approx_equal(LHS1, RHS1, significant=1)
        numpy.testing.assert_almost_equal(G4.norm(), numpy.sqrt(3*4), decimal = 0)
        
        u2 = ig4.allocate('random')
        w2 = G7.range_geometry().allocate('random')
        LHS2 = (G7.direct(u2) * w2).sum()
        RHS2 = (u2 * G7.adjoint(w2)).sum() 
        numpy.testing.assert_approx_equal(LHS2, RHS2, significant = 3)
        numpy.testing.assert_approx_equal(G7.norm(), numpy.sqrt(3*4), significant = 1)
        
        
        #check direct/adjoint for space/channels correlation
        
        ig_channel = ImageGeometry(voxel_num_x = 2, voxel_num_y = 3, channels = 2)
        G_no_channel = Gradient(ig_channel, correlation = 'Space', backend='numpy')
        G_channel = Gradient(ig_channel, correlation = 'SpaceChannels', backend='numpy')
        
        u3 = ig_channel.allocate('random_int')
        res_no_channel = G_no_channel.direct(u3)
        res_channel = G_channel.direct(u3)
        
        print(" Derivative for 3 directions, first is wrt Channel direction\n")
        print(res_channel[0].as_array())
        print(res_channel[1].as_array())
        print(res_channel[2].as_array())
        
        print(" Derivative for 2 directions, no Channel direction\n")
        print(res_no_channel[0].as_array())
        print(res_no_channel[1].as_array())  
        
        ig2D = ImageGeometry(voxel_num_x = 2, voxel_num_y = 3)
        u4 = ig2D.allocate('random_int')
        G2D = Gradient(ig2D, backend='numpy')
        res = G2D.direct(u4)  
        print(res[0].as_array())
        print(res[1].as_array())

        M, N = 20, 30
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random_int' )
        
        # check direct of Gradient and sparse matrix
        G = Gradient(ig, backend='numpy')
        norm1 = G.norm(iterations=300)
        print ("should be sqrt(8) {} {}".format(numpy.sqrt(8), norm1))
        numpy.testing.assert_almost_equal(norm1, numpy.sqrt(8), decimal=1)
        ig4 = ImageGeometry(M,N, channels=3)
        G4 = Gradient(ig4, correlation="SpaceChannels", backend='numpy')
        norm4 = G4.norm(iterations=300)
        print("should be sqrt(12) {} {}".format(numpy.sqrt(12), norm4))
        self.assertTrue((norm4 - numpy.sqrt(12))/norm4 < 0.2)

    def test_GradientOperator_4D(self):

        nc, nz, ny, nx = 3, 4, 5, 6
        size = nc * nz * ny * nx
        dim = [nc, nz, ny, nx]

        ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz, channels=nc)

        arr = numpy.arange(size).reshape(dim).astype(numpy.float32)**2

        data = ig.allocate()
        data.fill(arr)

        #neumann
        grad_py = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad_py.direct(data)
        gold_adjoint = grad_py.adjoint(gold_direct)

        grad_c = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
        out_direct = grad_c.direct(data)
        out_adjoint = grad_c.adjoint(out_direct)

        #print("GradientOperator, 4D, bnd_cond='Neumann', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("GradientOperator, 4D, bnd_cond='Neumann', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())

        #periodic
        grad_py = Gradient(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad_py.direct(data)
        gold_adjoint = grad_py.adjoint(gold_direct)

        grad_c = Gradient(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='c')
        out_direct = grad_c.direct(data)
        out_adjoint = grad_c.adjoint(out_direct)

        #print("Gradient, 4D, bnd_cond='Periodic', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("Gradient, 4D, bnd_cond='Periodic', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())

    def test_Gradient_4D_allocate(self):

        nc, nz, ny, nx = 3, 4, 5, 6
        size = nc * nz * ny * nx
        dim = [nc, nz, ny, nx]

        ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz, channels=nc)


        arr = numpy.arange(size).reshape(dim).astype(numpy.float32)**2

        data = ig.allocate()
        data.fill(arr)

        #numpy
        grad1 = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad1.direct(data)
        gold_adjoint = grad1.adjoint(gold_direct)

        grad2 = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        out_direct = grad2.range_geometry().allocate()
        out_adjoint = grad2.domain_geometry().allocate()
        grad2.direct(data, out=out_direct)
        grad2.adjoint(out_direct, out=out_adjoint)

        #print("GradientOperator, 4D, bnd_cond='Neumann', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("GradientOperator, 4D, bnd_cond='Neumann', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())

        #c
        grad1 = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
        gold_direct = grad1.direct(data)
        gold_adjoint = grad1.adjoint(gold_direct)

        grad2 = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
        out_direct = grad2.range_geometry().allocate()
        out_adjoint = grad2.domain_geometry().allocate()
        grad2.direct(data, out=out_direct)
        grad2.adjoint(out_direct, out=out_adjoint)

        #print("GradientOperator, 4D, bnd_cond='Neumann', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("GradientOperator, 4D, bnd_cond='Neumann', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())

    def test_Gradient_linearity(self):

        nc, nz, ny, nx = 3, 4, 5, 6
        ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz, channels=nc)

        grad = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
        self.assertTrue(LinearOperator.dot_test(grad))

        grad = Gradient(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='c')
        self.assertTrue(LinearOperator.dot_test(grad))

        grad = Gradient(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        self.assertTrue(LinearOperator.dot_test(grad))

        grad = Gradient(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='numpy')
        self.assertTrue(LinearOperator.dot_test(grad))
        
