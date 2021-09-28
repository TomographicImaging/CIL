# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

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
from cil.framework import ImageGeometry


from cil.optimisation.operators import GradientOperator
from cil.optimisation.operators import LinearOperator

class TestGradientOperator(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        # setup geometries for different tests
        numpy.random.seed(1)
        N, M, K, channels = 2, 3, 5, 3
        self.ig_2D = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        self.ig_2D_channels = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels) 
        self.ig_3D = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z= K) 
        self.ig_3D_channels = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels)         

        super(TestGradientOperator, self).__init__(*args, **kwargs)

    def test_GradientOperator(self): 
        N, M, K = 20, 30, 40
        channels = 10
        
        numpy.random.seed(1)
        
        # check range geometry, examples
        
        ig1 = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        ig3 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels) 
        ig4 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K) 
        
        G1 = GradientOperator(ig1, correlation = 'Space', backend='numpy') 
        print(G1.range_geometry().shape, '2D no channels')
            
        G4 = GradientOperator(ig3, correlation = 'SpaceChannels', backend='numpy')
        print(G4.range_geometry().shape, '2D with channels corr')  
        G5 = GradientOperator(ig3, correlation = 'Space', backend='numpy')
        print(G5.range_geometry().shape, '2D with channels no corr')
        
        G6 = GradientOperator(ig4, correlation = 'Space', backend='numpy')
        print(G6.range_geometry().shape, '3D with channels no corr')
        G7 = GradientOperator(ig4, correlation = 'SpaceChannels', backend='numpy')
        print(G7.range_geometry().shape, '3D with channels with corr')
        
        u = ig1.allocate(ImageGeometry.RANDOM)
        w = G1.range_geometry().allocate(ImageGeometry.RANDOM)
        
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
        G_no_channel = GradientOperator(ig_channel, correlation = 'Space', backend='numpy')
        G_channel = GradientOperator(ig_channel, correlation = 'SpaceChannels', backend='numpy')
        
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
        G2D = GradientOperator(ig2D, backend='numpy')
        res = G2D.direct(u4)  
        print(res[0].as_array())
        print(res[1].as_array())

        M, N = 20, 30
        ig = ImageGeometry(M, N)
        
        # check direct of GradientOperator and sparse matrix
        G = GradientOperator(ig, backend='numpy')
        norm1 = G.norm(iterations=300)
        print ("should be sqrt(8) {} {}".format(numpy.sqrt(8), norm1))
        numpy.testing.assert_almost_equal(norm1, numpy.sqrt(8), decimal=1)
        ig4 = ImageGeometry(M,N, channels=3)
        G4 = GradientOperator(ig4, correlation="SpaceChannels", backend='numpy')
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
        grad_py = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad_py.direct(data)
        gold_adjoint = grad_py.adjoint(gold_direct)

        grad_c = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
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
        grad_py = GradientOperator(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad_py.direct(data)
        gold_adjoint = grad_py.adjoint(gold_direct)

        grad_c = GradientOperator(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='c')
        out_direct = grad_c.direct(data)
        out_adjoint = grad_c.adjoint(out_direct)

        #print("GradientOperator, 4D, bnd_cond='Periodic', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("GradientOperator, 4D, bnd_cond='Periodic', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())


        #reordered
        data.reorder(['vertical','horizontal_x','channel','horizontal_y'])
        grad_py = GradientOperator(data.geometry, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad_py.direct(data)
        gold_adjoint = grad_py.adjoint(gold_direct)

        grad_c = GradientOperator(data.geometry, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
        out_direct = grad_c.direct(data)
        out_adjoint = grad_c.adjoint(out_direct)

        #print("GradientOperator, 4D, bnd_cond='Neumann', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("GradientOperator, 4D, bnd_cond='Neumann', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())


    def test_GradientOperator_4D_allocate(self):

        nc, nz, ny, nx = 3, 4, 5, 6
        size = nc * nz * ny * nx
        dim = [nc, nz, ny, nx]

        ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz, channels=nc)

        arr = numpy.arange(size).reshape(dim).astype(numpy.float32)**2

        data = ig.allocate()
        data.fill(arr)

        #numpy
        grad1 = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        gold_direct = grad1.direct(data)
        gold_adjoint = grad1.adjoint(gold_direct)

        grad2 = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        out_direct = grad2.range_geometry().allocate()
        out_adjoint = grad2.domain_geometry().allocate()
        grad2.direct(data, out=out_direct)
        grad2.adjoint(out_direct, out=out_adjoint)

        #print("GradientOperatorOperator, 4D, bnd_cond='Neumann', direct")
        numpy.testing.assert_array_equal(out_direct.get_item(0).as_array(), gold_direct.get_item(0).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(1).as_array(), gold_direct.get_item(1).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(2).as_array(), gold_direct.get_item(2).as_array())
        numpy.testing.assert_array_equal(out_direct.get_item(3).as_array(), gold_direct.get_item(3).as_array())

        #print("GradientOperator, 4D, bnd_cond='Neumann', adjoint")
        numpy.testing.assert_array_equal(out_adjoint.as_array(), gold_adjoint.as_array())

        #c
        grad1 = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
        gold_direct = grad1.direct(data)
        gold_adjoint = grad1.adjoint(gold_direct)

        grad2 = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')
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

    def test_GradientOperator_linearity(self):

        nc, nz, ny, nx = 3, 4, 5, 6
        ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz, channels=nc)

        grad = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='c')   
        self.assertTrue(LinearOperator.dot_test(grad))

        grad = GradientOperator(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='c')
        self.assertTrue(LinearOperator.dot_test(grad))

        grad = GradientOperator(ig, bnd_cond='Neumann', correlation='SpaceChannels', backend='numpy')
        self.assertTrue(LinearOperator.dot_test(grad))

        grad = GradientOperator(ig, bnd_cond='Periodic', correlation='SpaceChannels', backend='numpy')
        self.assertTrue(LinearOperator.dot_test(grad))
        
    def test_Gradient_c_numpy_voxel(self):
        
        numpy.random.seed(5)
        
        print("Test GradientOperator for 2D Geometry, ")
        
        ny, nx, nz = 3, 4, 5
        ig = ImageGeometry(voxel_num_y = ny, voxel_num_x = nx, voxel_size_x=0.1, voxel_size_y=0.5) 
            

        GD_C = GradientOperator(ig, backend = 'c')
        GD_numpy = GradientOperator(ig, backend = 'numpy')

        id = ig.allocate('random')
        direct_c = GD_C.direct(id)
        direct_numpy = GD_numpy.direct(id)
        numpy.testing.assert_allclose(direct_c[0].array, direct_numpy[0].array, atol=0.1) 
        numpy.testing.assert_allclose(direct_c[1].array, direct_numpy[1].array, atol=0.1) 

        direct_c *=0
        direct_numpy *=0
        GD_C.direct(id, out=direct_c)
        GD_numpy.direct(id, out=direct_numpy)
        numpy.testing.assert_allclose(direct_c[0].array, direct_numpy[0].array, atol=0.1) 
        numpy.testing.assert_allclose(direct_c[1].array, direct_numpy[1].array, atol=0.1) 

        adjoint_c = GD_C.adjoint(direct_c)
        adjoint_numpy = GD_numpy.adjoint(direct_numpy)
        numpy.testing.assert_allclose(adjoint_c.array, adjoint_numpy.array, atol=0.1) 
        numpy.testing.assert_allclose(adjoint_c.array, adjoint_numpy.array, atol=0.1) 

        adjoint_c *=0
        adjoint_numpy *=0
        GD_C.adjoint(direct_c, out=adjoint_c)
        GD_numpy.adjoint(direct_numpy, out=adjoint_numpy)
        numpy.testing.assert_allclose(adjoint_c.array, adjoint_numpy.array, atol=0.1) 
        numpy.testing.assert_allclose(adjoint_c.array, adjoint_numpy.array, atol=0.1) 


        
        print("Check Gradient_C, Gradient_numpy norms")
        Gradient_C_norm = GD_C.norm()
        Gradient_numpy_norm = GD_numpy.norm()   
        print(Gradient_C_norm, Gradient_numpy_norm)
        numpy.testing.assert_allclose(Gradient_C_norm, Gradient_numpy_norm, rtol=0.1) 
        numpy.testing.assert_allclose(numpy.sqrt((2/ig.voxel_size_x)**2 + (2/ig.voxel_size_y)**2), Gradient_numpy_norm, rtol=0.1) 
        numpy.testing.assert_allclose(numpy.sqrt((2/ig.voxel_size_x)**2 + (2/ig.voxel_size_y)**2), Gradient_C_norm, rtol=0.1) 
        print("Test passed\n")
        
        print("Check dot test")
        self.assertTrue(GD_C.dot_test(GD_C))
        self.assertTrue(GD_numpy.dot_test(GD_numpy))
        print("Test passed\n")
            
        print("Check dot test for Gradient Numpy with different method/bdn_cond")
        
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Neumann')    

        self.assertTrue(G_numpy1.dot_test(G_numpy1))
                
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
 
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))         
        
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Periodic')  
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        print("Test passed\n")
        
        print("Test GradientOperator for 2D Geometry passed\n")
        
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ###########################################################################
        
        print("Test GradientOperator for 3D Geometry, ")
        ig = ImageGeometry(voxel_num_y = ny, voxel_num_x = nx, voxel_num_z = nz, voxel_size_x=0.1, voxel_size_y=0.5, voxel_size_z = 0.4)  
        
        GD_C = GradientOperator(ig, backend = 'c')
        GD_numpy = GradientOperator(ig, backend = 'numpy')
        
        numpy.random.seed(5)
           
        print("Check Gradient_C, Gradient_numpy norms")
        Gradient_C_norm = GD_C.norm()
        Gradient_numpy_norm = GD_numpy.norm()    
        numpy.testing.assert_allclose(Gradient_C_norm, Gradient_numpy_norm, rtol=0.1) 
        numpy.testing.assert_allclose(numpy.sqrt((2/ig.voxel_size_z)**2 + (2/ig.voxel_size_x)**2 + (2/ig.voxel_size_y)**2), Gradient_numpy_norm, rtol=0.1) 
        numpy.testing.assert_allclose(numpy.sqrt((2/ig.voxel_size_z)**2 + (2/ig.voxel_size_x)**2 + (2/ig.voxel_size_y)**2), Gradient_C_norm, rtol=0.1) 
        print("Test passed\n")
        
        print("Check dot test")
        self.assertTrue(GD_C.dot_test(GD_C))
        self.assertTrue(GD_numpy.dot_test(GD_numpy))
        print("Test passed\n")
            
        print("Check dot test for GradientOperator Numpy with different method/bdn_cond")
            
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Neumann')  
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Neumann')
        self.assertTrue(G_numpy1.dot_test(G_numpy1)) 
        
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
       
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
      
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Periodic')       
        self.assertTrue(G_numpy1.dot_test(G_numpy1)) 
        
        print("Test passed\n")
        
        print("Test GradientOperator for 3D Geometry passed\n")
    
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ###########################################################################
        
        print("Test GradientOperator for 2D Geometry + channels, ")
        ig = ImageGeometry(5,10, voxel_size_x=0.1, voxel_size_y=0.5, channels = 10)  
        
        GD_C = GradientOperator(ig, backend = 'c')
        GD_numpy = GradientOperator(ig, backend = 'numpy')
                   
        print("Check Gradient_C, Gradient_numpy norms")
        Gradient_C_norm = GD_C.norm()
        Gradient_numpy_norm = GD_numpy.norm()    
        numpy.testing.assert_allclose(Gradient_C_norm, Gradient_numpy_norm, rtol=0.1) 
        print("Test passed\n")
        
        print("Check dot test")
        self.assertTrue(GD_C.dot_test(GD_C))
        self.assertTrue(GD_numpy.dot_test(GD_numpy))
        print("Test passed\n")
            
        print("Check dot test for GradientOperator Numpy with different method/bdn_cond")
            
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1)) 
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Neumann') 
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        print("Test passed\n")
        
        print("Test GradientOperator for 2D Geometry + channels passed\n")
        
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ###########################################################################
        
        print("Test GradientOperator for 3D Geometry + channels, ")
        ig = ImageGeometry(voxel_num_x = nx, voxel_num_y = ny, voxel_num_z=nz, voxel_size_x=0.1, voxel_size_y=0.5, voxel_size_z = 0.3, channels = 10)  
        
        GD_C = GradientOperator(ig, backend = 'c')
        GD_numpy = GradientOperator(ig, backend = 'numpy')
                   
        print("Check Gradient_C, Gradient_numpy norms")
        Gradient_C_norm = GD_C.norm()
        Gradient_numpy_norm = GD_numpy.norm()  
        numpy.testing.assert_allclose(Gradient_C_norm, Gradient_numpy_norm, rtol=0.1) 
        print("Test passed\n")
        
        print("Check dot test")
        self.assertTrue(GD_C.dot_test(GD_C))
        self.assertTrue(GD_numpy.dot_test(GD_numpy))
        print("Test passed\n")
            
        print("Check dot test for Gradient Numpy with different method/bdn_cond")
            
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Neumann')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1)) 
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Neumann') 
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        G_numpy1 = GradientOperator(ig, method = 'forward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))
        
        G_numpy1 = GradientOperator(ig, method = 'backward', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        G_numpy1 = GradientOperator(ig, method = 'centered', bnd_cond = 'Periodic')    
        self.assertTrue(G_numpy1.dot_test(G_numpy1))  
        
        print("Test passed\n")
        
        print("Test GradientOperator for 3D Geometry + channels passed\n")        

    def test_GradientOperator_split(self): 
        N, M, K = 2, 3, 4
        channels = 10
        
        numpy.random.seed(1)
        ig1 = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        ig2 = ImageGeometry(voxel_num_x = M, channels = channels)
        ig3 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels) 
        ig4 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K)

        data1 = ig1.allocate('random')
        data2 = ig2.allocate('random')
        data3 = ig3.allocate('random')
        data4 = ig4.allocate('random')


        G1 = GradientOperator(ig1, split=True, correlation='SpaceChannels')
        out1 = G1.direct(data1)
        ad1 = G1.adjoint(out1)

        G1_gold = GradientOperator(ig1, split=False, correlation='SpaceChannels')
        out1_gold = G1.direct(data1)
        ad1_gold = G1.adjoint(out1_gold)

        self.assertEquals(out1.shape, (2,1))
        self.assertEquals(out1_gold.shape, (2,1))
        numpy.testing.assert_allclose(ad1.array, ad1_gold.array, atol = 0.4)


        G2 = GradientOperator(ig2, split=True, correlation='SpaceChannels')
        out2 = G2.direct(data2)
        ad2 = G2.adjoint(out2)

        G2_gold = GradientOperator(ig2, split=False, correlation='SpaceChannels')
        out2_gold = G2_gold.direct(data2)
        ad2_gold = G2_gold.adjoint(out2_gold)

        self.assertEquals(out2.shape, (2,1))
        self.assertEquals(out2_gold.shape, (2,1))
        numpy.testing.assert_allclose(ad2.array, ad2_gold.array, atol = 0.4)
        numpy.testing.assert_allclose(out2[0].array, out2_gold[0].array, atol = 0.4)
        numpy.testing.assert_allclose(out2[1][0].array, out2_gold[1].array, atol = 0.4)


        G3 = GradientOperator(ig3, split=True, correlation='SpaceChannels')
        out3 = G3.direct(data3)
        ad3 = G3.adjoint(out3)

        G3_gold = GradientOperator(ig3, split=False, correlation='SpaceChannels')
        out3_gold = G3_gold.direct(data3)
        ad3_gold = G3_gold.adjoint(out3_gold)

        self.assertEquals(out3.shape, (2,1))
        self.assertEquals(out3_gold.shape, (3,1))
        numpy.testing.assert_allclose(ad3.array, ad3_gold.array, atol = 0.4)
        numpy.testing.assert_allclose(out3[0].array, out3_gold[0].array, atol = 0.4)
        numpy.testing.assert_allclose(out3[1][0].array, out3_gold[1].array, atol = 0.4)  
        numpy.testing.assert_allclose(out3[1][1].array, out3_gold[2].array, atol = 0.4)


        G4 = GradientOperator(ig4, split=True, correlation='SpaceChannels')
        out4 = G4.direct(data4)
        ad4 = G4.adjoint(out4)

        G4_gold = GradientOperator(ig4, split=False, correlation='SpaceChannels')
        out4_gold = G4_gold.direct(data4)
        ad4_gold = G4_gold.adjoint(out4_gold)

        self.assertEquals(out4.shape, (2,1))
        self.assertEquals(out4_gold.shape, (4,1))
        numpy.testing.assert_allclose(ad4.array, ad4_gold.array, atol = 0.4)
        numpy.testing.assert_allclose(out4[0].array, out4_gold[0].array, atol = 0.4)
        numpy.testing.assert_allclose(out4[1][0].array, out4_gold[1].array, atol = 0.4)  
        numpy.testing.assert_allclose(out4[1][1].array, out4_gold[2].array, atol = 0.4)        
        numpy.testing.assert_allclose(out4[1][2].array, out4_gold[3].array, atol = 0.4)        

        #non default order
        data4.reorder(['vertical','horizontal_x','channel','horizontal_y'])
        G4 = GradientOperator(data4.geometry, split=True, correlation='SpaceChannels')
        out4 = G4.direct(data4)
        ad4 = G4.adjoint(out4)

        G4_gold = GradientOperator(data4.geometry, split=False, correlation='SpaceChannels')
        out4_gold = G4_gold.direct(data4)
        ad4_gold = G4_gold.adjoint(out4_gold)

        self.assertEquals(out4.shape, (2,1))
        self.assertEquals(out4_gold.shape, (4,1))
        numpy.testing.assert_allclose(ad4.array, ad4_gold.array, atol = 0.4)
        numpy.testing.assert_allclose(out4[0].array, out4_gold[2].array, atol = 0.4)
        numpy.testing.assert_allclose(out4[1][0].array, out4_gold[0].array, atol = 0.4)  
        numpy.testing.assert_allclose(out4[1][1].array, out4_gold[1].array, atol = 0.4)        
        numpy.testing.assert_allclose(out4[1][2].array, out4_gold[3].array, atol = 0.4) 

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

        data1 = ig1.allocate('random')   
        data2 = ig2.allocate('random') 
        data3 = ig3.allocate('random')  

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
            self.assertTrue(LinearOperator.dot_test(Grad_numpy))

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
            self.assertTrue(LinearOperator.dot_test(Grad_c))        

            # test direct numpy vs direct c backends (with and without out)
            numpy.testing.assert_array_almost_equal(res5[0].as_array(), res1[0].as_array()) 
            numpy.testing.assert_array_almost_equal(res6[1].as_array(), res2[1].as_array())

            

