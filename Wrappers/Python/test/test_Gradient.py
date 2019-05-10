import unittest
import numpy
from ccpi.framework import ImageGeometry, AcquisitionGeometry
from ccpi.framework import ImageData, AcquisitionData
#from ccpi.optimisation.algorithms import GradientDescent
from ccpi.framework import BlockDataContainer
#from ccpi.optimisation.Algorithms import CGLS
import functools

from ccpi.optimisation.operators import Gradient, Identity, BlockOperator

class TestGradient(unittest.TestCase):
    def test_Gradient(self): 
        N, M, K = 20, 30, 40
        channels = 10
        
        # check range geometry, examples
        
        ig1 = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        ig2 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z = K) 
        ig3 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels) 
        ig4 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K) 
        
        G1 = Gradient(ig1, correlation = 'Space') 
        print(G1.range_geometry().shape, '2D no channels')
            
        G4 = Gradient(ig3, correlation = 'SpaceChannels')
        print(G4.range_geometry().shape, '2D with channels corr')
        G5 = Gradient(ig3, correlation = 'Space')
        print(G5.range_geometry().shape, '2D with channels no corr')
        
        G6 = Gradient(ig4, correlation = 'Space')
        print(G6.range_geometry().shape, '3D with channels no corr')
        G7 = Gradient(ig4, correlation = 'SpaceChannels')
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
        G_no_channel = Gradient(ig_channel, correlation = 'Space')
        G_channel = Gradient(ig_channel, correlation = 'SpaceChannels')
        
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
        G2D = Gradient(ig2D)
        res = G2D.direct(u4)  
        print(res[0].as_array())
        print(res[1].as_array())

        M, N = 20, 30
        ig = ImageGeometry(M, N)
        arr = ig.allocate('random_int' )
        
        # check direct of Gradient and sparse matrix
        G = Gradient(ig)
        norm1 = G.norm(iterations=300)
        print ("should be sqrt(8) {} {}".format(numpy.sqrt(8), norm1))
        numpy.testing.assert_almost_equal(norm1, numpy.sqrt(8), decimal=2)
        ig4 = ImageGeometry(M,N, channels=3)
        G4 = Gradient(ig4, correlation=Gradient.CORRELATION_SPACECHANNEL)
        norm4 = G4.norm(iterations=300)
        print ("should be sqrt(12) {} {}".format(numpy.sqrt(12), norm4))
        numpy.testing.assert_almost_equal(norm4, numpy.sqrt(12), decimal=2)
