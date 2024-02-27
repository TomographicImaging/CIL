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
from unittest.mock import Mock
from utils import initialise_tests
from cil.framework import ImageGeometry, BlockGeometry, VectorGeometry, BlockDataContainer, DataContainer
from cil.optimisation.operators import BlockOperator,\
    FiniteDifferenceOperator, SymmetrisedGradientOperator
import numpy
from timeit import default_timer as timer
from cil.optimisation.operators import GradientOperator, IdentityOperator,\
    DiagonalOperator, MaskOperator, ChannelwiseOperator, BlurringOperator
from cil.optimisation.operators import LinearOperator, MatrixOperator
import numpy   
from cil.optimisation.operators import SumOperator,  ZeroOperator, CompositionOperator, ProjectionMap

from cil.utilities import dataexample
import logging
from testclass import CCPiTestClass
import scipy

from cil.utilities.errors import InPlaceError



initialise_tests()

def dt(steps):
    return steps[-1] - steps[-2]


class TestOperator(CCPiTestClass):
  

    def test_MatrixOperator(self):        
        m = 30
        n = 20
        
        vg = VectorGeometry(n)
        
        Amat = numpy.random.randn(m, n)
        A = MatrixOperator(Amat)
        
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

        ## dot test for real and complex MatrixOperator
        self.assertTrue(A.dot_test(A))

        Amat = numpy.random.randn(m, n) + 1j*numpy.random.randn(m, n)
        A = MatrixOperator(Amat)  
        self.assertTrue(A.dot_test(A))      
        
        
        
    
    def test_ZeroOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate(3)
        out=ig.allocate(0)
        op1=ZeroOperator(ig)
        self.assertNumpyArrayEqual(op1.direct(img).array,out.array)
        self.assertNumpyArrayEqual(op1.adjoint(img).array,out.array)
        ig2 = ImageGeometry(10,15,30)
        out2=ig2.allocate(0)
        img2=ig2.allocate(5)
        op2=ZeroOperator(ig, ig2)
        self.assertNumpyArrayEqual(op2.direct(img).array,out2.array)
        self.assertNumpyArrayEqual(op2.adjoint(img2).array,out.array)
        self.assertEqual(op2.calculate_norm(),0)

    def test_ScaledOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        scalar = 0.5
        sid = scalar * IdentityOperator(ig)
        numpy.testing.assert_array_equal(scalar * img.as_array(), sid.direct(img).as_array())
    

    def test_DiagonalOperator(self):
        M = 3
        ig = ImageGeometry(M, M)
        x = ig.allocate('random',seed=100)
        diag = ig.allocate('random',seed=101)
        
        # Set up example DiagonalOperator
        D = DiagonalOperator(diag)
    
        # Apply direct and check whether result equals diag*x as expected.
        z = D.direct(x)
        numpy.testing.assert_array_equal(z.as_array(), (diag*x).as_array())
    
        # Apply adjoint and check whether results equals diag*(diag*x) as expected.
        y = D.adjoint(z)
        numpy.testing.assert_array_equal(y.as_array(), (diag*(diag*x)).as_array())

        # test norm of diagonal
        norm1 = D.norm()
        numpy.testing.assert_almost_equal(norm1, numpy.max(diag.array))
  

    def test_DiagonalOperator_complex(self):
        M = 3
        ig = ImageGeometry(M, M)
        x = ig.allocate('random',seed=100 ,dtype=numpy.complex64)
        diag = ig.allocate('random',seed=101 ,dtype=numpy.complex64)
        
        # Set up example DiagonalOperator
        D = DiagonalOperator(diag)
    
        # Apply direct and check whether result equals diag*x as expected.
        z = D.direct(x)
        numpy.testing.assert_array_equal(z.as_array(), (diag*x).as_array())
    
        # Apply adjoint and check whether results equals diag.conjugate()*(diag*x) as expected.
        y = D.adjoint(z)
        numpy.testing.assert_array_equal(y.as_array(), (diag.conjugate()*(diag*x)).as_array())

        # test norm of diagonal with complex value
        norm1 = D.norm()
        numpy.testing.assert_almost_equal(norm1, numpy.max(numpy.abs(diag.array)))        
        
    def test_MaskOperator(self):
        M = 3
        ig = ImageGeometry(M, M)
        x = ig.allocate('random',seed=100)
        
        mask = ig.allocate(True,dtype=bool)
        amask = mask.as_array()
        amask[2,1:3] = False
        amask[0,0] = False
        
        MO = MaskOperator(mask)
        
        # Apply direct and check whether result equals diag*x as expected.
        z = MO.direct(x)
        numpy.testing.assert_array_equal(z.as_array(), (mask*x).as_array())
    
        # Apply adjoint and check whether results equals diag*(diag*x) as expected.
        y = MO.adjoint(z)
        numpy.testing.assert_array_equal(y.as_array(), (mask*(mask*x)).as_array())
        

    def test_ChannelwiseOperator(self):
        M = 3
        channels = 4
        ig = ImageGeometry(M, M, channels=channels)
        igs = ImageGeometry(M, M)
        x = ig.allocate('random',seed=100)
        diag = igs.allocate('random',seed=101)
        
        D = DiagonalOperator(diag)
        C = ChannelwiseOperator(D,channels)
        
        y = C.direct(x)
        
        y2 = ig.allocate()
        C.direct(x,y2)
        
        for c in range(channels):
            numpy.testing.assert_array_equal(y.get_slice(channel=2).as_array(), \
                                             (diag*x.get_slice(channel=2)).as_array())
            numpy.testing.assert_array_equal(y2.get_slice(channel=2).as_array(), \
                                             (diag*x.get_slice(channel=2)).as_array())
        
        
        z = C.adjoint(y)
        
        z2 = ig.allocate()
        C.adjoint(y,z2)
        
        for c in range(channels):
            numpy.testing.assert_array_equal(z.get_slice(channel=2).as_array(), \
                                             (diag*(diag*x.get_slice(channel=2))).as_array())
            numpy.testing.assert_array_equal(z2.get_slice(channel=2).as_array(), \
                                             (diag*(diag*x.get_slice(channel=2))).as_array())
        
        
    def test_BlurringOperator(self):
        ig = ImageGeometry(100,100)
        
        # Parameters for point spread function PSF (size and std)
        ks          = 11; 
        ksigma      = 5.0
        
        # Create 1D PSF and 2D as outer product, then normalise.
        w           = numpy.exp(-numpy.arange(-(ks-1)/2,(ks-1)/2+1)**2/(2*ksigma**2))
        w.shape     = (ks,1)
        PSF         = w*numpy.transpose(w)
        PSF         = PSF/(PSF**2).sum()
        PSF         = PSF/PSF.sum()
        
        # Create blurring operator
        BOP = BlurringOperator(PSF,ig)
        
        # Run dot test to check validity of adjoint.
        self.assertTrue(BOP.dot_test(BOP))


    def test_IdentityOperator(self):
        ig = ImageGeometry(10,20,30)
        img = ig.allocate()
        # img.fill(numpy.ones((30,20,10)))
        self.assertTrue(img.shape == (30,20,10))
        #self.assertEqual(img.sum(), 2*float(10*20*30))
        self.assertEqual(img.sum(), 0.)
        Id = IdentityOperator(ig)
        y = Id.direct(img)
        numpy.testing.assert_array_equal(y.as_array(), img.as_array())


    def test_FiniteDifference(self):
        N, M = 2, 3
        numpy.random.seed(1)
        ig = ImageGeometry(N, M)
        Id = IdentityOperator(ig)

        FD = FiniteDifferenceOperator(ig, direction = 0, bnd_cond = 'Neumann')
        u = FD.domain_geometry().allocate('random')       
        
        res = FD.domain_geometry().allocate(ImageGeometry.RANDOM)
        FD.adjoint(u, out=res)
        w = FD.adjoint(u)

        self.assertNumpyArrayEqual(res.as_array(), w.as_array())
        
        res = Id.domain_geometry().allocate(ImageGeometry.RANDOM)
        Id.adjoint(u, out=res)
        w = Id.adjoint(u)

        self.assertNumpyArrayEqual(res.as_array(), w.as_array())
        self.assertNumpyArrayEqual(u.as_array(), w.as_array())

        G = GradientOperator(ig)

        u = G.range_geometry().allocate(ImageGeometry.RANDOM)
        res = G.domain_geometry().allocate()
        G.adjoint(u, out=res)
        w = G.adjoint(u)

        self.assertNumpyArrayEqual(res.as_array(), w.as_array())
        
        u = G.domain_geometry().allocate(ImageGeometry.RANDOM)
        res = G.range_geometry().allocate()
        G.direct(u, out=res)
        w = G.direct(u)
        self.assertBlockDataContainerEqual(res, w)
        
        # 2D       
        M, N = 2, 3
        ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N, voxel_size_x=0.1, voxel_size_y=0.4)
        x = ig.allocate('random')
                
        labels = ["horizontal_y", "horizontal_x"]
        
        for i, dir in enumerate(labels):
            FD1 = FiniteDifferenceOperator(ig, direction=i)    
            res1 = FD1.direct(x)
            res1b = FD1.adjoint(x)
                            
            FD2 = FiniteDifferenceOperator(ig, direction=labels[i])    
            res2 = FD2.direct(x)
            res2b = FD2.adjoint(x) 
            
            numpy.testing.assert_almost_equal(res1.as_array(), res2.as_array())
            numpy.testing.assert_almost_equal(res1b.as_array(), res2b.as_array())
            
        # 2D  + chan     
        M, N, K = 2,3,4
        ig1 = ImageGeometry(voxel_num_x=M, voxel_num_y=N, channels=K, voxel_size_x=0.1, voxel_size_y=0.4)
        x = ig1.allocate('random')
        
        labels = ["channel","horizontal_y", "horizontal_x"]
        
        for i, dir in enumerate(labels):
            FD1 = FiniteDifferenceOperator(ig1, direction=i)    
            res1 = FD1.direct(x)
            res1b = FD1.adjoint(x)    
            FD2 = FiniteDifferenceOperator(ig1, direction=labels[i])    
            res2 = FD2.direct(x)
            res2b = FD2.adjoint(x) 
            
            numpy.testing.assert_almost_equal(res1.as_array(), res2.as_array())
            numpy.testing.assert_almost_equal(res1b.as_array(), res2b.as_array()) 
            

    def test_PowerMethod(self):
        # 2x2 real matrix, dominant eigenvalue = 2
        M1 = numpy.array([[1,0],[1,2]], dtype=float)
        M1op = MatrixOperator(M1)
        res1 = M1op.PowerMethod(M1op,100)
        numpy.testing.assert_almost_equal(res1,2., decimal=4)

        res_scipy = scipy.linalg.eig(M1)
        numpy.testing.assert_almost_equal(res1,numpy.abs(res_scipy[0]).max(), decimal=4)

        # Test with the norm       
        res2 = M1op.norm()
        res1 = M1op.PowerMethod(M1op,100, method="composed_with_adjoint")
        numpy.testing.assert_almost_equal(res1,res2, decimal=4)


        # 2x3 real matrix, dominant eigenvalue = 4.711479432297657
        M1 = numpy.array([[1.,0.,3],[1,2.,3]])
        M1op = MatrixOperator(M1)
        res1 = M1op.PowerMethod(M1op,100)
        numpy.testing.assert_almost_equal(res1,4.711479432297657, decimal=4)

        res_scipy = scipy.linalg.eig(M1.T.conjugate()@M1)
        numpy.testing.assert_almost_equal(res1,numpy.sqrt(numpy.abs(res_scipy[0])).max(), decimal=4)        
        
        # 2x3 complex matrix, (real eigenvalues), dominant eigenvalue = 5.417602365823937
        M1 = numpy.array([[2,1j,0],[2j,5j,0]])
        M1op = MatrixOperator(M1)
        res1 = M1op.PowerMethod(M1op,100)
        numpy.testing.assert_almost_equal(res1,5.531859582980837, decimal=4) 
        res_scipy = scipy.linalg.eig(M1.T.conjugate()@M1)
        numpy.testing.assert_almost_equal(res1,numpy.sqrt(numpy.abs(res_scipy[0])).max(), decimal=4)        
                

        # 3x3 complex matrix, (real+complex eigenvalue), dominant eigenvalue = 3.1624439599276974
        M1 = numpy.array([[2,0,0],[1,2j,1j],[3, 3-1j,3]])
        M1op = MatrixOperator(M1)
        res1 = M1op.PowerMethod(M1op,150)
        numpy.testing.assert_almost_equal(res1, 3.1624439599276974, decimal=3) 
        res_scipy = scipy.linalg.eig(M1)
        numpy.testing.assert_almost_equal(res1,numpy.abs(res_scipy[0]).max(), decimal=4)        
                     
        # 2x2 non-diagonalisable nilpotent matrix
        M1=numpy.array([[0.,1.], [0.,0.]])
        M1op = MatrixOperator(M1)
        res1 = M1op.PowerMethod(M1op,5)
        numpy.testing.assert_almost_equal(res1,0, decimal=4) 
        res_scipy = scipy.linalg.eig(M1)
        numpy.testing.assert_almost_equal(res1,numpy.abs(res_scipy[0]).max(), decimal=4)         

        # 2x2 non-diagonalisable nilpotent matrix where method="composed_with_adjoint"
        M1=numpy.array([[0.,1.], [0.,0.]])
        M1op = MatrixOperator(M1)
        res1 = M1op.PowerMethod(M1op,5, method="composed_with_adjoint")
        numpy.testing.assert_almost_equal(res1,1, decimal=4)
        res_scipy = scipy.linalg.eig(M1.T@M1)
        numpy.testing.assert_almost_equal(res1,numpy.abs(res_scipy[0]).max(), decimal=4)          


        # 2x2 matrix, max absolute eigenvalue is not unique and initial vector chosen for non-convergence
        
        M1=numpy.array([[2.,1.], [0.,-2.]])
        M1op = MatrixOperator(M1)
        _,_,_,_,convergence = M1op.PowerMethod(M1op,100, initial=DataContainer(numpy.array([1.,1.])), return_all=True)
        numpy.testing.assert_equal(convergence,False) 

        # 2x2 matrix, max absolute eigenvalue is not unique and initial vector chosen for convergence
        
        M1=numpy.array([[2.,1.,0.],[0.,1.,1.], [0.,0.,1.]])
        M1op = MatrixOperator(M1)
        res1,_,_,_,convergence = M1op.PowerMethod(M1op,100, return_all=True)
        numpy.testing.assert_almost_equal(res1,2., decimal=4) 
        numpy.testing.assert_equal(convergence,True)     

        # Gradient Operator (float)
        ig = ImageGeometry(30,30)
        Grad = GradientOperator(ig)
        res1 = Grad.PowerMethod(Grad,500, tolerance=1e-6)
        numpy.testing.assert_almost_equal(res1, numpy.sqrt(8), decimal=2) 

        # Gradient Operator (complex)
        ig = ImageGeometry(30,30, dtype=complex)
        Grad = GradientOperator(ig, backend='numpy')
        res1 = Grad.PowerMethod(Grad,500, tolerance=1e-6)
        numpy.testing.assert_almost_equal(res1, numpy.sqrt(8), decimal=2)         
                    
        # Identity Operator
        Id = IdentityOperator(ig)
        res1 = Id.PowerMethod(Id,100)
        numpy.testing.assert_almost_equal(res1,1.0, decimal=4)        

        # Test errors produced if not a valid method
        try:
            res1 = Id.PowerMethod(Id,100, method='gobledigook')
        except ValueError:
            pass      


    def test_Norm(self):
        numpy.random.seed(1)
        N, M = 200, 300

        ig = ImageGeometry(N, M)
        G = GradientOperator(ig)

        self.assertIsNone(G._norm)
        
        #calculates norm
        self.assertAlmostEqual(G.norm(), numpy.sqrt(8), 2)

        #sets_norm
        G.set_norm(4)
        self.assertEqual(G._norm, 4)

        #gets cached norm
        self.assertEqual(G.norm(), 4)

        #sets cache to None
        G.set_norm(None)
        #recalculates norm
        self.assertAlmostEqual(G.norm(), numpy.sqrt(8), 2)


  
        #Check that the provided element is a number or None 
        with self.assertRaises(TypeError):
            G.set_norm['Banana']
        #Check that the provided norm is positive 
        with self.assertRaises(ValueError):
            G.set_norm(-1)

         # 2x2 real matrix, dominant eigenvalue = 2. Check norm uses the right flag for power method 
        M1 = numpy.array([[1,0],[1,2]], dtype=float)
        M1op = MatrixOperator(M1)
        res1 = M1op.norm()
        res2 = M1op.PowerMethod(M1op,100)
        res3 = M1op.PowerMethod(M1op,100, method="composed_with_adjoint")
        res4 = M1op.PowerMethod(M1op,100, method="direct_only")
        numpy.testing.assert_almost_equal(res1,res3, decimal=4)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res1,res4)



    def test_ProjectionMap(self):
        # Check if direct is correct
        ig1 = ImageGeometry(3,4)
        ig2 = ImageGeometry(5,6)
        ig3 = ImageGeometry(5,6,4)

        # Create BlockGeometry
        bg = BlockGeometry(ig1,ig2, ig3)
        x = bg.allocate(10)

        # Extract containers
        x0, x1, x2 = x[0], x[1], x[2]

        for i in range(3):
            proj_map = ProjectionMap(bg, i)
            # res1 is in ImageData from the X_{i} "ImageGeometry"
            res1 = proj_map.direct(x)

            # res2 is in ImageData from the X_{i} "ImageGeometry" using out
            res2 = bg.geometries[i].allocate(0)
            proj_map.direct(x, out=res2)

            # Check with and without out
            numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array())

            # Depending on which index is used, check if x0, x1, x2 are the same with res2
            if i==0:            
                numpy.testing.assert_array_almost_equal(x0.as_array(), res2.as_array())
            elif i==1: 
                numpy.testing.assert_array_almost_equal(x1.as_array(), res2.as_array())   
            elif i==2:
                numpy.testing.assert_array_almost_equal(x2.as_array(), res2.as_array())  
            else:
                pass      

        # Check if adjoint is correct

        bg = BlockGeometry(ig1, ig2, ig3, ig1, ig2, ig3)
        x = ig1.allocate(20)

        index=3
        proj_map = ProjectionMap(bg, index)

        res1 = bg.allocate(0)
        proj_map.adjoint(x, out=res1)

        # check if all indices return arrays filled with 0, except the input index

        for i in range(len(bg.geometries)):
            if i!=index:
                numpy.testing.assert_array_almost_equal(res1[i].as_array(), bg.geometries[i].allocate().as_array())   

        # Check error messages
        # Check if index is correct wrt length of Cartesian Product
        with self.assertRaises(ValueError):
            ig = ImageGeometry(3,4)
            bg = BlockGeometry(ig,ig)
            index = 3
            proj_map = ProjectionMap(bg, index)
        
        # Check error if an ImageGeometry is passed
        with self.assertRaises(ValueError):
            proj_map = ProjectionMap(ig, index)                
            
    
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


    def test_SymmetrisedGradientOperator1a(self):
        ###########################################################################  
        ## Symmetrized Gradient Tests
        ###########################################################################
        # 2D geometry no channels
        # ig = ImageGeometry(N, M)
        Grad = GradientOperator(self.ig)
        
        E1 = SymmetrisedGradientOperator(Grad.range_geometry())
        norm = LinearOperator.PowerMethod(E1, max_iteration=self.iterations)
        numpy.testing.assert_almost_equal(norm, numpy.sqrt(8), decimal = self.decimal)
        

    def test_SymmetrisedGradientOperator1b(self):
        ###########################################################################  
        ## Symmetrized GradientOperator Tests
        ###########################################################################
        # 2D geometry no channels
        # ig = ImageGeometry(N, M)
        Grad = GradientOperator(self.ig)
        
        E1 = SymmetrisedGradientOperator(Grad.range_geometry())
        numpy.random.seed(1)
        u1 = E1.domain_geometry().allocate('random')
        w1 = E1.range_geometry().allocate('random', symmetry = True)
                
        lhs = E1.direct(u1).dot(w1)
        rhs = u1.dot(E1.adjoint(w1))
        # self.assertAlmostEqual(lhs, rhs)
        numpy.testing.assert_allclose(lhs, rhs, rtol=1e-3)


    def test_SymmetrisedGradientOperator2(self):        
        ###########################################################################
        # 2D geometry with channels
        # ig2 = ImageGeometry(N, M, channels = C)
        Grad2 = GradientOperator(self.ig2, correlation = 'Space', backend='numpy')
        
        E2 = SymmetrisedGradientOperator(Grad2.range_geometry())
        numpy.random.seed(1)
        u2 = E2.domain_geometry().allocate('random')
        w2 = E2.range_geometry().allocate('random', symmetry = True)
    #    
        lhs2 = E2.direct(u2).dot(w2)
        rhs2 = u2.dot(E2.adjoint(w2))
            
        numpy.testing.assert_allclose(lhs2, rhs2, rtol=1e-3)
        

    def test_SymmetrisedGradientOperator2a(self):        
        ###########################################################################
        # 2D geometry with channels
        # ig2 = ImageGeometry(N, M, channels = C)
        Grad2 = GradientOperator(self.ig2, correlation = 'Space', backend='numpy')
        
        E2 = SymmetrisedGradientOperator(Grad2.range_geometry())
        norm = LinearOperator.PowerMethod(E2, max_iteration=self.iterations)
        numpy.testing.assert_almost_equal(norm, 
           numpy.sqrt(8), decimal = self.decimal)
        
    
    def test_SymmetrisedGradientOperator3a(self):
        ###########################################################################
        # 3D geometry no channels
        #ig3 = ImageGeometry(N, M, K)
        Grad3 = GradientOperator(self.ig3, correlation = 'Space')
        
        E3 = SymmetrisedGradientOperator(Grad3.range_geometry())

        norm = LinearOperator.PowerMethod(E3,max_iteration=100, tolerance = 0)
        numpy.testing.assert_almost_equal(norm, numpy.sqrt(12), decimal = self.decimal)
        

    def test_SymmetrisedGradientOperator3b(self):
        ###########################################################################
        # 3D geometry no channels
        #ig3 = ImageGeometry(N, M, K)
        Grad3 = GradientOperator(self.ig3, correlation = 'Space')
        
        E3 = SymmetrisedGradientOperator(Grad3.range_geometry())
        numpy.random.seed(1)
        u3 = E3.domain_geometry().allocate('random')
        w3 = E3.range_geometry().allocate('random', symmetry = True)
    #    
        lhs3 = E3.direct(u3).dot(w3)
        rhs3 = u3.dot(E3.adjoint(w3))
        
        numpy.testing.assert_almost_equal(lhs3, rhs3, decimal=3)  
        self.assertTrue( LinearOperator.dot_test(E3, range_init = w3, domain_init=u3, decimal=3) )


    def test_dot_test(self):
        Grad3 = GradientOperator(self.ig3, correlation = 'Space', backend='numpy')
             
        # self.assertAlmostEqual(lhs3, rhs3)
        self.assertTrue( LinearOperator.dot_test(Grad3 , verbose=True, decimal=4))
        self.assertTrue( LinearOperator.dot_test(Grad3 , verbose=True, decimal=4))


    def test_dot_test2(self):
        Grad3 = GradientOperator(self.ig3, correlation = 'SpaceChannel', backend='c')
             
        # self.assertAlmostEqual(lhs3, rhs3)
        # self.assertTrue( LinearOperator.dot_test(Grad3 , verbose=True))
        self.assertTrue( LinearOperator.dot_test(Grad3 , decimal=4, verbose=True))


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


class TestOperatorCompositionSum(unittest.TestCase):
    def setUp(self):        
        self.data = dataexample.BOAT.get(size=(128,128))
        self.ig = self.data.geometry


    def test_SumOperator(self):
        # numpy.random.seed(1)
        ig = self.ig
        data = self.data

        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        c = SumOperator(Id1,Id2)
        out = c.direct(data)

        numpy.testing.assert_array_almost_equal(out.as_array(),3 * data.as_array())


    def test_CompositionOperator_direct1(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        d = CompositionOperator(G, Id2)

        out1 = G.direct(data)
        out2 = d.direct(data)

        numpy.testing.assert_array_almost_equal(out2.get_item(0).as_array(),  out1.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(out2.get_item(1).as_array(),  out1.get_item(1).as_array())


    def test_CompositionOperator_direct2(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        

        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
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
        G = GradientOperator(domain_geometry=ig)
        

        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
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


    def test_CompositionOperator_direct4(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        d = CompositionOperator(G, Id1, Id2)

        out1 = G.direct(data)
        
        d_out = d.direct(data)

        numpy.testing.assert_array_almost_equal(d_out.get_item(0).as_array(),
                                                2 * out1.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(d_out.get_item(1).as_array(),
                                                2 * out1.get_item(1).as_array())


    def test_CompositionOperator_adjoint1(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        d = CompositionOperator(G, Id2)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  out1.as_array())
        

    def test_CompositionOperator_adjoint2(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        d = CompositionOperator(G, Id1)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  2 * out1.as_array())
    

    def test_CompositionOperator_adjoint3(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)    

        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        d = G.compose(Id1)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  2 * out1.as_array())


    def test_CompositionOperator_adjoint4(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        

        Id1 = 2 * IdentityOperator(ig)
        
        d = G.compose(-Id1)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  - 2 * out1.as_array())


    def test_CompositionOperator_adjoint5(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        Id1 = 3 * IdentityOperator(ig)
        Id = Id1 - IdentityOperator(ig)
        d = G.compose(Id)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  2 * out1.as_array())
    

    def test_CompositionOperator_adjoint6(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        

        Id1 = 3 * IdentityOperator(ig)
        Id = ZeroOperator(ig)
        d = G.compose(Id)
        da = d.direct(data)
        
        out1 = G.adjoint(da)
        out2 = d.adjoint(da)

        numpy.testing.assert_array_almost_equal(out2.as_array(),  0 * out1.as_array())


    def stest_CompositionOperator_direct4(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        sym = SymmetrisedGradientOperator(domain_geometry=ig)
        Id2 = IdentityOperator(ig)
        
        d = CompositionOperator(sym, Id2)

        out1 = G.direct(data)
        out2 = d.direct(data)

        numpy.testing.assert_array_almost_equal(out2.get_item(0).as_array(),  out1.get_item(0).as_array())
        numpy.testing.assert_array_almost_equal(out2.get_item(1).as_array(),  out1.get_item(1).as_array())


    def test_CompositionOperator_adjoint7(self):
        ig = self.ig
        data = self.data
        G = GradientOperator(domain_geometry=ig)
        
        Id1 = 2 * IdentityOperator(ig)
        Id2 = IdentityOperator(ig)
        
        d = CompositionOperator(G, Id1, Id2)

        out1 = G.direct(data)
        out2 = G.adjoint(out1)
        
        d_out = d.adjoint(out1)

        numpy.testing.assert_array_almost_equal(d_out.as_array(),
                                                2 * out2.as_array())
        numpy.testing.assert_array_almost_equal(d_out.as_array(),
                                                2 * out2.as_array())


    def test_CompositionOperator_norm1(self):
   
        M1 = numpy.array([[1,0],[1,2]], dtype=float)
        M1op = MatrixOperator(M1)
        
        M2=numpy.array([[3,1],[0,1]], dtype=float)
        M2op = MatrixOperator(M2)


        d = CompositionOperator(M1op, M2op)

        self.assertAlmostEqual(M2op.norm(), 3.1795867966025404,places=4)
        self.assertAlmostEqual(M1op.norm(), 2.288245580264566, places=4)
        self.assertAlmostEqual(d.norm(), 5.162277659459584, places=4)

    def test_CompositionOperator_norm2(self):
   
        M1 = numpy.array([[4,3],[1,1]], dtype=float)
        M1op = MatrixOperator(M1)
        
        M2=numpy.array([[1,-3],[-1,4]], dtype=float)
        M2op = MatrixOperator(M2)


        d = CompositionOperator(M1op, M2op)

        self.assertAlmostEqual(d.norm(), 1, places=4)
        

    def test_CompositionOperator_norm3(self):
        ig = self.ig
        M1 = numpy.array([[4,3],[1,1]], dtype=float)
        M1op = MatrixOperator(M1)
        
        M2=numpy.array([[1,-3],[-1,4]], dtype=float)
        M2op = MatrixOperator(M2)

        M3=numpy.array([[1,-5],[-1,6]], dtype=float)
        M3op = MatrixOperator(M2)

        d = CompositionOperator(M1op, M2op, M3op)

        out1 = M3op.norm()
        out2 = d.norm()

        numpy.testing.assert_almost_equal(out2, out1)


    def test_CompositionOperator_norm4(self):

        M1 = numpy.array([[4,0],[0,1]], dtype=float)
        M1op = MatrixOperator(M1)

        d = CompositionOperator(M1op)

        out1 = 4
        out2 = d.norm()

        numpy.testing.assert_almost_equal(out2, out1)
