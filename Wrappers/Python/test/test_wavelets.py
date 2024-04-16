import unittest
import numpy as np
import warnings
import pywt

from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.functions import L1Norm, L1Sparsity
from cil.optimisation.operators import WaveletOperator

from testclass import CCPiTestClass

def setUp(self):
    np.random.seed(1)
    print("Seed set")

class TestWavelets(CCPiTestClass):
    
    def test_wavelet_operator_init(self):
        m, n=11,13
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n)
        W = WaveletOperator(dg)
        self.assertEqual(W.wname, 'haar')
        self.assertEqual(pywt.Wavelet(W.wname).filter_bank, W._wavelet.filter_bank)
        self.assertEqual(W.correlation, None)
        self.assertEqual(W.bnd_cond, 'symmetric')
        self.assertEqual(W._trueAdj, True )
        self.assertEqual(W.level,int(np.log2(min(n,m))))
        self.assertNumpyArrayEqual(np.array((n,m)), W.domain_geometry().shape)    
                
        #Check is_linear
        self.assertTrue(W.is_linear())
        
        #Check is_orthogonal
        self.assertEqual(W.is_orthogonal(), W._wavelet.orthogonal)
    
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n)
        rg = ImageGeometry(voxel_num_x=n, voxel_num_y=m)
        with self.assertRaises(AttributeError):
            W = WaveletOperator(dg, rg)
        
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n)
        
        with self.assertWarnsRegex(UserWarning, "Level value of 100 is too high"):
            WaveletOperator(dg, level=100)
        
                
        W = WaveletOperator(dg, dg, wname='db2', level=0, axes=[0], bnd_cond='zero', true_adjoint=False)
        self.assertEqual(W.wname, 'db2')
        self.assertEqual(pywt.Wavelet(W.wname).filter_bank, W._wavelet.filter_bank)
        self.assertEqual(W.correlation, None)
        self.assertEqual(W.bnd_cond, 'zero')
        self.assertEqual(W._trueAdj, False)
        self.assertEqual(W.level,0)
        self.assertNumpyArrayEqual(np.array((n,m)), W.domain_geometry().shape)
        
        
    def test_wavelet_norm_init(self):
        m, n=11,13
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n)
        W = WaveletOperator(dg)
        Wnorm=L1Sparsity(W)
        from cil.optimisation.functions.L1Norm import _L1Norm, _WeightedL1Norm
        self.assertTrue(issubclass(Wnorm.l1norm.function.__class__, _L1Norm))
        
        Wnorm=L1Sparsity(W, weight=np.ones((13,11)))
        self.assertTrue(issubclass(Wnorm.l1norm.function.__class__, _WeightedL1Norm))
        self.assertNumpyArrayEqual(Wnorm.l1norm.function.weight, np.ones((13,11)))
    

    def test_WaveletOperator_dimensions(self):
        m, n = 42, 47
        level = 2
        for wname in ['haar', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'coif2', 'coif3']:
            for dg in [ImageGeometry(voxel_num_x=m, voxel_num_y=n), VectorGeometry(m), VectorGeometry(n)]:
                with self.subTest(msg=f"{wname} transform failed for {dg.__class__.__name__} of size {dg.shape}", wname=wname, dg=dg):
                    x = dg.allocate('random')

                    W = WaveletOperator(dg, wname=wname, level=level)
                    rg = W.range_geometry() # Range
                    Wx = W.direct(x)
                    y = W.adjoint(Wx)

                    self.assertEqual(Wx.shape, rg.shape, msg="Coefficient array shape should match range_geometry")
                    self.assertEqual(y.shape, dg.shape, msg="Adjoint array shape should match domain_geometry")

                    # Reconstruction should be (almost) the original input
                    self.assertNumpyArrayAlmostEqual(x.as_array(), y.as_array())

                    self.assertRaises(AttributeError, WaveletOperator, domain_geometry=dg, range_geometry=dg, wname=wname, msg="Geometry shape mismatch should raise error")
    
    def test_WaveletOperator_complex_input(self):
        m, n = 42, 47
        level = 2
        for wname in ['haar', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'coif2', 'coif3']:
            for dg in [ImageGeometry(voxel_num_x=m, voxel_num_y=n), VectorGeometry(m), VectorGeometry(n)]:
                with self.subTest(msg=f"{wname} transform failed for {dg.__class__.__name__} of size {dg.shape}", wname=wname, dg=dg):
                    x = dg.allocate('random', dtype='complex128')

                    W = WaveletOperator(dg, wname=wname, level=level)
                    rg = W.range_geometry() # Range
                    Wx = W.direct(x)
                    y = W.adjoint(Wx)

                    self.assertEqual(x.dtype, Wx.dtype, msg="Complex input should give complex coefficients")
                    self.assertEqual(Wx.dtype, y.dtype, msg="Complex coefficients should give complex output")

                    self.assertEqual(Wx.shape, rg.shape, msg="Coefficient array shape should match range_geometry")
                    self.assertEqual(y.shape, dg.shape, msg="Adjoint array shape should match domain_geometry")

                    # Reconstruction should be (almost) the original input
                    self.assertNumpyArrayAlmostEqual(x.as_array(), y.as_array())

                    z = 0*Wx
                    W.direct(x, out=z)
                    self.assertNumpyArrayAlmostEqual(Wx.as_array(), z.as_array())
    
    def test_WaveletOperator_dimensions_biorthogonal(self):
        m, n = 48, 47
        level = 2
        for wname in ["bior3.5", "bior3.3", "rbio3.5", "rbio4.4"]:
            for dg in [ImageGeometry(voxel_num_x=m, voxel_num_y=n), VectorGeometry(m), VectorGeometry(n)]:
                with self.subTest(msg=f"{wname} transform failed for {dg.__class__.__name__} of size {dg.shape}", wname=wname, dg=dg):
                    x = dg.allocate('random')

                    W = WaveletOperator(dg, wname=wname, level=level)
                    rg = W.range_geometry() # Range
                    Wx = W.direct(x)
                    y = W.adjoint(Wx)

                    self.assertEqual(Wx.shape, rg.shape, msg="Coefficient array shape should match range_geometry")
                    self.assertEqual(y.shape, dg.shape, msg="Adjoint array shape should match domain_geometry")
                    
                    #Check is_not_orthogonal
                    self.assertEqual(W.is_orthogonal(), False)

                    # Reconstruction should be imperfect
                    self.assertLess(0, (x - y).norm(), msg="Biorthogonal wavelets with true adjoint should no longer produce perfect reconstructions")

                    self.assertRaises(AttributeError, WaveletOperator, domain_geometry=dg, range_geometry=dg, wname=wname, msg="Geometry shape mismatch should raise error")
    
    def test_WaveletOperator_biorthogonal_reconstruction(self):
        m, n = 48, 49
        level = 2
        for wname in ["bior3.5", "bior3.3", "rbio3.5", "rbio4.4"]:
            for dg in [ImageGeometry(voxel_num_x=m, voxel_num_y=n), VectorGeometry(m), VectorGeometry(n)]:
                with self.subTest(msg=f"{wname} transform failed for {dg.__class__.__name__} of size {dg.shape}", wname=wname, dg=dg):
                    x = dg.allocate('random')

                    W = WaveletOperator(dg, wname=wname, level=level, true_adjoint=False)
                    rg = W.range_geometry() # Range
                    Wx = W.direct(x)
                    y = W.adjoint(Wx)

                    self.assertEqual(Wx.shape, rg.shape, msg="Coefficient array shape should match range_geometry")
                    self.assertEqual(y.shape, dg.shape, msg="Adjoint array shape should match domain_geometry")

                    # Reconstruction should be (almost) the original input
                    self.assertNumpyArrayAlmostEqual(x.as_array(), y.as_array())

                    self.assertRaises(AttributeError, WaveletOperator, domain_geometry=dg, range_geometry=dg, wname=wname, msg="Geometry shape mismatch should raise error")

    def test_wavelet_axes(self):
        m, n = 20, 21
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n) # Domain
        dg.channels = 32

        # Give axes as range
        W = WaveletOperator(dg, wname='db2', level=2, axes=range(1,3))
        rg = W.range_geometry()

        self.assertEqual(dg.shape[0], rg.shape[0], msg="Size of channels should remain unchanged")
        self.assertLess(dg.shape[1:], rg.shape[1:], msg="Decomposed dimensions should get extended")

        # Try each individual axis
        for ax in range(3):
            with self.subTest(msg=f"Failed for axis {ax}", ax=ax):
                W = WaveletOperator(dg, wname='db2', level=2, axes=[ax])
                rg = W.range_geometry()

                self.assertEqual(dg.shape[not ax], rg.shape[not ax], msg="Size of other dimensions should remain unchanged")
                self.assertEqual(dg.shape[ax] + 5, rg.shape[ax], msg="Decomposed dimension should get extended by 5 (when level=2)")

    def test_wavelet_correlation(self):
        m, n = 20, 21
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n) # Domain
        
        self.assertRaises(AttributeError, WaveletOperator, dg, wname='haar', correlation='channel', msg="Missing channels should raise error")

        dg.channels = 32
        self.assertRaises(AttributeError, WaveletOperator, dg, wname='haar', correlation='SPACEchannel', msg="Invalid correlation keyword should raise error")

        W = WaveletOperator(dg, wname='db2', level=2, correlation='Space')
        rg = W.range_geometry()
        self.assertEqual(dg.shape[0], rg.shape[0], msg="Size of channels should remain unchanged")
        self.assertLess(dg.shape[1:], rg.shape[1:], msg="Correlated dimensions should get extended")

        self.assertWarns(UserWarning, WaveletOperator, dg, axes=[0], correlation='channel', msg="Defining both axes and correlation should give warning")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = WaveletOperator(dg, wname='db2', level=2, axes=[1], correlation='channel')
        rg = W.range_geometry()

        self.assertEqual(dg.shape[0], rg.shape[0], msg="Axes should be prioritized over correlation")

    def test_wavelet_bnd_cond(self):
        m, n = 20, 24
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n) # Domain
        W = WaveletOperator(dg, wname='db2', level=2, bnd_cond='periodization') # Note: this is different from bnd_cond='periodic'
        rg = W.range_geometry()

        self.assertEqual(dg.shape, rg.shape, msg="Periodization convolution should require no padding")

        x = dg.allocate('random')
        c = rg.allocate('random')

        ip1 = c.dot(W.direct(x))
        ip2 = x.dot(W.adjoint(c))
        M = x.norm() # Normalization
        self.assertAlmostEqual(ip1/M, ip2/M, places=5, msg="Periodic convolution should be closest to true adjoint")

    def test_wavelet_adjoint(self):
        m, n = 48, 64
        dg = ImageGeometry(voxel_num_x=m, voxel_num_y=n) # Domain
        x = dg.allocate('random')
        for wname in ['haar', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'coif2', 'coif3', 'bior3.5', 'bior3.3', 'rbio3.5', 'rbio3.3']:
            with self.subTest(msg=f"Failed for wavelet {wname}", wname=wname):
                W = WaveletOperator(dg, wname=wname, level=2, bnd_cond='periodization') # Note: this is different from bnd_cond='periodic'
                rg = W.range_geometry()
                c = rg.allocate('random')

                ip1 = c.dot(W.direct(x))
                ip2 = x.dot(W.adjoint(c))
                M = x.norm() # Normalization
                self.assertAlmostEqual(ip1/M, ip2/M, places=5, msg="Periodization convolution should be closest to true adjoint")
    
    def test_WaveletOperator_norm(self):
        n = 64
        dg = VectorGeometry(n)
        for wname in ['haar', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'coif2', 'coif3']:
            with self.subTest(msg=f"Failed for wavelet {wname}", wname=wname):
                W = WaveletOperator(dg, wname=wname, level=1)
                self.assertEqual(W.norm(), 1.0, msg="Orthogonal wavelet transform should have unit norm")
        
        for wname in ['bior3.5', 'bior3.3', 'rbio3.5', 'rbio3.3']:
            with self.subTest(msg=f"Failed for wavelet {wname}", wname=wname):
                W = WaveletOperator(dg, wname=wname, level=2, bnd_cond='periodization') # Need to set bnd_cond for most faithful adjoint
                self.assertLess(0.9, W.norm())

    def test_L1Sparsity(self):
        n = 20
        wname = 'db2'
        level = 2

        dg = VectorGeometry(n)
        W = WaveletOperator(dg, wname=wname, level=level)
        WN = L1Sparsity(W)

        x = dg.allocate('random')
        Wx = W.direct(x)

        self.assertAlmostEqual(WN(x), np.sum(Wx.abs().as_array()), msg="L1Sparsity should be the sum of absolute values of the wavelet coefficients", places=5)

        y = W.adjoint(Wx)
        tau = 0.0
        prox = WN.proximal(x, tau=tau)

        self.assertAlmostEqual(0, (prox - y).norm(), places=5, msg="Prox_{tau=0}(  ) should be (almost) the identity")
        self.assertNumpyArrayAlmostEqual(y.as_array(), prox.as_array())

        tau = 1.1
        Wx /= Wx.abs().max() # Bound Wx to [-1, 1]
        W.adjoint(Wx, out=y)
        WN.proximal(y, tau=tau, out=prox)
        self.assertNumpyArrayAlmostEqual(np.zeros(x.shape), prox.as_array())

        # Test for sign problems in prox
        prox = x.copy()
        WN.proximal(prox, tau=tau, out=prox)
        proxNeg = WN.proximal(-x, tau=tau)
        self.assertNumpyArrayAlmostEqual(np.zeros(x.shape), (prox + proxNeg).as_array())

        convConj = WN.convex_conjugate(1.2*y)
        self.assertEqual(np.inf, convConj, msg="Some coefficient should be > 1")

        convConj = WN.convex_conjugate(0.9*y)
        self.assertEqual(0, convConj, msg="All coefficient should be < 1")

        dg = VectorGeometry(48)
        for true_adjoint in [True, False]:
                W = WaveletOperator(dg, wname='bior3.5', level=1, true_adjoint=true_adjoint)
                self.assertEqual(W.is_orthogonal(), False)
                self.assertWarns(UserWarning, L1Sparsity, W)


    
    def test_L1Sparsity_complex_input(self):
        n = 20
        wname = 'db2'
        level = 2

        dg = VectorGeometry(n)
        W = WaveletOperator(dg, wname=wname, level=level)
        self.assertEqual(W.is_orthogonal(), True)
        WN = L1Sparsity(W)

        x = dg.allocate('random', dtype='complex64')
        Wx = W.direct(x)

        self.assertAlmostEqual(WN(x), np.sum(Wx.abs().as_array()), msg="L1Sparsity should be the sum of absolute values of the wavelet coefficients", places =5 )

        y = W.adjoint(Wx)
        tau = 0.0
        prox = WN.proximal(x, tau=tau)

        self.assertAlmostEqual(0, (prox - y).norm(), places=5, msg="Prox_{tau=0}(  ) should be (almost) the identity")
        self.assertNumpyArrayAlmostEqual(y.as_array(), prox.as_array())

        tau = 1.1
        Wx /= Wx.abs().max() # Bound Wx to [-1, 1]
        W.adjoint(Wx, out=y)
        WN.proximal(y, tau=tau, out=prox)
        self.assertNumpyArrayAlmostEqual(np.zeros(x.shape), prox.as_array())

        # Test for sign problems in prox
        prox = x.copy()
        WN.proximal(prox, tau=tau, out=prox)
        proxNeg = WN.proximal(-x, tau=tau)
        self.assertNumpyArrayAlmostEqual(np.zeros(x.shape), (prox + proxNeg).as_array())

        convConj = WN.convex_conjugate(1.2*y)
        self.assertEqual(np.inf, convConj, msg="Some coefficient should be > 1")

        convConj = WN.convex_conjugate(0.9*y)
        self.assertEqual(0, convConj, msg="All coefficient should be < 1")


if __name__ == "__main__":
    unittest.main()