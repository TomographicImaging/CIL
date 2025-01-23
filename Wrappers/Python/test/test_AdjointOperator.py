
from utils import initialise_tests
from cil.framework import ImageGeometry
from cil.optimisation.operators import GradientOperator, AdjointOperator, MatrixOperator
import numpy as np 


from testclass import CCPiTestClass


initialise_tests()

class TestAdjointOperator(CCPiTestClass):
    
    def setUp(self):
        self.ig_real = ImageGeometry(3,4)
    
    def tearDown(self):
        pass   
    
    def test_direct_adjoint_gradient_operator(self):

        G = GradientOperator(self.ig_real)
        x = G.domain_geometry().allocate("random", seed=10)
        y = G.range_geometry().allocate("random", seed=10)
        G_adjoint = AdjointOperator(G)

        res1 = G.adjoint(y)
        res2 = G_adjoint.direct(y)
        np.testing.assert_allclose(res1.array, res2.array, atol=1e-6)

        res3 = G.direct(x)
        res4 = G_adjoint.adjoint(x)
        np.testing.assert_allclose(res3[0].array, res4[0].array, atol=1e-6)
        np.testing.assert_allclose(res3[1].array, res4[1].array, atol=1e-6)
        
    def test_direct_adjoint_matrix_operator(self):
        
        np.random.seed(10)
        n = 3  
        m = 5 
        Anp = np.random.normal(0,1, (m, n)).astype('float32')
        
        Amat = MatrixOperator(Anp)
        Amat_tr = MatrixOperator(Anp.T)

        x = Amat.domain_geometry().allocate("random")
        y = Amat.range_geometry().allocate("random")

        res1 = Amat.adjoint(y)
        res2 = Amat_tr.direct(y)
        np.testing.assert_allclose(res1.array, res2.array, atol=1e-6)

        res3 = Amat.direct(x)
        res4 = Amat_tr.adjoint(x)
        np.testing.assert_allclose(res3.array, res4.array, atol=1e-6) 

        self.assertTrue(Amat.dot_test(Amat), True)  
        self.assertTrue(Amat_tr.dot_test(Amat_tr), True)

        # <Ax, y> = <x, A^T y>
        lhs = res3.dot(y)
        rhs_a = x.dot(res2)
        rhs_b = x.dot(Amat.adjoint(y))
        np.testing.assert_allclose(lhs, rhs_a, atol=1e-6) 
        np.testing.assert_allclose(lhs, rhs_b, atol=1e-6) 

    def test_direct_adjoint_matrix_operator_complex(self):
        
        np.random.seed(10)
        n = 3  
        m = 2
        Anp = np.random.uniform(0,1, (n, m)) - 1.j*np.random.uniform(0,1, (n, m)) + 3.j*np.random.uniform(0,1, (n, m))
        
        Amat = MatrixOperator(Anp)
        Amat_tr = AdjointOperator(Amat)

        x = Amat.domain_geometry().allocate("random", dtype="complex")
        y = Amat.range_geometry().allocate("random", dtype="complex")


        # <Ax,y> = <x, A^* y> using numpy arrays
        # convention is to conjugate the second vector, see `dot` method in `framework`
        lhs_np = np.dot(Amat.direct(x).array,y.array.conjugate())
        rhs_np = np.dot(x.array, Amat.adjoint(y).conjugate().array)
        np.testing.assert_allclose(lhs_np, rhs_np, atol=1e-6) 

        # using `dot` method from DataContainer
        lhs = Amat.direct(x).dot(y) # conjugate of y is applied in the dot method
        rhs_a = x.dot(Amat.adjoint(y)) # conjugate of y is applied in the dot method
        rhs_b = x.dot(Amat_tr.direct(y)) 

        np.testing.assert_allclose(lhs, rhs_a, atol=1e-6)  
        np.testing.assert_allclose(lhs, rhs_b, atol=1e-6)         


    def test_adjoint_operator_and_dot(self):
        ig = ImageGeometry(3,4,  dtype="complex")
        G = GradientOperator(ig)
        div = AdjointOperator(G)

        x = G.domain.allocate("random_int")
        y = G.range.allocate("random_int")
        print(x.array)
        res1 = G.direct(x).dot(y)
        res2 = x.dot(div.direct(y))
        self.assertAlmostEqual(res1,res2)