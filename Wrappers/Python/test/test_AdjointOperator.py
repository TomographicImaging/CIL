
from utils import initialise_tests
from cil.framework import ImageGeometry
from cil.optimisation.operators import GradientOperator, AdjointOperator, MatrixOperator
import numpy as np 


from testclass import CCPiTestClass


initialise_tests()

class TestAdjointOperator(CCPiTestClass):
    
    def setUp(self):
        self.ig_real = ImageGeometry(3,4)
        self.ig_complex = ImageGeometry(4,5, dtype="complex")
    
    def tearDown(self):
        pass   
    
    def test_direct_adjoint_gradient_operator(self):

        G = GradientOperator(self.ig_real)
        x = G.domain_geometry().allocate("random", seed=10)
        y = G.range_geometry().allocate("random", seed=10)
        G_adjoint = AdjointOperator(G)

        res1 = G.adjoint(y)
        res2 = G_adjoint.direct(y)
        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3)

        res3 = G.direct(x)
        res4 = G_adjoint.adjoint(x)
        np.testing.assert_allclose(res3[0].array, res4[0].array, atol=1e-3)
        np.testing.assert_allclose(res3[1].array, res4[1].array, atol=1e-3)
        
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
        np.testing.assert_allclose(res1.array, res2.array, atol=1e-3)

        res3 = Amat.direct(x)
        res4 = Amat_tr.adjoint(x)
        np.testing.assert_allclose(res3.array, res4.array, atol=1e-3) 

        self.assertTrue(Amat.dot_test(Amat), True)  
        self.assertTrue(Amat_tr.dot_test(Amat_tr), True)

        # <Ax, y> = <x, A^T y>
        lhs = res3.dot(y)
        rhs_a = x.dot(res2)
        rhs_b = x.dot(Amat.adjoint(y))
        np.testing.assert_allclose(lhs, rhs_a, atol=1e-3) 
        np.testing.assert_allclose(lhs, rhs_b, atol=1e-3) 

    # def test_direct_adjoint_matrix_operator_complex(self):
        
    #     np.random.seed(10)
    #     n = 3  
    #     m = 2
    #     Anp = np.random.normal(0,1, (m, n)).astype('float32') + np.random.normal(-1,1, (m, n)).astype('float32') + 0j
        
    #     Amat = MatrixOperator(Anp)
    #     Amat_tr = MatrixOperator(Anp.T)

    #     x = Amat.domain_geometry().allocate("random", dtype="complex")
    #     y = Amat.range_geometry().allocate("random", dtype="complex")


    #     lhs = Amat.direct(x).dot(y)
    #     rhs = x.dot(Amat_tr.direct(y)).conjugate()
    #     print(lhs)
    #     print(rhs)
    #     # res1 = Amat.direct(x)
    #     # res2 = Amat_tr.adjoint(x)
    #     # print(res1.array)
    #     # print(res2.array)

    #     # res1 = Amat.adjoint(y)
    #     # res2 = Amat_tr.direct(y)
    #     # np.testing.assert_allclose(res1.array, res2.array, atol=1e-3)

    #     # res3 = Amat.direct(x)
    #     # res4 = Amat_tr.adjoint(x)
    #     # np.testing.assert_allclose(res3.array, res4.array, atol=1e-3) 

    #     # self.assertTrue(Amat.dot_test(Amat), True)  
    #     # self.assertTrue(Amat_tr.dot_test(Amat_tr), True)

    #     # # <Ax, y> = <x, A^T y>
    #     # lhs = res3.dot(y)
    #     # rhs_a = x.dot(res2)
    #     # rhs_b = x.dot(Amat.adjoint(y))
    #     # np.testing.assert_allclose(lhs, rhs_a, atol=1e-3) 
    #     # np.testing.assert_allclose(lhs, rhs_b, atol=1e-3)         



            



