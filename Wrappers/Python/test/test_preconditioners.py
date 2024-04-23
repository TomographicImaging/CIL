
from cil.optimisation.algorithms import SIRT, GD
from cil.optimisation.functions import  LeastSquares
from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.operators import IdentityOperator, MatrixOperator

from cil.optimisation.utilities import  Sensitivity, AdaptiveSensitivity
import numpy as np

from testclass import CCPiTestClass

class TestPreconditioners(CCPiTestClass):
    
    def test_sensitivity_init(self):
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)
        preconditioner = Sensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertNumpyArrayEqual(preconditioner.sensitivity.as_array(), ig.allocate(1.0).as_array())
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), ig.allocate(1.0).as_array())
    
        preconditioner = Sensitivity(A, reference=data)
        
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertNumpyArrayEqual(preconditioner.reference.as_array(), data.as_array())
        self.assertNumpyArrayEqual(preconditioner.sensitivity.as_array(), ig.allocate(1.0).as_array())
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), data.as_array())
    
    
    
    def test_sensitivity_safe_division(self):
        ig = VectorGeometry(10)
        data = ig.allocate('random')
        data.fill(np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]))
        A= MatrixOperator( np.diag([1/2,1/2,1/2,1/2,0.,0.,0.,0.,0.,0.]))
        preconditioner = Sensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertNumpyArrayEqual(preconditioner.sensitivity.as_array(),  np.array([1/2,1/2,1/2,1/2,0,0,0,0,0,0]))
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), np.array([2.,2.,2.,2.,0,0,0,0,0,0]))
    
    def test_sensitivity_against_sirt(self):
        
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)


        sirt = SIRT(ig.allocate(0), A, data,   update_objective_interval=1)
        sirt.run(10)
        
        M = 1./A.direct(A.domain_geometry().allocate(value=1.0)) 
        f = LeastSquares(A=A, b=data, c=0.5, weight=M)
        step_size = 1.
        preconditioner = Sensitivity(A)

        alg = GD(initial=ig.allocate(0), objective_function=f, 
               max_iteration=100, update_objective_interval=1, step_size = step_size)   
        self.assertEqual(alg.preconditioner, None)
        
        precond_pwls = GD(initial=ig.allocate(0), objective_function=f,   preconditioner = preconditioner,
               max_iteration=100, update_objective_interval=1, step_size = step_size)     
        
        def correct_update_objective(alg):
            # SIRT computes |Ax_{k} - b|_2^2
            # GD with weighted LeastSquares computes the objective included the weight, so we remove the weight
            return 0.5*(alg.objective_function.A.direct(alg.x) - alg.objective_function.b).squared_norm()
              
        precond_pwls.run(10)
        np.testing.assert_allclose(sirt.solution.array, precond_pwls.solution.array, atol=1e-4)
        np.testing.assert_allclose(sirt.get_last_loss(), correct_update_objective(precond_pwls), atol=1e-4)
        
        
    def test_adaptive_sensitivity_init(self):
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)
        preconditioner = AdaptiveSensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertEqual(preconditioner.delta, 1e-6)
        self.assertEqual(preconditioner.iterations, 100)
        self.assertNumpyArrayEqual(preconditioner.sensitivity.as_array(), ig.allocate(1.0).as_array())
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), ig.allocate(1.0).as_array())
    
        preconditioner = AdaptiveSensitivity(A, delta=3, iterations=400, reference=data)
        
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertNumpyArrayEqual(preconditioner.reference.as_array(), data.as_array())
        self.assertNumpyArrayEqual(preconditioner.sensitivity.as_array(), ig.allocate(1.0).as_array())
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), data.as_array())
        self.assertEqual(preconditioner.delta, 3)
        self.assertEqual(preconditioner.iterations, 400)
        pass
    
    
    def test_adaptive_sensitivity_converges(self):
        ig = ImageGeometry(7,8,4)
        data = ig.allocate('random')
        A= IdentityOperator(ig)
        initial=data+1e-3
       
   
        f = LeastSquares(A=A, b=data, c=0.5)
        step_size = 1.
        preconditioner = AdaptiveSensitivity(A, iterations=4000, delta=1e-8)

        
        precond_pwls = GD(initial=initial, objective_function=f,   preconditioner = preconditioner, update_objective_interval=1, step_size = step_size)     
        
      
        precond_pwls.run(1000)
        np.testing.assert_allclose(data.array, precond_pwls.solution.array, rtol=1e-5, atol=1e-4)
        
    