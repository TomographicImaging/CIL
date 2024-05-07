
from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA
from cil.optimisation.functions import  LeastSquares, IndicatorBox
from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.operators import IdentityOperator, MatrixOperator

from cil.optimisation.utilities import  Sensitivity, AdaptiveSensitivity, Adam, AdaGrad, Preconditioner
import numpy as np

from testclass import CCPiTestClass
from unittest.mock import MagicMock

class TestPreconditioners(CCPiTestClass):
    
    def test_preconditioner_called(self):
        
        
        ig = ImageGeometry(2,1,4)
        data = ig.allocate(1)
        A= IdentityOperator(ig)
        test_precon = MagicMock(None)
        f = LeastSquares(A=A, b=data, c=0.5)
        alg = GD(initial=ig.allocate('random', seed=10), objective_function=f, preconditioner = test_precon ,
               max_iteration=100, update_objective_interval=1, step_size = 0.0000001)
        alg.run(5)
        self.assertEqual(len(test_precon.mock_calls),5)
        
        test_precon = MagicMock(None)
        alg = ISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), preconditioner = test_precon ,
               max_iteration=100, update_objective_interval=1, step_size = 0.0000001)
        alg.run(5)
        self.assertEqual(len(test_precon.mock_calls),5)
    
        test_precon = MagicMock(None)
        alg = FISTA(initial=ig.allocate('random', seed=10), f=f, g=IndicatorBox(lower=0), preconditioner = test_precon ,
               max_iteration=100, update_objective_interval=1, step_size = 0.0000001)
        alg.run(5)
        self.assertEqual(len(test_precon.mock_calls),5)
        
        
    def test_sensitivity_init(self):
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)
        preconditioner = Sensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), ig.allocate(1.0).as_array())
    
        preconditioner = Sensitivity(A, reference=data)
        
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertNumpyArrayEqual(preconditioner.reference.as_array(), data.as_array())
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), data.as_array())
    
   
    
    def test_sensitivity_calculation(self):
        ig = VectorGeometry(10)
        data = ig.allocate('random')
        data.fill(np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]))
        A= MatrixOperator( np.diag([1/2,1/2,1/2,1/2,0.,0.,0.,0.,0.,0.]))
        preconditioner = Sensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), np.array([2.,2.,2.,2.,0,0,0,0,0,0]))
        
        f=LeastSquares(A, data)
        alg = GD(initial=ig.allocate(0), objective_function=f, 
               max_iteration=100, update_objective_interval=1, step_size = 1.)  
        alg.gradient_update=ig.allocate(2)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(), np.array([4.,4.,4.,4.,0,0,0,0,0,0]))

       
        

        
        
    def test_sensitivity_gd_against_sirt(self):
        
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random', seed=4)
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
        
    def test_sensitivity_ista_against_sirt(self): 
        
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)


        sirt = SIRT(ig.allocate(0), A, data, lower=0,  update_objective_interval=1)
        sirt.run(10)
        
        M = 1./A.direct(A.domain_geometry().allocate(value=1.0)) 
        f = LeastSquares(A=A, b=data, c=0.5, weight=M)
        g=IndicatorBox(lower=0)
        step_size = 1.
        preconditioner = Sensitivity(A)

        alg = ISTA(initial=ig.allocate(0), f=f, g=g, 
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
        
    def test_sensitivity_reference_ista_converges(self):
        ig = ImageGeometry(7,8,4)
        data = ig.allocate(0.5)
        A= IdentityOperator(ig)
        initial=ig.allocate('random', seed=2)

        f = LeastSquares(A=A, b=data, c=0.5)
        g=IndicatorBox(lower=0, upper=1)
        step_size = 1.
        preconditioner = Sensitivity(A, reference=ig.allocate(0.45))

        
        precond_pwls = ISTA(initial=initial, f=f, g=g,   preconditioner = preconditioner, update_objective_interval=1, step_size = step_size)     
        
    
        precond_pwls.run(30)
        self.assertNumpyArrayAlmostEqual(data.array, precond_pwls.solution.array, 4)

      
    
        
    def test_adaptive_sensitivity_init(self):
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)
        preconditioner = AdaptiveSensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertEqual(preconditioner.delta, 1e-6)
        self.assertEqual(preconditioner.iterations, 100)
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), ig.allocate(1.0).as_array())
    
        preconditioner = AdaptiveSensitivity(A, delta=3, iterations=400, reference=data)
        
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertNumpyArrayEqual(preconditioner.reference.as_array(), data.as_array())
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), data.as_array())
        self.assertEqual(preconditioner.delta, 3)
        self.assertEqual(preconditioner.iterations, 400)
        
    
    def test_adaptive_sensitivity_calculations(self):
        ig = VectorGeometry(10)
        data = ig.allocate('random')
        data.fill(np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]))
        A= MatrixOperator( np.diag([1/2,1/2,1/2,1/2,0.,0.,0.,0.,0.,0.]))
        preconditioner = AdaptiveSensitivity(A)
        self.assertNumpyArrayAlmostEqual(preconditioner.operator.direct(data).as_array(), A.direct(data).as_array())
        self.assertEqual(preconditioner.reference, None)
        self.assertNumpyArrayEqual(preconditioner.array.as_array(), np.array([2.,2.,2.,2.,0,0,0,0,0,0]))
        
        
        f=LeastSquares(A, data)
        alg = GD(initial=ig.allocate(0), objective_function=f, 
               max_iteration=100, update_objective_interval=1, step_size = 1.)  
        alg.gradient_update=ig.allocate(1)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.freezing_point.as_array(), np.array([2e-6,2e-6,2e-6,2e-6,0,0,0,0,0,0]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(), np.array([2e-6,2e-6,2e-6,2e-6,0,0,0,0,0,0]))
        alg.gradient_update=ig.allocate(1)
        alg.x=ig.allocate(1)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.freezing_point.as_array(), np.array([2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),0,0,0,0,0,0]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(), np.array([2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),0,0,0,0,0,0]))
       
        preconditioner = AdaptiveSensitivity(A, iterations=0)
        alg.gradient_update=ig.allocate(1)
        alg.x=ig.allocate(1)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.freezing_point.as_array(), np.array([2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),0,0,0,0,0,0]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(), np.array([2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),0,0,0,0,0,0]))
        alg.run(4)
        self.assertNumpyArrayAlmostEqual(preconditioner.freezing_point.as_array(), np.array([2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),2*(1+1e-6),0,0,0,0,0,0]))
        
       
    
    
    
    def test_adaptive_sensitivity_gd_converges(self):
        ig = ImageGeometry(7,8,4)
        data = ig.allocate('random', seed=2)
        A= IdentityOperator(ig)
        initial=ig.allocate(0)
   
        f = LeastSquares(A=A, b=data, c=0.5)
        step_size = 1.
        preconditioner = AdaptiveSensitivity(A, iterations=3000, delta=1e-8)

        
        precond_pwls = GD(initial=initial, objective_function=f,   preconditioner = preconditioner, update_objective_interval=1, step_size = step_size)     
        
      
        precond_pwls.run(3000)
        self.assertNumpyArrayAlmostEqual(data.array, precond_pwls.solution.array, 3)
        
        
    def test_adaptive_sensitivity_fista_converges(self):
        ig = ImageGeometry(7,8,4)
        data = ig.allocate(0.5)
        A= IdentityOperator(ig)
        initial = ig.allocate('random', seed=2)
   
        f = LeastSquares(A=A, b=data, c=0.5)
        g = IndicatorBox(lower=0, upper=1)
        step_size = 1.
        preconditioner = AdaptiveSensitivity(A, iterations=3000, delta=1e-8)

        
        precond_pwls = FISTA(initial=initial, f=f, g=g,   preconditioner = preconditioner, update_objective_interval=1, step_size = step_size)     
        
      
        precond_pwls.run(30)
        self.assertNumpyArrayAlmostEqual(data.array, precond_pwls.solution.array, 3)
        
        
        
    def test_Adam_init(self):
        preconditioner = Adam()
        self.assertEqual(preconditioner.epsilon, 1e-8)
        self.assertEqual(preconditioner.gamma, 0.9)
        self.assertEqual(preconditioner.beta, 0.999)
        self.assertEqual(preconditioner.gradient_accumulator, None)
        self.assertEqual(preconditioner.scaling_factor_accumulator, None)
        
        preconditioner = Adam(epsilon=1e-4, gamma=4, beta=5)
        self.assertEqual(preconditioner.epsilon, 1e-4)
        self.assertEqual(preconditioner.gamma, 4)
        self.assertEqual(preconditioner.beta, 5)
        self.assertEqual(preconditioner.gradient_accumulator, None)
        self.assertEqual(preconditioner.scaling_factor_accumulator, None)
        
    def test_Adam_calculations(self):
        preconditioner = Adam(gamma=.5, beta=.5, epsilon =1 )
        ig = VectorGeometry(10)
        data = ig.allocate('random')
        data.fill(np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]))
        A= MatrixOperator( np.diag([1/2,1/2,1/2,1/2,0.,0.,0.,0.,0.,0.]))
        
        f=LeastSquares(A, data)
        alg = GD(initial=ig.allocate(0), objective_function=f, 
               max_iteration=100, update_objective_interval=1, step_size = 1.)  
        alg.gradient_update=ig.allocate(1)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.gradient_accumulator.as_array(), np.array([1,1,1,1,1,1,1,1,1,1]))
        self.assertNumpyArrayAlmostEqual(preconditioner.scaling_factor_accumulator.as_array(), np.array([1,1,1,1,1,1,1,1,1,1]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(), np.sqrt(np.array([1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2])))
        
        alg.gradient_update=ig.allocate(2)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.gradient_accumulator.as_array(), np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]))
        self.assertNumpyArrayAlmostEqual(preconditioner.scaling_factor_accumulator.as_array(), np.array([2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(),  1.5*np.sqrt(np.array([2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7])))
       
       
        
    def test_Adam_converges(self):
        ig = ImageGeometry(7,8,4)
        data = ig.allocate('random', seed=2)
        A= IdentityOperator(ig)
        initial=ig.allocate(0)

        f = LeastSquares(A=A, b=data, c=0.5)
        step_size = 1
        preconditioner = Adam()

        
        ls_adam = GD(initial=initial, objective_function=f,   preconditioner = preconditioner, update_objective_interval=1, step_size = step_size)     
        
    
        ls_adam.run(200)
        self.assertNumpyArrayAlmostEqual(data.array, ls_adam.solution.array, 3)

    def test_AdaGrad_init(self):
        preconditioner = AdaGrad()
        self.assertEqual(preconditioner.epsilon, 1e-8)
        self.assertEqual(preconditioner.gradient_accumulator, None)
        
        preconditioner = AdaGrad(1e-4)
        self.assertEqual(preconditioner.epsilon, 1e-4)
        self.assertEqual(preconditioner.gradient_accumulator, None)
        
    def test_AdaGrad_calculations(self):
        preconditioner = AdaGrad( epsilon =1 )
        ig = VectorGeometry(10)
        data = ig.allocate('random')
        data.fill(np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]))
        A= MatrixOperator( np.diag([1/2,1/2,1/2,1/2,0.,0.,0.,0.,0.,0.]))
        
        f=LeastSquares(A, data)
        alg = GD(initial=ig.allocate(0), objective_function=f, 
               max_iteration=100, update_objective_interval=1, step_size = 1.)  
        alg.gradient_update=ig.allocate(1)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.gradient_accumulator.as_array(), np.array([1,1,1,1,1,1,1,1,1,1]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(), np.sqrt(np.array([1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2])))
        
        alg.gradient_update=ig.allocate(2)
        preconditioner(alg)
        self.assertNumpyArrayAlmostEqual(preconditioner.gradient_accumulator.as_array(), np.array([5,5,5,5,5,5,5,5,5,5]))
        self.assertNumpyArrayAlmostEqual(alg.gradient_update.as_array(),  2*np.sqrt(np.array([1/6,1/6,1/6,1/6,1/6,1/6,1/6,1/6,1/6,1/6])))
       
       
      
    def test_AdaGrad_converges(self):
            ig = ImageGeometry(7,8,4)
            data = ig.allocate('random', seed=2)
            A= IdentityOperator(ig)
            initial=ig.allocate(0)
    
            f = LeastSquares(A=A, b=data, c=0.5)
            step_size = 1
            preconditioner = AdaGrad()

            
            ls_ada = GD(initial=initial, objective_function=f,   preconditioner = preconditioner, update_objective_interval=1, step_size = step_size)     
            
        
            ls_ada.run(1500)
            self.assertNumpyArrayAlmostEqual(data.array, ls_ada.solution.array, 3)
            
    