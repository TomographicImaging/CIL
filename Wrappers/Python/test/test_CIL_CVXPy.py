# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cil.optimisation.functions import L2NormSquared, BlockFunction, TotalVariation, MixedL21Norm, ZeroFunction
from cil.optimisation.operators import GradientOperator,BlockOperator, ZeroOperator, SymmetrisedGradientOperator, IdentityOperator
from cil.optimisation.algorithms import PDHG
from cil.utilities import dataexample

import numpy as np
import scipy.sparse as sp

import unittest

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class Test_CIL_vs_CVXPy(unittest.TestCase):

    def setUp(self):

        # default test data
        self.data = dataexample.CAMERA.get(size=(32, 32))

    
    # Create gradient operator as a sparse matrix that will be used in CVXpy to define Gradient based regularisers
    def sparse_gradient_matrix(self, shape, direction='forward', order=1, boundaries='Neumann', **kwargs):
        
        len_shape = len(shape)    
        allMat = dict.fromkeys(range(len_shape))
        discretization = kwargs.get('discretization',[1.0]*len_shape)

        if order == 1:

            # loop over the different directions
            for i in range(0,len_shape):

                if direction == 'forward':
                
                    # create a sparse matrix with -1 in the main diagonal and 1 in the 1st diagonal
                    mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [0,1], shape[i], shape[i], format = 'lil')

                    # boundary conditions
                    if boundaries == 'Neumann':
                        mat[-1,:] = 0
                    elif boundaries == 'Periodic':
                        mat[-1,0] = 1

                elif direction == 'backward':

                    # create a sparse matrix with -1 in the -1 and 1 in the main diagonal
                    mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [-1,0], shape[i], shape[i], format = 'lil')

                    # boundary conditions
                    if boundaries == 'Neumann':
                        mat[:,-1] = 0
                    elif boundaries == 'Periodic':
                        mat[0,-1] = -1

                # Use Kronecker product to compute the full sparse matrix representing the GradientOperator. This will be applied
                # to a "flatten" array, i.e., a vector and "the reshaped" result describes the forward/backward differences for all
                # the directions from "len_shape" and under different boundary conditions, e.g., Neumann and Periodic.

                # The difference with the GradientOperator.py and FiniteDifferenceOperator.py is that we do not store in memory a matrix
                # in order to compute the matrix-vector multiplication "A*x". This is also known as "matrix free" optimisation problem.
                # However, to set up and use the API of CVXpy, we need this matrix representation of a linear operator such as the GradientOperator.
                # 
                # The following constructs the finite complete difference matrix for all (len_shape) dimensions and store the ith finite 
                # difference matrix in allmat[i].  In 2D we have
                # allmat[0] = D kron I
                # allmat[1] = I kron D
                # and in 3D
                # allmat[0] = D kron I kron I
                # allmat[1] = I kron D kron I
                # allmat[2] = I kron I kron D
                # and so on, for kron meaning the kronecker product.
                
                # For a (n x m) array, the forward difference operator in y-direction (x-direction) (with Neumann/Periodic bc) is a (n*m x n*m) sparse array containing -1,1.
                # Example, for a 3x3 array U, the forward difference operator in y-direction with Neumann bc is a 9x9 sparse array containing -1,1.
                # To create this sparse matrix, we first create a "kernel" matrix, shown below:
                # mat = [-1, 1, 0
                #        0, -1, 1,
                #        0, 0, 0].
                # where the last row is filled with zeros due to the Neumann boundary condition. Then, we use the Kronecker product: allMat[0] = mat x I_m =
                # matrix([[-1., 1., 0., 0., 0., 0., 0., 0., 0.],
                #         [ 0., -1., 1., 0., 0., 0., 0., 0., 0.],
                #         [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #         [ 0., 0., 0., -1., 1., 0., 0., 0., 0.],
                #         [ 0., 0., 0., 0., -1., 1., 0., 0., 0.],
                #         [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #         [ 0., 0., 0., 0., 0., 0., -1., 1., 0.],
                #         [ 0., 0., 0., 0., 0., 0., 0., -1., 1.],
                #         [ 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
                # where I_m is an (mxm) sparse array with ones in the diagonal.
                # Then allmat can be applied to a (3x3) array "flatten" columnwise
                # and represent the forward differences in y direction,i.e.,
                # [U_{21} - U_{11},
                #  U_{31} - U_{21},
                #        0        ,
                #       ...       ,
                #       ...       ,
                #       ...       ,
                #       ...       ,
                #       ...       ,
                #       ...       ]
                # For the x-direction, we have allmat[1] = I_n x mat.

                # For more details, see "Infimal Convolution Regularizations with Discrete l1-type Functionals, S. Setzer, G. Steidl and T. Teuber"
          
                # According to the direction, tmpGrad is either a kernel matrix or sparse eye array, which is updated 
                # using the kronecker product to derive the sparse matrices.
                if i==0:
                    tmpGrad = mat
                else: 
                    tmpGrad = sp.eye(shape[0])

                for j in range(1, len_shape):

                    if j == i:
                        tmpGrad = sp.kron(mat, tmpGrad ) 
                    else:
                        tmpGrad = sp.kron(sp.eye(shape[j]), tmpGrad )

                allMat[i] = tmpGrad

        else:
            raise NotImplementedError    

        return allMat        

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def tv_cvxpy_regulariser(self, u, isotropic=True, direction = "forward", boundaries = "Neumann"):

        G = self.sparse_gradient_matrix(u.shape, direction = direction, order = 1, boundaries = boundaries)   

        DX, DY = G[1], G[0]
    
        if isotropic:
            return cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u), DY @ cvxpy.vec(u)]), 2, axis = 0))
        else:
            return cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u), DY @ cvxpy.vec(u)]), 1, axis = 0)) 

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_cil_vs_cvxpy_totalvariation_isotropic(self):

        # solution
        u_cvx = cvxpy.Variable(self.data.shape)

        # regularisation parameter
        alpha = 0.1

        # fidelity term
        fidelity = 0.5 * cvxpy.sum_squares(u_cvx - self.data.array)   
        regulariser = (alpha**2) * self.tv_cvxpy_regulariser(u_cvx)

        # objective
        obj =  cvxpy.Minimize( regulariser +  fidelity)
        prob = cvxpy.Problem(obj, constraints = [])

        # Choose solver ( SCS, MOSEK(license needed) )
        tv_cvxpy = prob.solve(verbose = True, solver = cvxpy.SCS)        

        # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
        TV = TotalVariation(max_iteration=200)
        tv_cil = TV.proximal(self.data, tau=alpha**2)     

        # compare solution
        np.testing.assert_allclose(tv_cil.array, u_cvx.value,atol=1e-3)  

        # # compare objectives
        f = 0.5*L2NormSquared(b=self.data)
        cil_objective = f(tv_cil) + TV(tv_cil)*(alpha**2)
        np.testing.assert_allclose(cil_objective, obj.value, atol=1e-3)  

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_cil_vs_cvxpy_totalvariation_anisotropic(self):

            # solution
            u_cvx = cvxpy.Variable(self.data.shape)

            # regularisation parameter
            alpha = 0.1

            # fidelity term
            fidelity = 0.5 * cvxpy.sum_squares(u_cvx - self.data.array)   
            regulariser = alpha * self.tv_cvxpy_regulariser(u_cvx, isotropic=False)

            # objective
            obj =  cvxpy.Minimize( regulariser +  fidelity)
            prob = cvxpy.Problem(obj, constraints = [])

            # Choose solver ( SCS, MOSEK(license needed) )
            tv_cvxpy = prob.solve(verbose = True, solver = cvxpy.SCS)        

            # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
            TV = (alpha/3.0)*TotalVariation(max_iteration=200, isotropic=False)
            tv_cil = TV.proximal(self.data, tau=3.0)     

            # compare solution
            np.testing.assert_allclose(tv_cil.array, u_cvx.value, atol=1e-3)   

            # compare objectives
            f = 0.5*L2NormSquared(b=self.data)
            cil_objective = f(tv_cil) + TV(tv_cil)*(3)
            np.testing.assert_allclose(cil_objective, obj.value, atol=1e-3)    


    def tgv_cvxpy_regulariser(self,u, w1, w2, alpha0, alpha1, boundaries = "Neumann"):

        G1 = self.sparse_gradient_matrix(u.shape, direction = 'forward', order = 1, boundaries = boundaries)  
        DX, DY = G1[1], G1[0]

        G2 = self.sparse_gradient_matrix(u.shape, direction = 'backward', order = 1, boundaries = boundaries) 
        divX, divY = G2[1], G2[0]
    
        return alpha0 * cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u) - cvxpy.vec(w1), DY @ cvxpy.vec(u) - cvxpy.vec(w2)]), 2, axis = 0)) + \
            alpha1 * cvxpy.sum(cvxpy.norm(cvxpy.vstack([ divX @ cvxpy.vec(w1), divY @ cvxpy.vec(w2), \
                                        0.5 * ( divX @ cvxpy.vec(w2) + divY @ cvxpy.vec(w1) ), \
                                        0.5 * ( divX @ cvxpy.vec(w2) + divY @ cvxpy.vec(w1) ) ]), 2, axis = 0  ) )            

    def test_cil_vs_cvxpy_total_generalised_variation(self):
        
        # solution
        u_cvx = cvxpy.Variable(self.data.shape)
        w1_cvx = cvxpy.Variable(self.data.shape)
        w2_cvx = cvxpy.Variable(self.data.shape)

        # regularisation parameters
        alpha0 = 0.1
        alpha1 = 0.3

        # fidelity term
        fidelity = 0.5 * cvxpy.sum_squares(u_cvx - self.data.array)   
        regulariser = self.tgv_cvxpy_regulariser(u_cvx, w1_cvx, w2_cvx, alpha1, alpha0)

        # objective
        obj =  cvxpy.Minimize( regulariser +  fidelity)
        prob = cvxpy.Problem(obj, constraints = [])

        # Choose solver ( SCS, MOSEK(license needed) )
        tv_cvxpy = prob.solve(verbose = True, solver = cvxpy.SCS)   

        # setup and run PDHG algorithm for TGV denoising               
        ig = self.data.geometry

        K11 = GradientOperator(ig)
        K22 = SymmetrisedGradientOperator(K11.range)
        K12 = IdentityOperator(K11.range)
        K21 = ZeroOperator(ig, K22.range)

        # operator = [GradientOperator,  -IdentityOperator
        #             ZeroOperator,       SymmetrisedGradientOperator]
        K = BlockOperator(K11, -K12, K21, K22, shape=(2,2) )

        f1 = alpha1 * MixedL21Norm()
        f2 = alpha0 * MixedL21Norm()
        F = BlockFunction(f1, f2)
        G = BlockFunction(0.5 * L2NormSquared(b=self.data), ZeroFunction())

        sigma = 1./np.sqrt(12)
        tau = 1./np.sqrt(12)

        # Setup and run the PDHG algorithm
        pdhg_tgv = PDHG(f=F,g=G,operator=K,
                    max_iteration = 500, sigma=sigma, tau=tau,
                    update_objective_interval = 500)
        pdhg_tgv.run(verbose = 0)

        # compare solution
        np.testing.assert_allclose(pdhg_tgv.solution[0].array, u_cvx.value, atol=1e-3) 
                       




    



