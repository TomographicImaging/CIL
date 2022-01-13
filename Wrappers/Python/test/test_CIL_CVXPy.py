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

from cil.optimisation.functions import L2NormSquared, TotalVariation, MixedL21Norm, BlockFunction, ZeroFunction
from cil.optimisation.operators import GradientOperator, SymmetrisedGradientOperator, IdentityOperator, ZeroOperator, BlockOperator
from cil.optimisation.algorithms import PDHG
from cil.utilities import dataexample

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

import unittest

class Test_CIL_vs_CVXPy(unittest.TestCase):

    def setUp(self):

        # default test data
        self.data = dataexample.CAMERA.get(size=(32, 32))

    
    # Create gradient operator as sparse matrix that will be used in CVXpy
    def sparse_gradient_matrix(self, shape, direction='forward', order=1, boundaries='Neumann', **kwargs):
        
        len_shape = len(shape)    
        allMat = dict.fromkeys(range(len_shape))
        discretization = kwargs.get('discretization',[1.0]*len_shape)

        if order == 1:

            for i in range(0,len_shape):

                if direction == 'forward':

                    mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [0,1], shape[i], shape[i], format = 'lil')

                    if boundaries == 'Neumann':
                        mat[-1,:] = 0
                    elif boundaries == 'Periodic':
                        mat[-1,0] = 1

                elif direction == 'backward':

                    mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [-1,0], shape[i], shape[i], format = 'lil')

                    if boundaries == 'Neumann':
                        mat[:,-1] = 0
                    elif boundaries == 'Periodic':
                        mat[0,-1] = -1

                tmpGrad = mat if i == 0 else sp.eye(shape[0])

                for j in range(1, len_shape):

                    tmpGrad = sp.kron(mat, tmpGrad ) if j == i else sp.kron(sp.eye(shape[j]), tmpGrad )

                allMat[i] = tmpGrad

        else:
            raise NotImplementedError    

        return allMat        

    def tv_cvxpy_regulariser(self, u, isotropic=True, direction = "forward", boundaries = "Neumann"):

        G = self.sparse_gradient_matrix(u.shape, direction = direction, order = 1, boundaries = boundaries)   

        DX, DY = G[1], G[0]
    
        if isotropic:
            return cp.sum(cp.norm(cp.vstack([DX @ cp.vec(u), DY @ cp.vec(u)]), 2, axis = 0))
        else:
            return cp.sum(cp.norm(cp.vstack([DX @ cp.vec(u), DY @ cp.vec(u)]), 1, axis = 0)) 

    def test_cil_vs_cvxpy_totalvariation_isotropic(self):

        # solution
        u_cvx = cp.Variable(self.data.shape)

        # regularisation parameter
        alpha = 0.1

        # fidelity term
        fidelity = 0.5 * cp.sum_squares(u_cvx - self.data.array)   
        regulariser = alpha * self.tv_cvxpy_regulariser(u_cvx)

        # objective
        obj =  cp.Minimize( regulariser +  fidelity)
        prob = cp.Problem(obj, constraints = [])

        # Choose solver ( SCS, MOSEK(license needed) )
        tv_cvxpy = prob.solve(verbose = True, solver = cp.SCS)        

        # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
        TV = alpha * TotalVariation(max_iteration=200)
        tv_cil = TV.proximal(self.data, tau=1.0)     

        # compare solution
        np.testing.assert_allclose(tv_cil.array, u_cvx.value,atol=1e-3)   

    def test_cil_vs_cvxpy_totalvariation_anisotropic(self):

            # solution
            u_cvx = cp.Variable(self.data.shape)

            # regularisation parameter
            alpha = 0.1

            # fidelity term
            fidelity = 0.5 * cp.sum_squares(u_cvx - self.data.array)   
            regulariser = alpha * self.tv_cvxpy_regulariser(u_cvx, isotropic=False)

            # objective
            obj =  cp.Minimize( regulariser +  fidelity)
            prob = cp.Problem(obj, constraints = [])

            # Choose solver ( SCS, MOSEK(license needed) )
            tv_cvxpy = prob.solve(verbose = True, solver = cp.SCS)        

            # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
            TV = alpha * TotalVariation(max_iteration=200, isotropic=False)
            tv_cil = TV.proximal(self.data, tau=1.0)     

            # compare solution
            np.testing.assert_allclose(tv_cil.array, u_cvx.value, atol=1e-3) 

    def tgv_cvxpy_regulariser(self,u, w1, w2, alpha0, alpha1, boundaries = "Neumann"):

        G1 = self.sparse_gradient_matrix(u.shape, direction = 'forward', order = 1, boundaries = boundaries)  
        DX, DY = G1[1], G1[0]

        G2 = self.sparse_gradient_matrix(u.shape, direction = 'backward', order = 1, boundaries = boundaries) 
        divX, divY = G2[1], G2[0]
    
        return alpha0 * cp.sum(cp.norm(cp.vstack([DX @ cp.vec(u) - cp.vec(w1), DY @ cp.vec(u) - cp.vec(w2)]), 2, axis = 0)) + \
            alpha1 * cp.sum(cp.norm(cp.vstack([ divX @ cp.vec(w1), divY @ cp.vec(w2), \
                                        0.5 * ( divX @ cp.vec(w2) + divY @ cp.vec(w1) ), \
                                        0.5 * ( divX @ cp.vec(w2) + divY @ cp.vec(w1) ) ]), 2, axis = 0  ) )            

    def test_cil_vs_cvxpy_total_generalised_variation(self):
        
        # solution
        u_cvx = cp.Variable(self.data.shape)
        w1_cvx = cp.Variable(self.data.shape)
        w2_cvx = cp.Variable(self.data.shape)

        # regularisation parameters
        alpha0 = 0.1
        alpha1 = 0.3

        # fidelity term
        fidelity = 0.5 * cp.sum_squares(u_cvx - self.data.array)   
        regulariser = self.tgv_cvxpy_regulariser(u_cvx, w1_cvx, w2_cvx, alpha1, alpha0)

        # objective
        obj =  cp.Minimize( regulariser +  fidelity)
        prob = cp.Problem(obj, constraints = [])

        # Choose solver ( SCS, MOSEK(license needed) )
        tv_cvxpy = prob.solve(verbose = True, solver = cp.SCS)   

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
      


    



