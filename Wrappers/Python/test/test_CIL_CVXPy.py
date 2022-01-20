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

from cil.optimisation.functions import L2NormSquared
from cil.optimisation.functions import TotalVariation

from cil.utilities import dataexample

import numpy as np
import scipy.sparse as sp

import unittest

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy as cp

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

                # use Kronecker product to compute the full sparse matrix for the finite difference according to the direction
                # Dx = I_n x D -->  Sparse Eye x mat
                # Dy = D x I_m -->   mat x Sparse Eye
                # Reference: Infimal Convolution Regularizations with Discrete l1-type Functionals, S. Setzer, G. Steidl and T. Teuber                
                tmpGrad = mat if i == 0 else sp.eye(shape[0])

                for j in range(1, len_shape):

                    tmpGrad = sp.kron(mat, tmpGrad ) if j == i else sp.kron(sp.eye(shape[j]), tmpGrad )

                allMat[i] = tmpGrad

        else:
            raise NotImplementedError    

        return allMat        

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def tv_cvxpy_regulariser(self, u, isotropic=True, direction = "forward", boundaries = "Neumann"):

        G = self.sparse_gradient_matrix(u.shape, direction = direction, order = 1, boundaries = boundaries)   

        DX, DY = G[1], G[0]
    
        if isotropic:
            return cp.sum(cp.norm(cp.vstack([DX @ cp.vec(u), DY @ cp.vec(u)]), 2, axis = 0))
        else:
            return cp.sum(cp.norm(cp.vstack([DX @ cp.vec(u), DY @ cp.vec(u)]), 1, axis = 0)) 

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_cil_vs_cvxpy_totalvariation_isotropic(self):
        return self.isotropic_tv()

    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
    def test_cil_vs_cvxpy_totalvariation_isotropic_cupy(self):
        return self.isotropic_tv('cupy')

    def isotropic_tv(self, backend='numpy'):

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
        TV = alpha * TotalVariation(max_iteration=200, backend=backend)
        tv_cil = TV.proximal(self.data, tau=1.0)     

        # compare solution
        np.testing.assert_allclose(tv_cil.array, u_cvx.value,atol=1e-3)  

        # compare objectives
        f = 0.5*L2NormSquared(b=self.data)
        cil_objective = TV(tv_cil) + f(tv_cil)
        np.testing.assert_allclose(cil_objective, obj.value, atol=1e-3)  


        
    @unittest.skipUnless(has_cvxpy, "CVXpy not installed")
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

            # compare objectives
            f = 0.5*L2NormSquared(b=self.data)
            cil_objective = TV(tv_cil) + f(tv_cil)
            np.testing.assert_allclose(cil_objective, obj.value, atol=1e-3)              




    



