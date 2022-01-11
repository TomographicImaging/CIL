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
from cvxpy import *

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
            return sum(norm(vstack([DX @ vec(u), DY @ vec(u)]), 2, axis = 0))
        else:
            return sum(norm(vstack([DX @ vec(u), DY @ vec(u)]), 1, axis = 0)) 

    def test_cil_vs_cvxpy_totalvariation_isotropic(self):

        # solution
        u_cvx = Variable(self.data.shape)

        # regularisation parameter
        alpha = 0.1

        # fidelity term
        fidelity = 0.5 * sum_squares(u_cvx - self.data.array)   
        regulariser = alpha * self.tv_cvxpy_regulariser(u_cvx)

        # objective
        obj =  Minimize( regulariser +  fidelity)
        prob = Problem(obj, constraints = [])

        # Choose solver ( SCS, MOSEK(license needed) )
        tv_cvxpy = prob.solve(verbose = True, solver = SCS)        

        # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
        TV = alpha * TotalVariation(max_iteration=200)
        tv_cil = TV.proximal(self.data, tau=1.0)     

        # compare solution
        np.testing.assert_almost_equal(tv_cil.array, u_cvx.value, decimal=3)   

    def test_cil_vs_cvxpy_totalvariation_anisotropic(self):

            # solution
            u_cvx = Variable(self.data.shape)

            # regularisation parameter
            alpha = 0.1

            # fidelity term
            fidelity = 0.5 * sum_squares(u_cvx - self.data.array)   
            regulariser = alpha * self.tv_cvxpy_regulariser(u_cvx, isotropic=False)

            # objective
            obj =  Minimize( regulariser +  fidelity)
            prob = Problem(obj, constraints = [])

            # Choose solver ( SCS, MOSEK(license needed) )
            tv_cvxpy = prob.solve(verbose = True, solver = SCS)        

            # use TotalVariation from CIL (with Fast Gradient Projection algorithm)
            TV = alpha * TotalVariation(max_iteration=200, isotropic=False)
            tv_cil = TV.proximal(self.data, tau=1.0)     

            # compare solution
            np.testing.assert_almost_equal(tv_cil.array, u_cvx.value, decimal=3)   




    



