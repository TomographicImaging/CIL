#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:44:05 2019

@author: evangelos
"""

import scipy.sparse as sp
import numpy as np
import sys
sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/working_with_CompDataContainers')

from ccpi.framework import ImageData
from GradientOperator import Gradient



def GradOper(size, discStep, direction, order, bndrs): 
    
#    '''' Gradient sparse matrices of 1st and 2nd order(Hessian) 
#    with Neumann and Periodic boundary conditions.
#    1st order ---> Gradient/Divergence sparse matrices for Neum, Per
#    2nd order ---> Hessian and its adjoint for Neum, Per
#    
#    Structure of matrices is for columnwise operations, i.e.,
#    
#    Example:
#    
#    u = np.random.randint(10, size = (4,3))
#    D = GradOper(u.shape, [1,1], 'for', '1', 'Neum')
#    DYu = np.reshape(D[0] * u.flatten('F'), u.shape, order = 'F') 
#    DXu = np.reshape(D[0] * u.flatten('F'), u.shape, order = 'F')
#    
#    ''''

    # TODO labelling the output matrices i.e, DX, DY
    
    allMat = dict.fromkeys((range(len(discStep))))
        
    if len(size)!=len(discStep):
        raise ValueError('Check dimensions of discStep and input size') 

    if 0 in discStep:
        raise ValueError('Zero step size')
                                                                 
    if order == '1':

        for i in range(0,len(discStep)):
        # Get matrices structure to apply kron product
            if direction == 'for':
                
                mat = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')

                if bndrs == 'Neumann':
                    mat[-1,:] = 0
                elif bndrs == 'Periodic':
                    mat[-1,0] = 1

            elif direction == 'back':
                
                mat = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [-1,0], size[i], size[i], format = 'lil')

                if bndrs == 'Neumann':
                    mat[:,-1] = 0
                elif bndrs == 'Periodic':
                    mat[0,-1] = -1
                                     
            tmpGrad = mat if i == 0 else sp.eye(size[0])
#
            for j in range(1, len(size)):
#
                tmpGrad = sp.kron(mat, tmpGrad ) if j == i else sp.kron(sp.eye(size[j]), tmpGrad )

            allMat[i] = tmpGrad
          
        return allMat  ##################      Gives output DY, DX           

    elif order == '2': 
        
        mat = dict.fromkeys(range(len(discStep)))
        mat1 = mat.copy()
        allMat = dict.fromkeys(range(len(discStep))) 
        allMat1 = dict.fromkeys(range(len(discStep),2*len(discStep)))
        
        for i in range(0,len(discStep)):

            if direction == 'for' or direction == 'back':
                mat[i] = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')
                mat1[i] = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')
            
            if bndrs == 'Neumann':
                mat[i][-1,:] = 0 # zero last row
                mat1[i][:,0] = 0 # zero first col
                
            if bndrs == 'Periodic':
                mat[i][-1,0] = 1 # zero last row
                mat1[i][-1,0] = 1 # zero first col                 
                                                  
        for i in range(0,len(discStep)):
                        
            if bndrs == 'Neumann':
                tmpGrad1 = -mat1[i].T                 
                tmpGrad = -mat[i].T * mat[i] if i == 0 else sp.eye(size[0])
            
                for j in range(1, len(discStep)):
                    tmpGrad = sp.kron(-mat[j].T * mat[j], tmpGrad ) if j == i else sp.kron(sp.eye(size[j]), tmpGrad )
                    tmpGrad1 = sp.kron(tmpGrad1,mat[j-1]) if j == i else sp.kron(mat[j], tmpGrad1)
                                                  
            if bndrs == 'Periodic': 
                tmpGrad1 = mat1[i]
                tmpGrad = -mat[i].T * mat[i] if i == 0 else sp.eye(size[0])
              
                for j in range(1, len(discStep)):
                    tmpGrad = sp.kron(-mat[j].T * mat[j], tmpGrad ) if j == i else sp.kron(sp.eye(size[j]), tmpGrad )
                    tmpGrad1 = sp.kron(tmpGrad1,mat[j-1]) if j == i else sp.kron(mat1[j],mat[j-1]) #sp.kron(mat[j], tmpGrad1)
                    
            allMat[i] = tmpGrad # diagonal elements H22, H11
            if direction == 'for':
                allMat1[i+len(discStep)] = tmpGrad1 # rest elements H12, H21
            elif direction == 'back':
                allMat1[i+len(discStep)] = tmpGrad1.T
                                               
        allMat.update(allMat1)

    return allMat #######  H22, H11, H12, H21  


if __name__ == '__main__':
    
    
    # Compare Forward difference with Sparse Matrices represent the finite differences
    
    #2D  forward
    u = ImageData(np.random.randint(10, size = (4,5)))
    
    forGrad = GradOper(u.shape, [1]*len(u.shape), direction = 'for', order = '1', bndrs = 'Neumann')
    
    grad_sparse = np.zeros((len(u.shape),)+u.shape)
    for i in range(len(u.shape)):
        grad_sparse[i] = np.reshape(forGrad[i]*u.as_array().flatten('F'), u.shape, 'F')
            
    G = Gradient(u.shape)
    
    z = G.direct(u)
    
    np.testing.assert_array_equal(grad_sparse, z.as_array())
    
    #3D forward
    u1 = ImageData(np.random.randint(10, size = (2,2,3)))
    
    forGrad1 = GradOper(u1.shape, [1]*len(u1.shape), direction = 'for', order = '1', bndrs = 'Periodic')
    
    grad_sparse1 = np.zeros((len(u1.shape),)+u1.shape)
    for i in range(len(u1.shape)):
        grad_sparse1[i] = np.reshape(forGrad1[i]*u1.as_array().flatten('F'), u1.shape, 'F')
            
    G1 = Gradient(u1.shape, bnd_cond='Periodic')
    
    z1 = G1.direct(u1)
    
    np.testing.assert_array_equal(grad_sparse1, z1.as_array())  
    
    
    # Compute preconditioning sigma, tau 
    # ref: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.381.6056&rep=rep1&type=pdf
    
    u2 = ImageData(np.random.randint(10, size = (2,3)))
    
    forGrad2 = GradOper(u2.shape, [1]*len(u2.shape), direction = 'for', order = '1', bndrs = 'Neumann')
    
    tau = np.zeros(u2.shape)
    sigma = np.zeros((len(u2.shape),)+u2.shape)
    for i in range(len(forGrad2)):        
        tau +=  np.reshape(np.abs(forGrad2[i]).sum(axis=0), u2.shape, 'F')
        sigma[i] = np.reshape(np.abs(forGrad2[i]).sum(axis=1), u2.shape, 'F')
    
    
    
    
    
    
    
    
    
    
    

