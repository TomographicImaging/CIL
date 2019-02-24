#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:44:05 2019

@author: evangelos
"""

import scipy.sparse as sp
import numpy as np

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
    
    u = ImageData(np.random.randint(10, size = (3,3)))
    forGrad = GradOper(u.shape, [1]*len(u.shape), direction = 'for', order = '1', bndrs = 'Neumann')
    
    
    
    

