#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:26:06 2018

@author: evangelos
"""

import scipy.sparse as sp
import numpy as np
from cvxpy import *


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
    
    # TODO central differences
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

                if bndrs == 'Neum':
                    mat[-1,:] = 0
                elif bndrs == 'Per':
                    mat[-1,0] = 1

            elif direction == 'back':
                
                mat = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [-1,0], size[i], size[i], format = 'lil')

                if bndrs == 'Neum':
                    mat[:,-1] = 0
                elif bndrs == 'Per':
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
            
            if bndrs == 'Neum':
                mat[i][-1,:] = 0 # zero last row
                mat1[i][:,0] = 0 # zero first col
                
            if bndrs == 'Per':
                mat[i][-1,0] = 1 # zero last row
                mat1[i][-1,0] = 1 # zero first col                 
                                                  
        for i in range(0,len(discStep)):
                        
            if bndrs == 'Neum':
                tmpGrad1 = -mat1[i].T                 
                tmpGrad = -mat[i].T * mat[i] if i == 0 else sp.eye(size[0])
            
                for j in range(1, len(discStep)):
                    tmpGrad = sp.kron(-mat[j].T * mat[j], tmpGrad ) if j == i else sp.kron(sp.eye(size[j]), tmpGrad )
                    tmpGrad1 = sp.kron(tmpGrad1,mat[j-1]) if j == i else sp.kron(mat[j], tmpGrad1)
                                                  
            if bndrs == 'Per': 
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
                
def TV_cvx(u):
        
    discStep = np.ones(len(u.shape))
    tmp = GradOper(u.shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')    
    DX, DY = tmp[1], tmp[0]
  
    return sum(norm(vstack([DX * vec(u), DY * vec(u)]), 2, axis = 0))

def L2GradientSq(u):
        
    discStep = np.ones(len(u.shape))
    tmp = GradOper(u.shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')    
    DX, DY = tmp[1], tmp[0]
  
    return sum_squares(norm(vstack([DX * vec(u), DY * vec(u)]), 2, axis = 0))

def tv2(u):
        
    discStep = np.ones(len(u.shape))
    tmp = GradOper(u.shape, [1,1], 'for', '2', 'Neum')
    H11, H22, H12, H21 = tmp[1], tmp[0], tmp[2], tmp[3]
    return sum(norm(vstack([H11 * vec(u), H22 * vec(u), H12 * vec(u), H21 * vec(u)]), 2, axis = 0))

def ictv(u, v, alpha0, alpha1):
    
    discStep = np.ones(len(u.shape))
    tmp0 = GradOper(u.shape, discStep, 'for', '1', 'Neum') 
    tmp1 = GradOper(u.shape, discStep, 'for', '2', 'Neum')
    DX, DY = tmp0[1], tmp0[0]
    H11, H22, H12, H21 = tmp1[1], tmp1[0], tmp1[2], tmp1[3]
    return alpha0 * sum(norm(vstack([DX * vec(u), DY * vec(u)]), 2, axis = 0))  + \
           alpha1 * sum(norm(vstack([H11 * vec(v), H22 * vec(v), H12 * vec(v), H21 * vec(v)]), 2, axis = 0))
    

def tv_funVec(variable_cvx, method):
    
#    Ref: See "Collaborative Total Variation: A General Framework for Vectorial TV Models"
        
    if method == 'aniso':
        tmp = 0
        for i in range(0,len(variable_cvx)):
            tmp += tv_fun(variable_cvx[i])
        return tmp
    
    if  method == 'iso':
        tmp = []
        for i in range(0, len(variable_cvx)):
            tmp.append(tv_fun(variable_cvx[i]))      
        return sum( norm(vstack(tmp), 2, axis = 0 ))
    
    if method == 'isol2':
        
        discStep = [1, 1]
        grad = GradOper(variable_cvx[0].shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')            
        
        tmp = []
        for i in range(0, len(variable_cvx)):
            for j in range(0, len(grad)):
                tmp.append(grad[j]*vec(variable_cvx[i]))
        return sum(norm(vstack(tmp), 2, axis = 0))
    
    if method == 'anis_linf':
        
        discStep = [1, 1]
        grad = GradOper(variable_cvx[0].shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')            
                
        a = 0
        for j in range(0, len(grad)):
            tmp = []
            for i in range(0, len(variable_cvx)):
                tmp.append(grad[j]*vec(variable_cvx[i]))
            a += sum(norm ( vstack(tmp), 'inf')) 
            
        return a      
        
        
def tgv(u, w1, w2, alpha0, alpha1):

    discStep = np.ones(len(u.shape))
    tmp = GradOper(u.shape, discStep, direction = 'for', order = '1', bndrs = 'Neum') 
    DX, DY = tmp[1], tmp[0]
    
    tmp1 = GradOper(u.shape, discStep, direction = 'back', order = '1', bndrs = 'Neum')    
    divX, divY = tmp1[1], tmp1[0]
  
    return alpha0 * sum(norm(vstack([DX * vec(u) - vec(w1), DY * vec(u) - vec(w2)]), 2, axis = 0)) + \
           alpha1 * sum(norm(vstack([ divX * vec(w1), divY * vec(w2), \
                                      0.5 * ( divX * vec(w2) + divY * vec(w1) ), \
                                      0.5 * ( divX * vec(w2) + divY * vec(w1) ) ]), 2, axis = 0  ) )
