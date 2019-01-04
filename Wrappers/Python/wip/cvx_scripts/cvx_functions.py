#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:26:06 2018

@author: evangelos
"""

import scipy.sparse as sp
import numpy as np
from cvxpy import *


def GradOper(size, discStep, direction = 'for', order = '1', bndrs = 'Neum'):  

    allMat = dict.fromkeys((range(len(discStep))))
        
    if len(size)!=len(discStep):
        raise ValueError('Check dimensions of discStep and input size') 

    if 0 in discStep:
        raise ValueError('Zero step size')
                           
    # First order gradient arrays, forward/backward for Neummann, Periodic conditions
                                      
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

            for j in range(1, len(size)):

                tmpGrad = sp.kron(mat, tmpGrad ) if j == i else sp.kron(sp.eye(size[j]), tmpGrad )

            allMat[i] = tmpGrad

    elif order == '2':
        pass
#        if len
    
    
        #TODO works for 1D/2D only        
#        mat = {}
#        
#        for i in range(0,len(discStep)):
#            
#            if direction == 'for':
#                mat[i] = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')
#            elif direction == 'back':
#                mat[i] = 1/discStep[i] * sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [-1,0], size[i], size[i], format = 'lil')
#                
#            if bndrs == 'Neum':
#                mat[-1,:] = 0
#            elif bndrs == 'Per':
#                mat[-1,0] = 1
#
#
#            if bndrs == 'Neum':
#                mat[:,-1] = 0
#            elif bndrs == 'Per':
#                mat[0,-1] = -1
#
#            tmpGrad = -mat.T*mat if i == 0 else sp.eye(size[0])
#            tmpGrad1 = -mat.T*mat if i == 0 else sp.eye(size[0])
#
#            for j in range(1, len(size)):
#
#                tmpGrad = sp.kron(-mat.T * mat, tmpGrad ) if j == i else sp.kron(sp.eye(size[j]), tmpGrad )
#
#            allMat[i] = tmpGrad
               
    return allMat


def tv_fun(u):
        
    discStep = np.ones(len(u.shape))
    tmp = GradOper(u.shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')    
    DX, DY = tmp[0], tmp[1]
  
    return sum(norm(vstack([DX * vec(u), DY * vec(u)]), 2, axis = 0))


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
        
        
def tgv_fun(w1, w2):

    discStep = np.ones(len(w1.shape))
    tmp = GradOper(w1.shape, discStep, direction = 'back', order = '1', bndrs = 'Neum')    
    divX, divY = tmp[0], tmp[1]
  
    return sum( norm( vstack([ divX * vec(w1), divY * vec(w2), \
        0.5 * ( divX * vec(w2) + divY * vec(w1) ), \
        0.5 * ( divX * vec(w2) + divY * vec(w1) ) ]), 2, axis = 0  ) )
        
    
    
#%%
#u = np.random.randint(10, size = (2,3))
#size = u.shape
#mat1 = []
#mat2 = []
#
#for i in range(0, len(u.shape)):
#    tmp1 = sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')
#    tmp1[-1,:] = 0
#    mat1.append(tmp1)
#    
#    tmp2 = sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [-1,0], size[i], size[i], format = 'lil')
#    tmp2[:,-1] = 0
#    mat2.append(tmp2)
#    
#    
#H11 = sp.kron(sp.eye(size[1]), -mat1[0].T * mat1[0])
#H22 = sp.kron(-mat1[1].T * mat1[1], sp.eye(size[0])) 
#
#H12 = sp.kron(-mat2[1], mat1[0].T)
#print(np.reshape(H12*u.flatten('F'), u.shape, 'F'))

#  np.reshape(grad[1]*u.flatten('F'), u.shape, 'F')
    