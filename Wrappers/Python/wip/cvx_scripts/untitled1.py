#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:11:44 2018

@author: evangelos
"""

import numpy as np
import mosek
import matplotlib.pyplot as plt
from cvxpy import *
from ccpi.optimisation.funcs import Norm2sq, ZeroFun, Norm1, TV2D, Norm2
from ccpi.optimisation.ops import FiniteDiff2D
import scipy.sparse as sp
import matplotlib.pyplot as plt
from math import sqrt


#%%

N = 64
x = np.zeros((N,N))
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
plt.show()

y = x + 0.2*np.random.randn(N, N)

fig = plt.figure()
im = plt.imshow(y)
plt.title('Noisy image')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()
#%%

def gradOper(img):    
    n, m = np.shape(img)
    
    data1 = np.zeros((2,n))
    data1[0,:] = - np.ones((1,n))
    
    data1[1,:] =   np.ones((1,n))

    data2 = np.zeros((2,m))
    data2[0,:] = - np.ones((1,m))
    data2[1,:] =   np.ones((1,m))

    Dx = sp.spdiags(data1, [0,1], n, n, format = 'lil')
    Dx[-1] = 0
    DX = sp.kron(np.eye(m),Dx)

    Dy = sp.spdiags(data2, [0,1], m, m, format = 'lil')
    Dy[-1] = 0
    DY = sp.kron(Dy,np.eye(n))  
                
    return DX, DY

#%%
    

N = 2
img1 = np.random.randint(5,size = (N,N))

#%%
DX, DY = gradOper(img1)
a = (DX * img1.flatten('F')).reshape((N,N))
b = (DY * img1.flatten('F')).reshape((N,N))
stacked1 = np.concatenate((a, b), axis=0)  
sum(norm(stacked1, p=2, axis=1)).value

   
DX, DY = gradOper(img1)
a = (DX * img1.flatten())
b = (DY * img1.flatten())
np.sum(np.sqrt(a**2 + b**2))


#Z = (DX * img1.flatten('F')).reshape((N,N))
#(DY * img1.flatten('F')).reshape((N,N)) 

#%% 

def gradOper(img):    
    n, m = np.shape(img)
    
    data1 = np.zeros((2,n))
    data1[0,:] = - np.ones((1,n))
    
    data1[1,:] =   np.ones((1,n))

    data2 = np.zeros((2,m))
    data2[0,:] = - np.ones((1,m))
    data2[1,:] =   np.ones((1,m))

    Dx = sp.spdiags(data1, [0,1], n, n, format = 'lil')
    Dx[-1] = 0
    DX = sp.kron(np.eye(m),Dx)

    Dy = sp.spdiags(data2, [0,1], m, m, format = 'lil')
    Dy[-1] = 0
    DY = sp.kron(Dy,np.eye(n))  
                
    return DX, DY


def tv_function(img1):
    
    DX, DY = gradOper(img1)
    
    c = vec(img1)
    a = (DX * c)
    b = (DY * c)
    
    return sum(norm(vstack([a, b]), 2, axis = 0))




#%%



use_cvxpy = True
lam1_denoise = 10
N = 64


if use_cvxpy:
    # Compare to CVXPY
    
    # Construct the problem.
    x1_denoise = Variable((N,N))
    
#    grads = hstack( [DX * vec(x1_denoise),DY * vec(x1_denoise)])
    objective1_denoise =  Minimize(0.5*sum_squares(x1_denoise - y) +  lam1_denoise * my_tv(x1_denoise) )
#    objective1_denoise =  Minimize(lam1_denoise* tv(x1_denoise) + 5)
                         
#    Minimize(0.5*sum_squares(x1_denoise - y) +
    
    prob1_denoise = Problem(objective1_denoise)

    # The optimal objective is returned by prob.solve().
#    result1_denoise = prob1_denoise.solve(verbose=False,solver=SCS,eps=1e-12)

    result1_denoise = prob1_denoise.solve(verbose=True,solver=MOSEK)
    
    # The optimal solution for x is stored in x.value and optimal objective value 
    # is in result as well as in objective.value
    print("CVXPY least squares plus 1-norm solution and objective value:")
    print(x1_denoise.value)
    print(objective1_denoise.value)



x1_cvx = x1_denoise.value


x1_cvx.shape = (N,N)

fig = plt.figure()
im = plt.imshow(x1_cvx)
plt.title('Noisy image')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()



#%%

value = img1
rows, cols = value.shape
values = [value] 
diffs = []
for mat in values:
  diffs += [
                mat[0:rows-1, 1:cols] - mat[0:rows-1, 0:cols-1],
                mat[1:rows, 0:cols-1] - mat[0:rows-1, 0:cols-1],
            ]
length = diffs[0].shape[0]*diffs[1].shape[1]
stacked = vstack([reshape(diff, (1, length)) for diff in diffs])
sum(norm(stacked, p=2, axis=0)).value













