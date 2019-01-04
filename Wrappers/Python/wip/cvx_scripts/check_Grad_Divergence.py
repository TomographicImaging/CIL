#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from cvx_functions import *
import numpy as np
import matplotlib.pyplot as plt

#%%

def check_divGrad(u, w, order, bndrs):
     
#    discStep = np.random.randint(1000, size = (len(u.shape)) )
    discStep = np.ones(len(u.shape ))
    
    forGrad = GradOper(u.shape, discStep, direction = 'for', order = order, bndrs = bndrs)
    backGrad = GradOper(u.shape, discStep, direction = 'back', order = order, bndrs = bndrs)

    Du = np.zeros((len(u.shape),) + u.shape)
    tmp_divw = np.zeros((len(u.shape),) + u.shape)

    for i in range(len(u.shape)):
        Du[i,:] = np.reshape(forGrad[i]*u.flatten('F'), u.shape, 'F')
        tmp_divw[i,:] = np.reshape(backGrad[i]*w[i].flatten('F'), u.shape, 'F')  

    divw = np.sum(tmp_divw,axis=0)
    
    if np.sum(np.multiply(Du,w)) + np.sum(np.multiply(u,divw)) < 1e-12:
        print()
        print('####################################')
        print( 'Passed the gradient/divergence test')
        print('####################################')
        print()      
        
    return Du, tmp_divw       
                                    
u = np.random.randint(10, size = (2,3,2) )
w = np.random.randint(10, size = ((len(u.shape),) + u.shape) )   

#Du, divw = check_divGrad(u, w, '1', 'Per')
Du, divw = check_divGrad(u, w, '1', 'Neum')



#%%

u = np.random.randint(10, size = (2,3))
size = u.shape

mat_for = []
mat_back = []

for i in range(len(u.shape)):
    tmp1 =  sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')        
    
    
    mat_for.append(tmp1)
    
    
    

H12 = sp.kron( -mat_for[1].T, mat_for[0] )
   
    
    
#%%
   
for i in range(len(u.shape)):
    tmp1 =  sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [0,1], size[i], size[i], format = 'lil')
    tmp1[:,0] = 0
    mat_for.append(tmp1)
    
#    tmp2 = sp.spdiags(np.vstack([-np.ones((1,size[i])),np.ones((1,size[i]))]), [-1,0], size[i], size[i], format = 'lil')
#    tmp2[-1,:] = 0
#    mat_back.append(tmp2)

















#%%

#backGrad = GradOper(u.shape, discStep, direction = 'back', order = '1', bndrs = 'Neum')

#%% Raise errors

#u = np.random.randint(5, size=(2, 4))
#discStep = [1,1]
#size = u.shape
#forGrad = GradOper(size, discStep, direction = 'for', order = '1', bndrs = 'Neum')
#backGrad = GradOper(size, discStep, direction = 'back', order = '1', bndrs = 'Neum')



#%% test 1D 
    
#u = np.random.randint(20, size = (4,1) )   
#discStep = [1,1]
#size = u.shape
#
#forGrad = GradOper(size, discStep, direction = 'for', order = '1', bndrs = 'Neum')
#backGrad = GradOper(size, discStep, direction = 'back', order = '1', bndrs = 'Neum')
#
#Dx_u = np.reshape(forGrad[0]*u.flatten('F'), size, 'F')
#divx_u = np.reshape(backGrad[0]*u.flatten('F'), size, 'F')
#
#print('Matrix is \n{}'.format(u))
#print()
#print('Forward diff for x \n{}'.format(Dx_u))
#print('Backward diff for x \n{}'.format(divx_u))
#
#w = np.random.randint(10, size = u.shape )
#divx_w = np.reshape(backGrad[0]*w.flatten('F'), size, 'F')
#
#print(np.sum(np.multiply(Dx_u,w)))
#print(np.sum(np.multiply(u,divx_w)))


#%% test 2D, 1st order



#%%




forGrad = GradOper(u.shape, discStep, direction = 'for', order = '1', bndrs = 'Neum')
backGrad = GradOper(u.shape, discStep, direction = 'back', order = '1', bndrs = 'Neum' )

Du = np.zeros((len(discStep),) + u.shape)
tmp_divw = np.zeros((len(discStep),) + u.shape)

for i in range(len(discStep)):
    Du[i,:] = np.reshape(forGrad[i]*u.flatten('F'), u.shape, 'F')
    tmp_divw[i,:] = np.reshape(backGrad[i]*w[i].flatten('F'), u.shape, 'F')  


divw = np.sum(tmp_divw,axis=0)

print(np.sum(np.multiply(Du,w)))
print(np.sum(np.multiply(u,divw)))

#%% test 1D, 2nd order

u = np.random.randint(10, size = (2,3,2) )  
discStep = [1,1,1]
size = u.shape 
w = np.random.randint(10, size = ((len(discStep),) + u.shape) ) 

forGrad = GradOper(size, discStep, direction = 'for', order = '2', bndrs = 'Neum')
backGrad = GradOper(size, discStep, direction = 'back', order = '2', bndrs = 'Neum')

Du = np.zeros((len(discStep),) + size)
tmp_divw = np.zeros((len(discStep),) + size)

for i in range(len(discStep)):
    Du[i,:] = np.reshape(forGrad[i]*u.flatten('F'), size, 'F')
    tmp_divw[i,:] = np.reshape(backGrad[i]*w[i].flatten('F'), size, 'F')  

divw = np.sum(tmp_divw,axis=0)

print(np.sum(np.multiply(Du,w)))
print(np.sum(np.multiply(u,divw)))


#%%










