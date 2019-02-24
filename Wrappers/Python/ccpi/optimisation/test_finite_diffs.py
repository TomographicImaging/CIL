
from ccpi.framework import DataContainer, ImageData, ImageGeometry
import numpy as np
from operators import *

#%%
########################### FOR 2D  ###########################################
N, M = 2, 3
u = np.random.randint(20, size=(N,M))
w = np.random.randint(20, size=(len(u.shape),N,M))

DYu = finite_diff(u, direction = 0, method = 'for')
DXu = finite_diff(u, direction = 1, method = 'for')
divYw = finite_diff(w[0], direction = 0, method = 'back')
divXw = finite_diff(w[1], direction = 1, method = 'back')

print(np.sum(DYu * w[0] + DXu * w[1]  ))
print(np.sum(u * (divXw + divYw )))

#%%
########################### FOR 3D  ###########################################

N, M, C = 200, 300, 200
u1 = np.random.randint(10, size=(C, N, M))
w1 = np.random.randint(10, size=(3, C, N, M))

grad = np.zeros(w1.shape)

for i in range(len(u1.shape)):
    grad[i] = finite_diff(u1, direction = i, method = 'for')
    

DZu1 = finite_diff(u1, direction = 0, method = 'for')
DXu1 = finite_diff(u1, direction = 2, method = 'for')
DYu1 = finite_diff(u1, direction = 1, method = 'for')

divZw1 = finite_diff(w1[0], direction = 0, method = 'back')
divXw1 = finite_diff(w1[2], direction = 2, method = 'back')
divYw1 = finite_diff(w1[1], direction = 1, method = 'back')

print(np.sum(DZu1 * w1[0] + DXu1 * w1[2] + DYu1 * w1[1]  ))
print(np.sum(u1 * (divXw1 + divYw1 + divZw1)))
    
#ig = ImageGeometry(channels=C, voxel_num_y=M, voxel_num_x=N)    
#D = gradient(ig)
#t0 = time.time()
#D.direct(u1)
#t1 = time.time()
#print(t1-t0)

#%%
########################### sYM GRAD ##########################################

N, M, C = 200, 300, 200
#np.random.seed(10)
ig_sym_grad = ImageGeometry(voxel_num_x=M, voxel_num_y = N, channels=2)

#D = sym_gradient()
u = np.random.randint(10, size=(2,2,3))
w = np.random.randint(10, size=(3,2,3))

#Eu = D.direct(u)

Eu11 = finite_diff(u[0], direction = 1, method = 'back')
Eu22 = finite_diff(u[1], direction = 0, method = 'back')
Eu12 = 0.5 * (finite_diff(u[0], direction = 0, method = 'back') + \
              finite_diff(u[1], direction = 1, method = 'back') )

divE1 = finite_diff(w[0], direction = 1, method = 'for') + \
         finite_diff(w[2], direction = 0, method = 'for')
         
divE2 = finite_diff(w[2], direction = 1, method = 'for') + \
        finite_diff(w[1], direction = 0, method = 'for')        

print(np.sum( Eu11 * w[0] + Eu22 * w[1] +  2*Eu12 * w[2]))
print(-np.sum( u[0] * divE1 + u[1] * divE2))


D = sym_gradient(ig_sym_grad)
Eu = D.direct(ImageData(u))
divEw = D.adjoint(ImageData(w))

print(np.sum( Eu.as_array()[0] * w[0] + Eu.as_array()[1] * w[1] +  2*Eu.as_array()[2] * w[2]))
print(np.sum( u[0] * divEw.as_array()[0] + u[1] * divEw.as_array()[1]))

#%%


