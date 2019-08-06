

from ccpi.framework import *
import numpy as np
from cvxpy import *

m = 500
n = 1500

A = np.random.rand(m, n)
b = np.random.rand(m)

x = np.zeros(n)
r = b - np.dot(A,x)
s = np.dot(A.T, r)
#%%

# initialize

p = s
norms0 = VectorData(s).norm()
gamma = norms0**2
normx = VectorData(x).norm()
xmax = normx
it = 1000
for _ in range(it):
    
    q = np.dot(A, p)
    delta = VectorData(q).squared_norm() 
    alpha = gamma/delta
    
    x += alpha * p
    r -= alpha * q
    
    s = np.dot(A.T, r)
    
    norms = VectorData(s).norm()
    gamma1 = gamma
    gamma = norms**2
    beta = gamma/gamma1
    p = s + beta * p
    
    normx = VectorData(x).norm()
    xmax = np.maximum(xmax, normx)
    if gamma<=1e-12:
        break
    
    
#%%    
x1 = Variable(n)        
obj =  Minimize( sum_squares(A*x1 - b))
prob = Problem(obj)
result = prob.solve(verbose = True, solver = MOSEK)    

#%%

print(VectorData(x1.value - x).squared_norm())

