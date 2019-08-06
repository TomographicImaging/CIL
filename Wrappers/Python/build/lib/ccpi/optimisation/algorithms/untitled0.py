#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:51:31 2019

@author: evangelos
"""

#%%
import numpy as np
from ccpi.optimisation.operators import *
from ccpi.optimisation.algorithms import *
from ccpi.optimisation.functions import *
from ccpi.framework import *

#Problem dimension.
m = 1000
n = 200

np.random.seed(10)
# Create matrix A and data b
Amat = np.asarray( np.random.randn(m, n), dtype=numpy.float32)
bmat = np.asarray( np.random.randn(m), dtype=numpy.float32)

# Geometries for data and solution
vgb = VectorGeometry(m)
vgx = VectorGeometry(n)

b = vgb.allocate(0, dtype=numpy.float32)
b.fill(bmat)
A = LinearOperatorMatrix(Amat)

print('Rank of A is {} '.format(np.linalg.matrix_rank(Amat)))


# Setup and run CGLS
x_init = vgx.allocate()

cgls = CGLS(x_init = x_init, operator = A, data = b)
cgls.max_iteration = 200
cgls.update_objective_interval = 20
cgls.run(200, verbose = True)


# Run FISTA for least squares with lower/upper bound 

f = FunctionOperatorComposition(L2NormSquared(b=b), A)

fista0 = FISTA(x_init = x_init, f = f, g = ZeroFunction())
fista0.max_iteration = 2000
fista0.update_objective_interval = 300
fista0.run(2000, verbose=True)

#%%





#
#try:
#    import cvxpy as cp
#    cvx = True
#except ImportError:
#    cvx = False
#
#if not cvx:
#    print("Please install the cvxpy module to run this demo")
#else:
#    # Compare with CVX
#    x = cp.Variable(n)
#    objective = cp.Minimize(cp.sum_squares(A.A*x - b.as_array()))
#    prob = cp.Problem(objective)
#    result = prob.solve(solver = cp.SCS)
#
#    print('Error = {}'.format((cgls.get_output() - VectorData(np.asarray(x.value).T[0])).norm()))


#%%