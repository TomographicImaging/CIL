#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

import numpy
from ccpi.optimisation.operators import *
from ccpi.optimisation.algorithms import *
from ccpi.optimisation.functions import *
from ccpi.framework import *

# Problem dimension.
m = 200
n = 200

numpy.random.seed(10)

# Create matrix A and data b
Amat = numpy.asarray( numpy.random.randn(m, n), dtype = numpy.float32)
bmat = numpy.asarray( numpy.random.randn(m), dtype=numpy.float32)

# Geometries for data and solution
vgb = VectorGeometry(m)
vgx = VectorGeometry(n)

b = vgb.allocate(0, dtype=numpy.float32)
b.fill(bmat)
A = LinearOperatorMatrix(Amat)

# Setup and run CGLS
x_init = vgx.allocate()

cgls = CGLS(x_init = x_init, operator = A, data = b)
cgls.max_iteration = 2000
cgls.update_objective_interval = 200
cgls.run(2000, verbose = True)

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False

# Compare with CVX
x = Variable(n)
obj = Minimize(sum_squares(A.A*x - b.as_array()))
prob = Problem(obj)

# choose solver
if 'MOSEK' in installed_solvers():
    solver = MOSEK
else:
    solver = SCS 

result = prob.solve(solver = MOSEK)


diff_sol = x.value - cgls.get_output().as_array()
 
print('Error |CVX - CGLS| = {}'.format(numpy.sum(numpy.abs(diff_sol))))
print('CVX objective = {}'.format(obj.value))
print('CGLS objective = {}'.format(cgls.objective[-1]))




