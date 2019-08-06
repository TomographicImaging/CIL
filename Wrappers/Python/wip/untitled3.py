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

import cvxpy as cp
import numpy as np
from ccpi.optimisation.operators import *
from ccpi.optimisation.algorithms import *
from ccpi.optimisation.functions import *
from ccpi.framework import *

# Problem dimension.
m = 400
n = 200

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


# Setuo and run CGLS
x_init = vgx.allocate()

cgls = CGLS(x_init = x_init, operator = A, data = b)
cgls.max_iteration = 1000
cgls.update_objective_interval = 20

cgls.run(100, verbose = True)

# Compare with CVX
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A.A*x - b.as_array()))
prob = cp.Problem(objective)
result = prob.solve(solver = cp.SCS)


print('Error = {}'.format((cgls.get_output() - VectorData(x.value)).norm()))



