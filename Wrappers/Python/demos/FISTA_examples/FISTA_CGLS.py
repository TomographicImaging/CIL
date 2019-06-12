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

from ccpi.framework import AcquisitionGeometry
from ccpi.optimisation.algorithms import FISTA
from ccpi.optimisation.functions import IndicatorBox, ZeroFunction, \
                         L2NormSquared, FunctionOperatorComposition
from ccpi.astra.operators import AstraProjectorSimple

import numpy as np
import matplotlib.pyplot as plt
from ccpi.framework import TestData
import os, sys
from ccpi.optimisation.funcs import Norm2sq

# Load Data 
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
                     
N = 50
M = 50
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))

ig = data.geometry

# Show Ground Truth 
plt.figure(figsize=(5,5))
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.show()

#%%
# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam and 0 to 2pi for fanbeam, set the width of a detector 
# pixel relative to an object pixel, the number of detector pixels, and the 
# source-origin and origin-detector distance (here the origin-detector distance 
# set to 0 to simulate a "virtual detector" with same detector pixel size as
# object pixel size).

#device = input('Available device: GPU==1 / CPU==0 ')

#if device=='1':
#    dev = 'gpu'
#else:
#    dev = 'cpu'

test_case = 1
dev = 'cpu'

if test_case==1:
    
    detectors = N
    angles_num = N    
    det_w = 1.0
    
    angles = np.linspace(0, np.pi, angles_num, endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             detectors,det_w)
  
elif test_case==2:
    
    SourceOrig = 200
    OrigDetec = 0
    angles = np.linspace(0,2*np.pi,angles_num)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             detectors,
                             det_w,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)
else:
    NotImplemented

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.    
Aop = AstraProjectorSimple(ig, ag, dev)
    
# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
sin = Aop.direct(data)
back_proj = Aop.adjoint(sin)

plt.figure(figsize=(5,5))
plt.imshow(sin.array)
plt.title('Simulated data')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(back_proj.array)
plt.title('Backprojected data')
plt.show()

#%%

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:

f = FunctionOperatorComposition(L2NormSquared(b=sin), Aop)
#f = Norm2sq(Aop, sin, c=0.5, memopt=True)
g = ZeroFunction()

x_init = ig.allocate()
fista = FISTA(x_init=x_init, f=f, g=g)
fista.max_iteration = 1000
fista.update_objective_interval = 100
fista.run(1000, verbose=True)

plt.figure()
plt.imshow(fista.get_output().as_array())
plt.title('FISTA unconstrained')
plt.colorbar()
plt.show()


# Run FISTA for least squares with lower/upper bound 
fista0 = FISTA(x_init=x_init, f=f, g=IndicatorBox(lower=0,upper=1))
fista0.max_iteration = 1000
fista0.update_objective_interval = 100
fista0.run(1000, verbose=True)

plt.figure()
plt.imshow(fista0.get_output().as_array())
plt.title('FISTA constrained in [0,1]')
plt.colorbar()
plt.show()

#plt.figure()
#plt.semilogy(fista0.objective)
#plt.title('FISTA constrained in [0,1]')
#plt.show()

#%% Check with CVX solution

import astra
import numpy

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False
    
if cvx_not_installable:
    
    ##Construct problem    
    u = Variable(N*N)

    # create matrix representation for Astra operator
    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)

    proj_id = astra.create_projector('linear', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)
    
    ProjMat = astra.matrix.get(matrix_id)
    
    tmp = sin.as_array().ravel()
    
    fidelity = 0.5 * sum_squares(ProjMat * u - tmp)

    solver = MOSEK
    obj =  Minimize(fidelity)
    constraints = [u>=0, u<=1]
    prob = Problem(obj, constraints=constraints)
    result = prob.solve(verbose = True, solver = solver)   
         
    diff_cvx = numpy.abs( fista0.get_output().as_array() - np.reshape(u.value, (N,M) ))
           
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(fista0.get_output().as_array())
    plt.title('FISTA solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(np.reshape(u.value, (N, M)))
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,N,M), fista0.get_output().as_array()[int(N/2),:], label = 'FISTA')
    plt.plot(np.linspace(0,N,M), np.reshape(u.value, (N,M) )[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (FISTA) {} '.format(fista0.loss[1]))    