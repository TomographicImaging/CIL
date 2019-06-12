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

""" 

"Tikhonov regularization" for Poisson denoising using FISTA algorithm:

Problem:     min_x, x>0  \alpha * ||\nabla x||_{2}^{2} + \int x - g * log(x) 

             \alpha: Regularization parameter
             
             \nabla: Gradient operator  
             
             g: Noisy Data with Poisson Noise
               
                       
"""

from ccpi.framework import ImageData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import FISTA

from ccpi.optimisation.operators import Gradient
from ccpi.optimisation.functions import KullbackLeibler, L2NormSquared, FunctionOperatorComposition

from ccpi.framework import TestData
import os, sys
from skimage.util import random_noise

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))

# Load Data                      
N = 100
M = 100
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))

ig = data.geometry
ag = ig

# Create Noisy data with Poisson noise
n1 = random_noise(data.as_array(), mode = 'poisson', seed = 10)
noisy_data = ImageData(n1)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Regularisation Parameter
alpha = 10

# Setup and run the FISTA algorithm
operator = Gradient(ig)
fid = KullbackLeibler(noisy_data)

def KL_Prox_PosCone(x, tau, out=None):
        
    if out is None: 
        tmp = 0.5 *( (x - fid.bnoise - tau) + ( (x + fid.bnoise - tau)**2 + 4*tau*fid.b   ) .sqrt() )
        return tmp.maximum(0)
    else:            
        tmp =  0.5 *( (x - fid.bnoise - tau) + 
                    ( (x + fid.bnoise - tau)**2 + 4*tau*fid.b   ) .sqrt()
                    )
        x.add(fid.bnoise, out=out)
        out -= tau
        out *= out
        tmp = fid.b * (4 * tau)
        out.add(tmp, out=out)
        out.sqrt(out=out)
        
        x.subtract(fid.bnoise, out=tmp)
        tmp -= tau
        
        out += tmp
        
        out *= 0.5
        
        # ADD the constraint here
        out.maximum(0, out=out)
        
fid.proximal = KL_Prox_PosCone

reg = FunctionOperatorComposition(alpha * L2NormSquared(), operator)

x_init = ig.allocate()
fista = FISTA(x_init=x_init , f=reg, g=fid)
fista.max_iteration = 2000
fista.update_objective_interval = 500
fista.run(2000, verbose=True)

# Show results
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(fista.get_output().as_array())
plt.title('Reconstruction')
plt.colorbar()
plt.show()

plt.plot(np.linspace(0,N,M), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,M), fista.get_output().as_array()[int(N/2),:], label = 'Reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False


if cvx_not_installable:

    ##Construct problem    
    u1 = Variable(ig.shape)
    q = Variable()
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    regulariser = alpha * sum_squares(norm(vstack([DX.matrix() * vec(u1), DY.matrix() * vec(u1)]), 2, axis = 0))    
    fidelity = sum(kl_div(noisy_data.as_array(), u1))  
    
    constraints = [q>=fidelity, u1>=0]    
    
    solver = SCS
    obj =  Minimize( regulariser +  q)
    prob = Problem(obj, constraints)
    result = prob.solve(verbose = True, solver = solver)
    
    diff_cvx = numpy.abs( fista.get_output().as_array() - u1.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(fista.get_output().as_array())
    plt.title('FISTA solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(u1.value)
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,N,M), fista.get_output().as_array()[int(N/2),:], label = 'FISTA')
    plt.plot(np.linspace(0,N,M), u1.value[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (FISTA) {} '.format(fista.loss[1]))



