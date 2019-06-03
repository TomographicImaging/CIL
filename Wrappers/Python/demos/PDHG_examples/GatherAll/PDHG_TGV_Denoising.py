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

Total Generalised Variation (TGV) Denoising using PDHG algorithm:


Problem:     min_{u} \alpha * ||\nabla u - w||_{2,1} +
                     \beta *  || E u ||_{2,1} +
                     Fidelity(u, g)
                     
             \nabla: Gradient operator 
             E: Symmetrized Gradient operator             
             \alpha: Regularization parameter
             \beta:  Regularization parameter             
             
             g: Noisy Data 
                          
             Fidelity =  1) L2NormSquarred ( \frac{1}{2} * || u - g ||_{2}^{2} ) if Noise is Gaussian
                         2) L1Norm ( ||u - g||_{1} )if Noise is Salt & Pepper
                         3) Kullback Leibler (\int u - g * log(u) + Id_{u>0})  if Noise is Poisson
                                
                          
             Method = 0 ( PDHG - split ) :  K = [ \nabla, - Identity
                                                  ZeroOperator, E 
                                                  Identity, ZeroOperator]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = [ \nabla, - Identity
                                                  ZeroOperator, E ] 
                          
             Default: TGV denoising
             noise = Gaussian
             Fidelity = L2NormSquarred 
             method = 0             
                                                                
"""

from ccpi.framework import ImageData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, \
                        Gradient, SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, L2NormSquared
                      
from ccpi.framework import TestData
import os, sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
print ("Applying {} noise")

if len(sys.argv) > 2:
    method = sys.argv[2]
else:
    method = '0'
print ("method ", method)


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SHAPES)
ig = data.geometry
ag = ig

# Create noisy data. 
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2, seed=10)
elif noise == 'poisson':
    scale = 5
    n1 = random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ImageData(n1)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Regularisation Parameter depending on the noise distribution
if noise == 's&p':
    alpha = 0.8
elif noise == 'poisson':
    alpha = .3
elif noise == 'gaussian':
    alpha = .2

# TODO add ref why this choice
beta = 2 * alpha

# Fidelity
if noise == 's&p':
    f3 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f3 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f3 = 0.5 * L2NormSquared(b=noisy_data)

if method == '0':
    
    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())
        
    op31 = Identity(ig, ag)
    op32 = ZeroOperator(op22.domain_geometry(), ag)
    
    operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
        
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm() 
    
    f = BlockFunction(f1, f2, f3)         
    g = ZeroFunction()
        
else:
    
    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())    
    
    operator = BlockOperator(op11, -1*op12, op21, op22, shape=(2,2) )      
    
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm()     
    
    f = BlockFunction(f1, f2)         
    g = BlockFunction(f3, ZeroFunction())
     
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 100
pdhg.run(2000)

# Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.subset(channel=0).as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.subset(channel=0).as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(pdhg.get_output()[0].as_array())
plt.title('TGV Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output()[0].as_array()[int(ig.shape[0]/2),:], label = 'TGV reconstruction')
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
    
    u = Variable(ig.shape)
    w1 = Variable(ig.shape)
    w2 = Variable(ig.shape)
    
    # create TGV regulariser
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u) - vec(w1), \
                                           DY.matrix() * vec(u) - vec(w2)]), 2, axis = 0)) + \
                  beta * sum(norm(vstack([ DX.matrix().transpose() * vec(w1), DY.matrix().transpose() * vec(w2), \
                                      0.5 * ( DX.matrix().transpose() * vec(w2) + DY.matrix().transpose() * vec(w1) ), \
                                      0.5 * ( DX.matrix().transpose() * vec(w2) + DY.matrix().transpose() * vec(w1) ) ]), 2, axis = 0  ) )  
    
    constraints = []
    
    # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS      
    
    # fidelity
    if noise == 's&p':
        fidelity = pnorm( u - noisy_data.as_array(),1)
    elif noise == 'poisson':
        fidelity = sum(kl_div(noisy_data.as_array(), u)) 
        solver = SCS
    elif noise == 'gaussian':
        fidelity = 0.5 * sum_squares(noisy_data.as_array() - u)
        
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)
    
    diff_cvx = numpy.abs( pdhg.get_output()[0].as_array() - u.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output()[0].as_array())
    plt.title('PDHG solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(u.value)
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output()[0].as_array()[int(ig.shape[0]/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), u.value[int(ig.shape[0]/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))