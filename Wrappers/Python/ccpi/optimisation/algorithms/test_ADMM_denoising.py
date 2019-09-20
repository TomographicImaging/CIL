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

Total Variation Denoising using PDHG algorithm:


Problem:     min_{u},   \alpha * ||\nabla u||_{2,1} + Fidelity(u, g)

             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data 
                          
             Fidelity =  1) L2NormSquarred ( \frac{1}{2} * || u - g ||_{2}^{2} ) if Noise is Gaussian
                         2) L1Norm ( ||u - g||_{1} )if Noise is Salt & Pepper
                         3) Kullback Leibler (\int u - g * log(u) + Id_{u>0})  if Noise is Poisson
                                                       
             Method = 0 ( PDHG - split ) :  K = [ \nabla,
                                                 Identity]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = \nabla    
             
             
             Default: ROF denoising
             noise = Gaussian
             Fidelity = L2NormSquarred 
             method = 0
             
             
"""

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os
import sys

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
    n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    scale = 5
    n1 = TestData.random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ig.allocate()
noisy_data.fill(n1)

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
    alpha = 1
elif noise == 'gaussian':
    alpha = .003

# fidelity
if noise == 's&p':
    f2 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f2 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f2 = 0.5 * L2NormSquared(b=noisy_data)

if method == '0':

    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f = BlockFunction(alpha * MixedL21Norm(), f2) 
    g = ZeroFunction()
    
else:
    
    pass
#    operator = Gradient(ig)
#    f =  alpha * MixedL21Norm()
#    g = f2
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 200
pdhg.update_objective_interval = 100
pdhg.run(2000)

#%%

#g1new = f2
#g2new = alpha * MixedL21Norm()
gnew = f
fnew = g

sigma_new = 1
tau_new = sigma_new/normK**2

x = operator.domain_geometry().allocate()
z = operator.range_geometry().allocate()
u = operator.range_geometry().allocate()

x0 = operator.domain_geometry().allocate()
z0 = operator.range_geometry().allocate()
u0 = operator.range_geometry().allocate()

x1 = operator.domain_geometry().allocate()
z1 = operator.range_geometry().allocate()
u1 = operator.range_geometry().allocate()

tmp1 = operator.range_geometry().allocate()
tmp3 = operator.range_geometry().allocate()
tmp2 = operator.domain_geometry().allocate()

for i in range(5000):
    
    
#    tmp = x0 - (tau_new/sigma_new) * operator.adjoint(operator.direct(x0) - z0 + u0)
#    x1 = fnew.proximal(tmp, tau_new)
#    
#    tmp1 = operator.direct(x1) + u0
#    z1 = gnew.proximal(tmp1, sigma_new)
#    
#    u1 = u0 + operator.direct(x1) - z1
    
    
    operator.direct(x0, out = tmp1)
    tmp1.add(u0, out = tmp1) 
    tmp1.add(-1 * z0, out = tmp1)
    
    operator.adjoint(tmp1, out = tmp2)
    
    fnew.proximal( x0 - (tau_new/sigma_new) * tmp2, tau_new, out = x1)
    
    operator.direct(x1, out = tmp3)
    tmp3 += u0

    gnew.proximal(tmp3, sigma_new, out = z1)

    operator.direct(x1, out = u1)
    u1 += u0
    u1 -= z1

    u0.fill(u1)
    z0.fill(z1)
    x0.fill(x1)
        
    if i%200 == 0:
        print(i)
        plt.imshow(x0.as_array())
        plt.colorbar()
        plt.show()

#%%
        
plt.figure(figsize=(15,15))

plt.subplot(4,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()

plt.subplot(4,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()

plt.subplot(4,1,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('Tikhonov Reconstruction')
plt.colorbar()

plt.subplot(4,1,4)
plt.imshow(x0.as_array())
plt.title('Tikhonov Reconstruction')
plt.colorbar()

plt.show()

plt.imshow(np.abs(x0.as_array() - pdhg.get_output().as_array()))
plt.colorbar()
plt.show()


#plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(N/2),:], label = 'Tikhonov reconstruction')
#plt.legend()
#plt.title('Middle Line Profiles')
#plt.show()        
        
        
