import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction,L2NormSquared, weighted_L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os
import sys

# Poisson denoising with PDHG and using weighted LS
# See  1) A. Sawatzky. (Nonlocal) Total Variation in Medical Imaging. PhD thesis, University of
#         Munster, 2011. CAM Report 11-47, UCLA.
#      2) http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=836D77B8DF41CCDECA4949568C9F2D97?doi=10.1.1.394.3926&rep=rep1&type=pdf

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.CAMERA, size=(512, 512))
ig = data.geometry
ag = ig

scale = 5
#n1 = TestData.random_noise( data.as_array()/scale, mode = 'poisson', seed = 42, clip=False)*scale
n1 = TestData.random_noise( data.as_array(), mode = 'gaussian', seed = 42)
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
 
#%%
# Regularisation Parameter depending on the noise distribution
alpha = 1e5

# We need positive weight
weight = 1/(noisy_data)
weight.array[weight.array == np.inf] = 1e-12


# Create operators
op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE, num_threads=3)
op2 = Identity(ig, ag)

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions      
f2 = 0.5 * weighted_L2NormSquared(weight=weight, b=noisy_data)
#f2 = 0.5 * L2NormSquared(b=noisy_data)
f = BlockFunction(alpha * MixedL21Norm(), f2) 
g = ZeroFunction()
    
# Compute operator Norm
normK = operator.norm()
    
# Primal & dual stepsizes
sigma = 100
tau = 1/(sigma*normK**2)
    
    
# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, 
            max_iteration = 2000,
            update_objective_interval = 500)
pdhg.run(verbose=True, very_verbose = True)
    

# Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('TV Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()
    
        
