
from ccpi.optimisation.algorithms import PDHG, CGLS, FISTA

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, \
                      MixedL21Norm, BlockFunction,L2NormSquared, IndicatorBox, WeightedL2NormSquared,\
                          KullbackLeibler, FunctionOperatorComposition
from ccpi.framework import TestData, BlockDataContainer
import os
import sys
from ccpi.optimisation.operators import DiagonalOperator   
import matplotlib.pyplot as plt
import numpy as np
    

# Poisson denoising with PDHG and using weighted LS
# See  1) A. Sawatzky. (Nonlocal) Total Variation in Medical Imaging. PhD thesis, University of
#         Munster, 2011. CAM Report 11-47, UCLA.
#      2) http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=836D77B8DF41CCDECA4949568C9F2D97?doi=10.1.1.394.3926&rep=rep1&type=pdf


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.CAMERA, size=(512, 512))
ig = data.geometry
ag = ig

scale = 10
n1 = TestData.random_noise( data.as_array()/scale, mode = 'poisson', seed = 10, clip=False) * scale
#n1 = TestData.random_noise( data.as_array(), mode = 'gaussian', seed = 42)
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


weight = 1/(noisy_data)
weight.array[weight.array == np.inf] = 1e-12    
A = Identity(ig)

print("Run Tikhonov regularised CGLS with weight")

x_init = ig.allocate()  
alpha = 20

op1 = np.sqrt(alpha) * Gradient(ig)

weight_operator = DiagonalOperator(weight.sqrt())
tmp_A = CompositionOperator(weight_operator, A)
   
block_op1 = BlockOperator( tmp_A,  op1, shape=(2,1))
block_data1 = BlockDataContainer(weight.sqrt() * noisy_data, op1.range_geometry().allocate())
   
cgls = CGLS(x_init = x_init, operator = block_op1, data = block_data1,
            max_iteration = 1000,
            update_objective_interval = 200, tolerance=1e-12)
cgls.run(verbose=True)

  
print("Run Tikhonov weighted LS with FISTA")

f1 = LeastSquares(A, noisy_data, 1, weight)
f2 = FunctionOperatorComposition(alpha *  L2NormSquared(), Gradient(ig))
objective = f1 + f2
  
# Set up and run FISTA algorithms
fista = FISTA(x_init, objective, max_iteration = 1000, 
           update_objective_interval = 200)
fista.run(verbose=True)

plt.imshow(fista.get_output().as_array())
plt.colorbar()
plt.show()    

plt.imshow(cgls.get_output().as_array())
plt.colorbar()
plt.show()

plt.figure()
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), cgls.get_output().as_array()[int(ig.shape[0]/2),:], label = 'Tikhonov + Weighted Regularised CGLS')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), fista.get_output().as_array()[int(ig.shape[0]/2),:], label = 'Tikhonov + Weighted with FISTA')
plt.legend()
plt.show()    

print("compare PDHG TV denoising with KL and weight L2 squared norm")    


# Regularisation Parameter depending on the noise distribution
alpha = 0.5

# Create operators
op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE, num_threads=3)
op2 = Identity(ig, ag)

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions      
f2 = 0.5 * WeightedL2NormSquared(weight=weight, b=noisy_data)
f = BlockFunction(alpha * MixedL21Norm(), f2) 
g = IndicatorBox(lower=0)
    
# Compute operator Norm
normK = operator.norm()
    
# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
            
print("compare PDHG weight L2 squared norm")    
# Setup and run the PDHG algorithm
pdhg1 = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, 
            max_iteration = 2000,
            update_objective_interval = 500)
pdhg1.run(verbose=True, very_verbose = True)
    
# Create functions      
f2 = KullbackLeibler(b=noisy_data)
f = BlockFunction(alpha * MixedL21Norm(), f2) 
g = IndicatorBox(lower=0)

print("compare PDHG Kullback Leibler")     

pdhg2 = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, 
            max_iteration = 2000,
            update_objective_interval = 500)
pdhg2.run(verbose=True, very_verbose = True)    
    
# Show results
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(pdhg1.get_output().as_array())
plt.title('TV weighted Reconstruction')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(pdhg2.get_output().as_array())
plt.title('TV KL Reconstruction')
plt.colorbar()
plt.show()    

plt.figure(figsize=(10,4))
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg1.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV weighted reconstruction')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg2.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV KL reconstruction')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'Truth')    
plt.legend()
plt.show()





    
    

    
    
    





