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
Problem:     min_x, x>0  \alpha * ||\nabla x||_{2,1} + ||x-g||_{1}
             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data with Salt & Pepper Noise
             
             
             Method = 0 ( PDHG - split ) :  K = [ \nabla,
                                                 Identity]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = \nabla    
             
             
"""

import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
from ccpi.optimisation.functions import MixedL21Norm, MixedL11Norm, L2NormSquared, BlockFunction, L1Norm                      
from ccpi.framework import TestData, ImageGeometry
import os, sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.PEPPERS, size=(256,256))
ig = data.geometry
ag = ig

# Create noisy data. 
n1 = random_noise(data.as_array(), mode = 'gaussian', var = 0.15, seed = 50)
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

# Regularisation Parameter
operator = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
f1 =  5 * MixedL21Norm()
g = 0.5 * L2NormSquared(b=noisy_data)
            
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg1 = PDHG(f=f1,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg1.max_iteration = 2000
pdhg1.update_objective_interval = 200
pdhg1.run(1000)


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
plt.title('TV Reconstruction')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(pdhg2.get_output().as_array())
plt.title('TV Reconstruction')
plt.colorbar()
plt.show()

