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


import numpy as np 
import numpy                          
import matplotlib.pyplot as plt


from ccpi.optimisation.operators import Gradient
from ccpi.optimisation.functions import SmoothMixedL21Norm,  L2NormSquared, FunctionOperatorComposition
from ccpi.optimisation.algorithms import GradientDescent

from ccpi.framework import TestData
import os
import sys


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(512,512))
ig = data.geometry
ag = ig

#%%

n1 = TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 10)

noisy_data = ig.allocate()
noisy_data.fill(n1)

# Show Ground Truth and Noisy Data
#plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
#plt.imshow(data.as_array())
#plt.title('Ground Truth')
#plt.colorbar()
#plt.subplot(1,2,2)
#plt.imshow(noisy_data.as_array())
#plt.title('Noisy Data')
#plt.colorbar()
#plt.show()


alpha = 0.3
epsilon = 1e-4


Grad = Gradient(ig)

f1 = FunctionOperatorComposition( alpha * SmoothMixedL21Norm(epsilon), Grad)
f2 = 0.5 * L2NormSquared(b=noisy_data)
objective_function = f1  +  f2
step_size = 0.005

x_init = noisy_data
gd = GradientDescent(x_init, objective_function, step_size,
                     max_iteration = 2000,update_objective_interval = 100)
gd.run(verbose=True)


## Show results
#plt.figure(figsize=(20,5))
#plt.subplot(1,4,1)
#plt.imshow(data.as_array())
#plt.title('Ground Truth')
#plt.colorbar()
#plt.subplot(1,4,2)
#plt.imshow(noisy_data.as_array())
#plt.title('Noisy Data')
#plt.colorbar()
#plt.subplot(1,4,3)
#plt.imshow(gd.get_output().as_array())
#plt.title('GD Reconstruction')
#plt.colorbar()
#plt.subplot(1,4,4)
#plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), gd.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
#plt.legend()
#plt.title('Middle Line Profiles')
#plt.show()
