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

Total Variation 2D Tomography Reconstruction using PDHG algorithm:


Problem:     min_u  \alpha * ||\nabla u||_{2,1} + \frac{1}{2}||Au - g||^{2}
             min_u, u>0  \alpha * ||\nabla u||_{2,1} + \int A u  - g log (Au + \eta)

             \nabla: Gradient operator              
             A: System Matrix
             g: Noisy sinogram 
             \eta: Background noise
             
             \alpha: Regularization parameter
 
"""

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry,\
     AcquisitionData, AcquisitionGeometrySubsetGenerator

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, StochasticAlgorithm

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, IndicatorBox
                      
from ccpi.astra.operators import AstraProjectorSimple
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector

import os, sys
import tomophantom
from tomophantom import TomoP2D

from ccpi.utilities.display import plotter2D


class AstraSubsetProjectorSimple(AstraProjectorSimple):
    
    def __init__(self, geomv, geomp, device, **kwargs):
        kwargs = {'indices':None, 
                  'subset_acquisition_geometry':None,
                  'subset_id' : 0,
                  'number_of_subsets' : kwargs.get('number_of_subsets', 1)
                  }
        # This does not forward to its parent class :(
        super(AstraSubsetProjectorSimple, self).__init__(geomv, geomp, device)
        self.notify_new_subset(0, kwargs.get('number_of_subsets',1))
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        # print ('AstraSubsetProjectorSimple notify_new_subset')
        # updates the sinogram geometry and updates the projectors
        self.subset_id = subset_id
        self.number_of_subsets = number_of_subsets

        ag , indices = AcquisitionGeometrySubsetGenerator.generate_subset(
            self.sinogram_geometry, 
            subset_id, 
            number_of_subsets,
            AcquisitionGeometrySubsetGenerator.RANDOM)

        self.indices = indices
        device = self.fp.device
        self.subset_acquisition_geometry = ag
        
        self.fp = AstraForwardProjector(volume_geometry=self.volume_geometry,
                                        sinogram_geometry=ag,
                                        proj_id = None,
                                        device=device)

        self.bp = AstraBackProjector(volume_geometry = self.volume_geometry,
                                        sinogram_geometry = ag,
                                        proj_id = None,
                                        device = device)
        self.subs = self.subset_acquisition_geometry.allocate(0)
        

    def direct(self, image_data, out=None):
        self.fp.set_input(image_data)
        ret = self.fp.get_output()
            
        if out is None:
            out = self.sinogram_geometry.allocate(0)
            #print (self.indices)
            out.as_array()[self.indices] = ret.as_array()[:]
            return out
        else:
            out.as_array()[self.indices] = ret.as_array()[:]
            

    def adjoint(self, acquisition_data, out=None):
        self.subs.fill(acquisition_data.as_array()[self.indices])
        self.bp.set_input(self.subs)
        
        if out is None:
            return self.bp.get_output()
        else:
            # out.as_array()[self.indices] = ret.as_array()[:]
            out += self.bp.get_output()

class SPDHG(StochasticAlgorithm, PDHG):
    def __init__(self, **kwargs):
        super(SPDHG, self).__init__(**kwargs)
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.operator.notify_new_subset(subset_id, number_of_subsets)
# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
    
model = 1 # select a model number from the library
N = 128 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

phantom_2D = TomoP2D.Model(model, N, path_library2D)    
data = ImageData(phantom_2D)
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)

# Create acquisition data and geometry
detectors = N
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors)

# Select device
device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
Aos = AstraSubsetProjectorSimple(ig, ag, dev)

sin = Aop.direct(data)

# Create noisy data. Apply Gaussian noise
noises = ['gaussian', 'poisson']
noise = noises[which_noise]

if noise == 'poisson':
    scale = 5
    eta = 0
    noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
elif noise == 'gaussian':
    n1 = np.random.normal(0, 1, size = ag.shape)
    noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
    
else:
    raise ValueError('Unsupported Noise ', noise)

# Show Ground Truth and Noisy Data
plotter2D([data, noisy_data], titles=['Ground Truth', 'Noisy Data'])


def bo_notify(self, subset_id, number_of_subsets):
    for el in self.operators:
        el.notify_new_subset(subset_id, number_of_subsets)
def do_nothing(self, subset_id, number_of_subsets):
    pass
setattr(BlockOperator, 'notify_new_subset', bo_notify)

setattr(Gradient, 'notify_new_subset', do_nothing)

# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Compute operator Norm
normK = operator.norm()

# Create functions
if noise == 'poisson':
    
    alpha = 2
    f2 = KullbackLeibler(noisy_data)  
    g =  IndicatorBox(lower=0)    
    sigma = 1
    tau = 1/(sigma*normK**2)     
        
elif noise == 'gaussian':   
    
    alpha = 10
    f2 = 0.5 * L2NormSquared(b=noisy_data)                                         
    g = ZeroFunction()
    sigma = 10
    tau = 1/(sigma*normK**2) 
    
f1 = alpha * MixedL21Norm() 
f = BlockFunction(f1, f2)   

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run()


# Setup and run the SPDHG algorithm
# Create BlockOperator
operator_os = BlockOperator(op1, Aos, shape=(2,1) ) 

# Compute operator Norm

nsubs = 10
spdhg = SPDHG(f=f,g=g,operator=operator_os, tau=tau, sigma=sigma, number_of_subsets=nsubs)
spdhg.max_iteration = int(pdhg.max_iteration / nsubs)
spdhg.update_objective_interval = int(pdhg.update_objective_interval / nsubs)
spdhg.run(20)

plotter2D([data, noisy_data, pdhg.get_output(), spdhg.get_output()], 
   titles=['Ground Truth', 'Noisy Data', 'PDHG TV', 'SPDHG TV'])

# plt.figure(figsize=(15,15))
# plt.subplot(3,1,1)
# plt.imshow(data.as_array())
# plt.title('Ground Truth')
# plt.colorbar()
# plt.subplot(3,1,2)
# plt.imshow(noisy_data.as_array())
# plt.title('Noisy Data')
# plt.colorbar()
# plt.subplot(3,1,3)
# plt.imshow(pdhg.get_output().as_array())
# plt.title('TV Reconstruction')
# plt.colorbar()
# plt.show()
# plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(N/2),:], label = 'GTruth')
# plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
# plt.legend()
# plt.title('Middle Line Profiles')
# plt.show()



