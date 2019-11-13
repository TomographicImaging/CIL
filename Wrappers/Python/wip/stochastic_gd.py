#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.framework import ImageData, TestData, ImageGeometry, AcquisitionGeometry, AcquisitionData, BlockDataContainer

from ccpi.optimisation.functions import L2NormSquared, ZeroFunction, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, FunctionOperatorComposition
from ccpi.optimisation.operators import Gradient, BlockOperator
from ccpi.optimisation.algorithms import PDHG, SIRT, CGLS

from ccpi.astra.operators import AstraProjectorSimple, AstraProjector3DSimple
from ccpi.astra.processors import FBP, AstraForwardProjector, AstraBackProjector

import tomophantom
from tomophantom import TomoP2D
import os, sys, time

import matplotlib.pyplot as plt


import numpy as np

# from utilities import islicer, link_islicer, psnr, plotter2D
from ccpi.utilities.display import show

from ccpi.optimisation.algorithms import Algorithm, GradientDescent   
import numpy


from ccpi.optimisation.functions import Norm2Sq
from ccpi.utilities.display import plotter2D


# stochastic imports

from ccpi.optimisation.algorithms import StochasticGradientDescent
from ccpi.optimisation.functions import StochasticNorm2Sq
from ccpi.framework import AcquisitionGeometrySubsetGenerator

# get_ipython().magic(u'matplotlib inline')


# In[4]:


model = 12 # select a model number from the library
N = 512 # set dimension of the phantom
device = 'gpu'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

phantom = TomoP2D.Model(model, N, path_library2D) 

# Define image geometry.
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                   voxel_size_x = 0.1,
                   voxel_size_y = 0.1)
im_data = ig.allocate()
im_data.fill(phantom)

# show(im_data, title = 'TomoPhantom', cmap = 'inferno')
# Create AcquisitionGeometry and AcquisitionData 
detectors = N
angles = np.linspace(0, np.pi, 180, dtype='float32')
ag = AcquisitionGeometry('parallel','2D', angles, detectors,
                        pixel_size_h = 0.1)

# Create projection operator using Astra-Toolbox. Available CPU/CPU
A = AstraProjectorSimple(ig, ag, device = device)
data = A.direct(im_data)


# In[ ]:




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
    
# In[ ]:


# Create projection operator using Astra-Toolbox. Available CPU/CPU


#%%
l2 = Norm2Sq(A=A, b=data)
gd = GradientDescent(x_init=im_data*0., objective_function=l2, rate=1e-4 , 
     update_objective_interval=10, max_iteration=100)
tgd0 = time.time()
gd.run()
tgd1 = time.time()

#%%

#%%
nsubs = 10
sl2 = StochasticNorm2Sq(A=AstraSubsetProjectorSimple(ig, ag, device = 'gpu'),
                        b=data, number_of_subsets=nsubs)

sgd = StochasticGradientDescent(x_init=im_data*0., 
                                objective_function=sl2, rate=1e-3, 
                                update_objective_interval=10, max_iteration=100, 
                                number_of_subsets=nsubs)


tsgd0 = time.time()
sgd.run()
tsgd1 = time.time()
#%%
plotter2D([im_data, gd.get_output(), sgd.get_output()], titles=['ground truth', 
           'gd {}'.format(tgd1-tgd0), 'sgd {}'.format(tsgd1-tsgd0)])




    

