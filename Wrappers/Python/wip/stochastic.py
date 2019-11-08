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
import os, sys

import matplotlib.pyplot as plt


import numpy as np

# from utilities import islicer, link_islicer, psnr, plotter2D
from ccpi.utilities.show_utilities import show

from ccpi.optimisation.algorithms import Algorithm, GradientDescent   
import numpy


from ccpi.optimisation.functions import Norm2Sq
from ccpi.utilities import plotter2D

# get_ipython().magic(u'matplotlib inline')


# In[4]:


model = 12 # select a model number from the library
N = 256 # set dimension of the phantom
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
A = AstraProjectorSimple(ig, ag, device = 'gpu')
data = A.direct(im_data)


# In[ ]:




class StochasticAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super(StochasticAlgorithm, self).__init__(**kwargs)
        self.number_of_subsets = kwargs.get('number_of_subsets', 1)
        self.current_subset_id = 0
        self.epoch = kwargs.get('max_iteration',0)
        
    def update_subset(self):
        if self.number_of_subsets == self.current_subset_id + 1:
            # increment epoch
            self.epoch += 1
            self.iteration = 0
            self.current_subset_id = 0
        self.current_subset_id += 1
        # this callback must be defined by the concrete implementation of the 
        # algorithm to link to the appropriate object dealing with subsets
        self.notify_new_subset(self.current_subset_id, self.number_of_subsets)
        
    def should_stop(self):
        '''default stopping cryterion: number of iterations
        
        The user can change this in concrete implementatition of iterative algorithms.'''
        return self.max_epoch_stop_cryterion()
    
    def max_epoch_stop_cryterion(self):
        '''default stop cryterion for iterative algorithm: max_iteration reached'''
        return self.iteration >= self.max_iteration
    def notify_new_subset(self, subset_id, number_of_subsets):
        raise NotImplemented('This callback must be implemented by the concrete algorithm')
    
    def __next__(self):
        super(StochasticAlgorithm, self).__next__()
        self.update_subset()
        
class StochasticGradientDescent(StochasticAlgorithm, GradientDescent):
    def __init__(self, **kwargs):
        super(StochasticGradientDescent, self).__init__(**kwargs)
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.objective_function.notify_new_subset(subset_id, number_of_subsets)
        
        


# In[47]:


def notify_new_subset(self, subset_id, number_of_subsets):
    self.A.notify_new_subset(subset_id, number_of_subsets)

setattr(Norm2Sq, 'notify_new_subset', notify_new_subset)



### Changes in the Operator required to work as OS operator

def generate_acquisition_geometry_for_random_subset(ag, subset_id, number_of_subsets):
    ags = ag.clone()
    angles = ags.angles
    
    indices = random_indices(angles, subset_id, number_of_subsets)
    ags.angles = ags.angles[indices]
    return ags , indices
    
def random_indices(angles, subset_id, number_of_subsets):
    N = int(numpy.floor(float(len(angles))/float(number_of_subsets)))
    print ("How many angles?? ", N)
    shape = (N,)
    indices = numpy.random.choice(range(len(angles)),size=shape)
    ret = numpy.asarray(numpy.zeros_like(angles), dtype=numpy.bool)
    for i,el in enumerate(indices):
        ret[el] = True
    return ret


def operator_notify_new_subset(self, subset_id, number_of_subsets):
    # updates the sinogram geometry and updates the projectors
    ag , indices = generate_acquisition_geometry_for_subset(self.sinogram_geometry, subset_id, number_of_subsets)
    self.indices = indices
    device = self.fp.device
    
    self.fp = AstraForwardProjector(volume_geometry=geomv,
                                    sinogram_geometry=ag,
                                    proj_id = None,
                                    device=device)

    self.bp = AstraBackProjector(volume_geometry = ag,
                                    sinogram_geometry = geomp,
                                    proj_id = None,
                                    device = device)

def os_direct(self, IM, out=None):
    self.fp.set_input(IM)
    ret = self.fp.get_output()
        
    if out is None:
        out = self.sinogram_geometry.allocate(0)
        out.as_array()[self.indices] = ret
        return out
    else:
        out.as_array()[self.indices] = ret
        

def os_adjoint(self, DATA, out=None):
    self.bp.set_input(DATA)

    if out is None:
        return self.bp.get_output()
    else:
        out.fill(self.bp.get_output())

    
N = 10
angles = numpy.asarray([i for i in range(N)], dtype=numpy.float32) / 180. * numpy.pi
random_indices(angles, 0, 3)

print (len(ag.angles))
print (ag)

ags, indices = generate_acquisition_geometry_for_random_subset(ag, 0, 20)

print (len(ags.angles), indices)
print (ags)


# modify the AstraProjectorSimple to be with OS


# In[40]:


f = Norm2Sq(A, data)



# In[48]:


a = angles.copy()
b = a[random_indices(angles, 0,3)]
print (b)


# In[53]:


# idx = random_indices(angles, 0,3)
# a[idx] = [1,2,3]


# In[54]:


class AstraSubsetProjectorSimple(AstraProjectorSimple):
    def __init__(self, geomv, geomp, device):
        kwargs = {'indices':None, 
                  'subset_acquisition_geometry':None,
                  'subset_id' : 0,
                  'number_of_subsets' : 1
                  }
        # This does not forward to its parent class :(
        super(AstraSubsetProjectorSimple, self).__init__(geomv, geomp, device)

    def notify_new_subset(self, subset_id, number_of_subsets):
        # updates the sinogram geometry and updates the projectors
        self.subset_id = subset_id
        self.number_of_subsets = number_of_subsets

        ag , indices = generate_acquisition_geometry_for_random_subset(
            self.sinogram_geometry, 
            subset_id, 
            number_of_subsets)

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

    def direct(self, image_data, out=None):
        self.fp.set_input(image_data)
        ret = self.fp.get_output()
            
        if out is None:
            out = self.sinogram_geometry.allocate(0)
            print (self.indices)
            out.as_array()[self.indices] = ret.as_array()[:]
            return out
        else:
            out.as_array()[self.indices] = ret.as_array()[:]
            

    def adjoint(self, acquisition_data, out=None):
        self.bp.set_input(acquisition_data)

        if out is None:
            return self.bp.get_output()
        else:
            out.fill(self.bp.get_output())

# In[ ]:


# Create projection operator using Astra-Toolbox. Available CPU/CPU

subs = 10
A_os = AstraSubsetProjectorSimple(ig, ag, device = 'gpu')
A_os.notify_new_subset(1, 10)
data_os = A_os.direct(im_data)

A_os1 = AstraSubsetProjectorSimple(ig, ag, device = 'gpu')
A_os1.notify_new_subset(1, 1)
data_os1 = A_os1.direct(im_data)


plotter2D([data, data_os, data_os1], titles=['No subsets', '{} subsets'.format(subs), '1 subset'])

