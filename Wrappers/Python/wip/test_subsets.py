# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



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
#import islicer, link_islicer, psnr, plotter2D
from ccpi.utilities.display import show

from ccpi.optimisation.algorithms import Algorithm, GradientDescent   
import numpy


from ccpi.optimisation.functions import Norm2Sq
from ccpi.utilities.display import plotter2D


# stochastic imports

from ccpi.optimisation.algorithms import StochasticGradientDescent
#from ccpi.optimisation.functions import StochasticNorm2Sq
from ccpi.framework import AcquisitionGeometrySubsetGenerator

# get_ipython().magic(u'matplotlib inline')


# In[4]:


model = 12 # select a model number from the library
N = 1024 # set dimension of the phantom
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

data_no_subset = data.copy()

nsubs = 10
step_rate = 0.0002328

data.geometry.generate_subsets(nsubs,'uniform')
b = data.copy()
print (b.shape)
#OS_A = AstraSubsetProjectorSimple(ig, data.geometry, device = 'gpu')
#### Check single steps
#
## GD does A.adjoint(A.direct(x) - b)
#print (data.geometry.subset_id)
#
#x = OS_A.domain_geometry().allocate(0)
#tmp = OS_A.domain_geometry().allocate(0)
#
##%%
#for j in range(10):
#    for i in range(nsubs):
#        data.geometry.subset_id = i
#        OS_A.notify_new_subset(i, nsubs)
#        #print(numpy.where(data.geometry.subsets[i]>0))
#        axmb = OS_A.direct(x)
#        print (data.geometry.angles)
#        print (axmb.geometry.angles)
#    #    print (axmb.shape)
#        numpy.testing.assert_array_equal(data.geometry.angles, axmb.geometry.angles)
#        axmb =- data
#        
#    #    print (diff.geometry.angles)
#    #    print (diff.shape)
#        
#        OS_A.adjoint(axmb, out=tmp)
#        tmp *= -step_rate
#        
#        x+=tmp
#    
#    
#plotter2D(x)