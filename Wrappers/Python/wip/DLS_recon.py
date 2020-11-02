#from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.utilities import dataexample
from cil.optimisation.algorithms import CGLS
from cil.optimisation.operators import BlockOperator, Gradient
from cil.processors import Resizer, CenterOfRotationFinder
from cil.plugins.ccpi_reconstruction.operators import ProjectionOperatorFactory
from cil.utilities.display import plotter2D

# All external imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy

data_raw = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

#Convert the data from intensity to attenuation by taking the negative log
data_raw.log(out=data_raw)
data_raw *= -1

data = data_raw.subset(dimensions=['vertical','angle','horizontal'])
print(data_raw)
print(data)

# Have you noticed the bad pixel in the top left of each angular projection. 
# Use Resizer() to remove the first row of data.
# define the region of interest here
roi_crop = [(1, data.shape[0]),-1,-1]
#initialise the processsor
resizer = Resizer(roi=roi_crop)

#set the input data
resizer.set_input(data)

#get the output data
data_reduced = resizer.get_output()

centre_of_rotation = 86.25
shift = (centre_of_rotation - data.shape[2]/2)
data = data_reduced.subset(dimensions=['angle','vertical','horizontal'])

#allocate the memory
data_centred = data.geometry.allocate()
#use scipy to do a translation and interpolation of each projection image
shifted = scipy.ndimage.interpolation.shift(data.as_array(), (0,0,-shift), order=3,mode='nearest')
data_centred.fill(shifted)

# create the CCPi Projector
ag = data_centred.geometry

print("default ig", ag.get_ImageGeometry())

projector = ProjectionOperatorFactory.get_operator(ag)
ig = projector.domain_geometry()

print("ccpi ig", ig)


# Data need to be padded. This should use the Padder
if ig.shape != ag.get_ImageGeometry().shape:
    # needs to pad the data
    ag.config.panel.num_pixels[1] = 136
    d = ag.allocate(0)
    d.array[:,1:-1,:] = data_centred.as_array()[:]

cgls = CGLS(operator=projector, data=d, max_iteration=10, update_objective_interval=1)
cgls.run(5, verbose=1)

plotter2D([cgls.get_output().subset(horizontal_x=80),
           cgls.get_output().subset(horizontal_y=80),
           cgls.get_output().subset(vertical=80)], 
           titles=['horizontal_x', 'horizontal_y', 'vertical'])