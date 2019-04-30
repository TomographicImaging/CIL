# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData
from astropy.io import fits
import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, KullbackLeibler, L2NormSquared,\
                      MixedL21Norm, BlockFunction

from ccpi.astra.ops import AstraProjectorSimple


# load IMAT file
#filename_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed_test_flat/sino/rebin_slice_350/sino_log_rebin_141.fits' 
filename_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed_test_flat/sino/rebin_slice_350/sino_log_rebin_564.fits' 

sino_handler = fits.open(filename_sino)
sino_tmp = numpy.array(sino_handler[0].data, dtype=float)
# reorder sino coordinate: channels, angles, detectors
sinogram = numpy.rollaxis(sino_tmp, 2)
sino_handler.close()
#%%
# white beam data
sinogram_wb = sinogram.sum(axis=0)

pixh = sinogram_wb.shape[1] # detectors
pixv = sinogram_wb.shape[1] # detectors

# WhiteBeam Geometry
igWB = ImageGeometry(voxel_num_x = pixh, voxel_num_y = pixv)

# Load Golden angles
with open("golden_angles.txt") as f:
    angles_string = [line.rstrip() for line in f]
    angles = numpy.array(angles_string).astype(float)
agWB = AcquisitionGeometry('parallel', '2D',  angles * numpy.pi / 180, pixh)
op_WB = AstraProjectorSimple(igWB, agWB, 'cpu')
sinogram_aqdata = AcquisitionData(sinogram_wb, agWB)

# BackProjection
result_bp = op_WB.adjoint(sinogram_aqdata)

plt.imshow(result_bp.subset(channel=50).array)
plt.title('BackProjection')
plt.show()



#%%

# Regularisation Parameter
alpha = 10

# Create operators
op1 = Gradient(igWB)
op2 = op_WB

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions
      
f1 = alpha * MixedL21Norm()
f2 = KullbackLeibler(sinogram_aqdata)    
#f2 = L2NormSquared(b = sinogram_aqdata)  
f = BlockFunction(f1, f2)  
                                      
g = ZeroFunction()

diag_precon =  True

if diag_precon:
    
    def tau_sigma_precond(operator):
        
        tau = 1/operator.sum_abs_row()
        sigma = 1/ operator.sum_abs_col()
               
        return tau, sigma

    tau, sigma = tau_sigma_precond(operator)
             
else:
    # Compute operator Norm
    normK = operator.norm()
    print ("normK", normK)
    # Primal & dual stepsizes
    sigma = 0.1
    tau = 1/(sigma*normK**2)

#%%


## Primal & dual stepsizes
sigma = 0.1
tau = 1/(sigma*normK**2)
#
#
## Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 10000
pdhg.update_objective_interval = 500

def circ_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
           radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def show_result(niter, objective, solution):
    
    mask = circ_mask(pixh, pixv, center=None, radius = 220)  # 55 with 141,     
    plt.imshow(solution.as_array() *  mask)
    plt.colorbar()
    plt.title("Iter: {}".format(niter))
    plt.show()
    

    print( "{:04}/{:04} {:<5} {:.4f} {:<5} {:.4f} {:<5} {:.4f}".\
                      format(niter, pdhg.max_iteration,'', \
                             objective[0],'',\
                             objective[1],'',\
                             objective[2]))

pdhg.run(10000, callback = show_result)

#%%

mask = circ_mask(pixh, pixv, center=None, radius = 210) #  55 with 141,
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(pdhg.get_output().as_array() *  mask)
plt.title('Ground Truth')
plt.colorbar()
plt.show()
