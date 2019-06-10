

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:50:27 2019

@author: vaggelis
"""

from ccpi.framework import ImageData, ImageGeometry, BlockDataContainer, AcquisitionGeometry, AcquisitionData
from astropy.io import fits
import numpy
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import CGLS, PDHG
from ccpi.optimisation.functions import MixedL21Norm, L2NormSquared, BlockFunction, ZeroFunction, KullbackLeibler, IndicatorBox
from ccpi.optimisation.operators import Gradient, BlockOperator

from ccpi.astra.operators import  AstraProjectorMC, AstraProjectorSimple

import pickle


# load file

#filename_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed_test_flat/sino/rebin_slice_350/sino_log_rebin_282.fits' 
#filename_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed_test_flat/sino/rebin_slice_350/sino_log_rebin_564.fits'
#filename_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed_test_flat/sino/rebin_slice_350/sino_log_rebin_141.fits'
filename_sino = '/media/newhd/shared/DataProcessed/IMAT_beamtime_Feb_2019/preprocessed_test_flat/sino/rebin_slice_350/sino_log_rebin_80_channels.fits'

sino_handler = fits.open(filename_sino)
sino = numpy.array(sino_handler[0].data, dtype=float)

# change axis order: channels, angles, detectors
sino_new = numpy.rollaxis(sino, 2)
sino_handler.close()


sino_shape = sino_new.shape

num_channels = sino_shape[0] # channelss
num_pixels_h = sino_shape[2] # detectors
num_pixels_v = sino_shape[2] # detectors
num_angles = sino_shape[1] # angles


ig = ImageGeometry(voxel_num_x = num_pixels_h, voxel_num_y = num_pixels_v, channels = num_channels)

with open("/media/newhd/vaggelis/CCPi/IMAT_reconstruction/CCPi-Framework/Wrappers/Python/ccpi/optimisation/IMAT_data/golden_angles_new.txt") as f:
    angles_string = [line.rstrip() for line in f]
    angles = numpy.array(angles_string).astype(float)
                                                                                 

ag = AcquisitionGeometry('parallel', '2D',  angles * numpy.pi / 180, pixel_num_h = num_pixels_h, channels = num_channels)
op_MC = AstraProjectorMC(ig, ag, 'gpu')

sino_aqdata = AcquisitionData(sino_new, ag)
result_bp = op_MC.adjoint(sino_aqdata)

#%%

channel = [40, 60]
for j in range(2):        
    z4 = sino_aqdata.as_array()[channel[j]] 
    plt.figure(figsize=(10,6))
    plt.imshow(z4, cmap='viridis')
    plt.axis('off')
    plt.savefig('Sino_141/Sinogram_ch_{}_.png'.format(channel[j]), bbox_inches='tight', transparent=True)
    plt.show() 

#%%

def callback(iteration, objective, x):
    plt.imshow(x.as_array()[40])
    plt.colorbar()
    plt.show()

#%%
# CGLS 

x_init = ig.allocate()      
cgls1 = CGLS(x_init=x_init, operator=op_MC, data=sino_aqdata)
cgls1.max_iteration = 100
cgls1.update_objective_interval = 2
cgls1.run(20,verbose=True, callback=callback)

plt.imshow(cgls1.get_output().subset(channel=20).array)
plt.title('CGLS')
plt.colorbar()
plt.show()

#%%
with open('Sino_141/CGLS/CGLS_{}_iter.pkl'.format(20), 'wb') as f:
    z = cgls1.get_output()
    pickle.dump(z, f)  
    
#%% 
#% Tikhonov Space
    
x_init = ig.allocate() 
alpha = [1,3,5,10,20,50]

for a in alpha:
    
    Grad = Gradient(ig, correlation = Gradient.CORRELATION_SPACE)
    operator = BlockOperator(op_MC, a * Grad, shape=(2,1))
    blockData = BlockDataContainer(sino_aqdata, \
                                   Grad.range_geometry().allocate())
    cgls2 = CGLS()
    cgls2.max_iteration = 500
    cgls2.set_up(x_init, operator, blockData)
    cgls2.update_objective_interval = 50
    cgls2.run(100,verbose=True)
        
    with open('Sino_141/CGLS_Space/CGLS_a_{}.pkl'.format(a), 'wb') as f:
        z = cgls2.get_output()
        pickle.dump(z, f)
   
#% Tikhonov SpaceChannels

for a1 in alpha:
    
    Grad1 = Gradient(ig, correlation = Gradient.CORRELATION_SPACECHANNEL)
    operator1 = BlockOperator(op_MC, a1 * Grad1, shape=(2,1))
    blockData1 = BlockDataContainer(sino_aqdata, \
                                   Grad1.range_geometry().allocate())
    cgls3 = CGLS()
    cgls3.max_iteration = 500
    cgls3.set_up(x_init, operator1, blockData1)
    cgls3.update_objective_interval = 10
    cgls3.run(100, verbose=True)
    
    with open('Sino_141/CGLS_SpaceChannels/CGLS_a_{}.pkl'.format(a1), 'wb') as f1:
        z1 = cgls3.get_output()
        pickle.dump(z1, f1)
    
  
    
#%%
#
ig_tmp = ImageGeometry(voxel_num_x = num_pixels_h, voxel_num_y = num_pixels_v)
ag_tmp = AcquisitionGeometry('parallel', '2D',  angles * numpy.pi / 180, pixel_num_h = num_pixels_h)
op_tmp = AstraProjectorSimple(ig_tmp, ag_tmp, 'gpu')
normK1 = op_tmp.norm()

alpha_TV = [2, 5, 10] # for powder

# Create operators
op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACECHANNEL)
op2 = op_MC

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 


for alpha in alpha_TV:
# Create functions      
    f1 = alpha * MixedL21Norm()
    
    f2 = KullbackLeibler(sino_aqdata)    
    f = BlockFunction(f1, f2)  
    g = IndicatorBox(lower=0)
    
    # Compute operator Norm
    normK = numpy.sqrt(8 + normK1**2) 
    
    # Primal & dual stepsizes
    sigma = 1
    tau = 1/(sigma*normK**2)
    
    # Setup and run the PDHG algorithm
    pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
    pdhg.max_iteration = 5000
    pdhg.update_objective_interval = 500
#    pdhg.run(2000, verbose=True, callback=callback)   
    pdhg.run(5000, verbose=True, callback=callback)  
#    
    with open('Sino_141/TV_SpaceChannels/TV_a = {}.pkl'.format(alpha), 'wb') as f3:
        z3 = pdhg.get_output()
        pickle.dump(z3, f3)    
#    
#
#
#
##%%
#        
#ig_tmp = ImageGeometry(voxel_num_x = num_pixels_h, voxel_num_y = num_pixels_v)
#ag_tmp = AcquisitionGeometry('parallel', '2D',  angles * numpy.pi / 180, pixel_num_h = num_pixels_h)
#op_tmp = AstraProjectorSimple(ig_tmp, ag_tmp, 'gpu')
#normK1 = op_tmp.norm()
#
#alpha_TV = 10 # for powder
#
## Create operators
#op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACECHANNEL)
#op2 = op_MC
#
## Create BlockOperator
#operator = BlockOperator(op1, op2, shape=(2,1) ) 
#
#
## Create functions      
#f1 = alpha_TV * MixedL21Norm()
#f2 = 0.5 * L2NormSquared(b=sino_aqdata)    
#f = BlockFunction(f1, f2)  
#g = ZeroFunction()
#
## Compute operator Norm
##normK = 8.70320267279591 # For powder Run one time no need to compute again takes time
#normK = numpy.sqrt(8 + normK1**2) # for carbon
#
## Primal & dual stepsizes
#sigma = 0.1
#tau = 1/(sigma*normK**2)
#
#def callback(iteration, objective, x):
#    plt.imshow(x.as_array()[100])
#    plt.colorbar()
#    plt.show()
#
## Setup and run the PDHG algorithm
#pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
#pdhg.max_iteration = 2000
#pdhg.update_objective_interval = 100
#pdhg.run(2000, verbose=True)       
#        
#        
#        
#        
#        
#        
        
        
        
        
        
        
        
        
        
        
#%%
    
#with open('/media/newhd/vaggelis/CCPi/IMAT_reconstruction/CCPi-Framework/Wrappers/Python/ccpi/optimisation/CGLS_Tikhonov/CGLS_Space/CGLS_Space_a = 50.pkl', 'wb') as f:
#    z = cgls2.get_output()
#    pickle.dump(z, f)
#      
        
        #%%
with open('Sino_141/CGLS_Space/CGLS_Space_a_20.pkl', 'rb') as f1:
    x = pickle.load(f1)
    
with open('Sino_141/CGLS_SpaceChannels/CGLS_SpaceChannels_a_20.pkl', 'rb') as f1:
    x1 = pickle.load(f1)    
    
   
    
#    
plt.imshow(x.as_array()[40]*mask)
plt.colorbar()
plt.show()

plt.imshow(x1.as_array()[40]*mask)
plt.colorbar()
plt.show()

plt.plot(x.as_array()[40,100,:])
plt.plot(x1.as_array()[40,100,:])
plt.show()

#%%

# Show results

def circ_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

mask = circ_mask(141, 141, center=None, radius = 55)
plt.imshow(numpy.multiply(x.as_array()[40],mask))
plt.show()
#%%
#channel = [100, 200, 300]
#
#for i in range(3):     
#    tmp = cgls1.get_output().as_array()[channel[i]] 
#    
#    z =  tmp * mask
#    plt.figure(figsize=(10,6))
#    plt.imshow(z, vmin=0, cmap='viridis')
#    plt.axis('off')
##    plt.clim(0, 0.02)
##    plt.colorbar()
##    del z
#    plt.savefig('CGLS_282/CGLS_Chan_{}.png'.format(channel[i]), bbox_inches='tight', transparent=True)
#    plt.show()
#    
#    
##%% Line Profiles
#    
#n1, n2, n3 =  cgs.get_output().as_array().shape
#mask = circ_mask(564, 564, center=None, radius = 220)
#material = ['Cu', 'Fe', 'Ni']
#ycoords = [200, 300, 380]
#    
#for i in range(3):
#    z = cgs.get_output().as_array()[channel[i]]  * mask
#    
#    for k1 in range(len(ycoords)):
#        plt.plot(numpy.arange(0,n2), z[ycoords[k1],:])
#        plt.title('Channel {}: {}'.format(channel[i], material[k1]))
#        plt.savefig('CGLS/line_profile_chan_{}_material_{}.png'.\
#                        format(channel[i], material[k1]), bbox_inches='tight')
#        plt.show()
#    
#    
#
#
#
##%%
#
#%%
        


#%%

#plt.imshow(pdhg.get_output().subset(channel=100).as_array())
#plt.show()
