#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:00:53 2020

@author: evangelos
"""
        

from ccpi.processors import RingRemoval
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry
from ccpi.astra.operators import AstraProjectorSimple

import tomophantom
from tomophantom import TomoP2D       
import os
import numpy as np
import matplotlib.pyplot as plt

from ccpi.io import NEXUSDataReader
    
def test_2D_demo_ring():
    
    print("Start 2D ring removal in simulated sinogram")
    
    model = 1 # select a model number from the library
    N = 512 # set dimension of the phantom
    path = os.path.dirname(tomophantom.__file__)
    path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
    
    phantom_2D = TomoP2D.Model(model, N, path_library2D)    
    data = ImageData(phantom_2D)
    ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_size_x = 0.1, voxel_size_y = 0.1)
    
    # Create acquisition data and geometry
    detectors = N
    angles = np.linspace(0, np.pi, 120)
    ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
        
    Aop = AstraProjectorSimple(ig, ag, 'cpu')
    sin = ag.allocate()
    Aop.direct(data, out = sin)
    
    sin_stripe = 0*sin
#        sin_stripe = sin
    tmp = sin.as_array()
    tmp[:,::20]=0
    sin_stripe.fill(tmp)
    
    
    ring_removal = RingRemoval(4, "db25", 20, info = True)
    ring_removal.set_input(sin_stripe)
    ring_recon = ring_removal.get_output()
    

    plt.figure(figsize=(20,10))
    plt.subplot(3,1,1)
    plt.imshow(sin_stripe.as_array())
    plt.subplot(3,1,2)
    plt.imshow(ring_recon.as_array())    
    plt.subplot(3,1,3)
    plt.imshow((sin-ring_recon).abs().as_array())    
    plt.show() 
    
    print("Finish first demo")
    
    
def test_3D():
    
    print("Start 3D ring removal in real data")
    
    # Load data --> sirt after flat and after ring
    read_sinogram3D = NEXUSDataReader()
    read_sinogram3D.set_up(nexus_file = 'Sinogram_Rock_sample_3D.nxs')
    
    sinogram3D = read_sinogram3D.load_data()
    
    print(sinogram3D.shape)
    
    ring_removal = RingRemoval(4, "db25", 20, info = True)
    ring_removal.set_input(sinogram3D)
    ring_recon = ring_removal.get_output()
    
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.imshow(sinogram3D.subset(vertical=0).as_array(), cmap = "inferno")
    plt.title("Stripes Sinogram")
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(ring_recon.subset(vertical=0).as_array(), cmap = "inferno") 
    plt.title("Reconstructed")
    plt.colorbar() 
    plt.show()
    
    print("Finish demo")
    
def test_2D_channels():
    
    print("Start 2D+channels ring removal in real data")

    read_sinogram2D_chan = NEXUSDataReader()
    read_sinogram2D_chan.set_up(nexus_file = 'Sinogram_Rock_sample_2D_channels.nxs')
    
    sinogram2D_chan = read_sinogram2D_chan.load_data()
    
    print(sinogram2D_chan.shape)
    
    ring_removal = RingRemoval(4, "db25", 20, info = True)
    ring_removal.set_input(sinogram2D_chan)
    ring_recon = ring_removal.get_output()
    
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.imshow(sinogram2D_chan.subset(channel=0).as_array(), cmap = "inferno")
    plt.title("Stripes Sinogram")
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(ring_recon.subset(channel=0).as_array(), cmap = "inferno") 
    plt.title("Reconstructed")
    plt.colorbar() 
    plt.show()
    
    print("Finish demo")
    
    
def test_3D_channels():
    
    print("Start 3D+channels ring removal in real data")

    # Load data --> sirt after flat and after ring
    read_sinogram3D_chan = NEXUSDataReader()
    read_sinogram3D_chan.set_up(nexus_file = 'Sinogram_Rock_sample_3D_channels.nxs')
    
    sinogram3D_chan = read_sinogram3D_chan.load_data()
    
    print(sinogram3D_chan.shape)
    
    ring_removal = RingRemoval(4, "db25", 20, info = True)
    ring_removal.set_input(sinogram3D_chan)
    ring_recon = ring_removal.get_output()
            
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.imshow(sinogram3D_chan.subset(channel=0, vertical=0).as_array(), cmap = "inferno")
    plt.title("Stripes Sinogram")
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(ring_recon.subset(channel=0, vertical=0).as_array(), cmap = "inferno") 
    plt.title("Reconstructed")
    plt.colorbar() 
    plt.show()
    
    print("Finish demo")
    

test_2D_demo_ring()    
test_2D_channels()
test_3D()    
test_3D_channels()    
 
              
    
    
    
    
