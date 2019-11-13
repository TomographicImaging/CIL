# -*- coding: utf-8 -*-
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

from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy
from ccpi.framework import ImageGeometry
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotter2D(datacontainers, titles=None, fix_range=False, stretch_y=False, cmap='gray', axis_labels=None):
    '''plotter2D(datacontainers=[], titles=[], fix_range=False, stretch_y=False, cmap='gray', axes_labels=['X','Y'])
    
    plots 1 or more 2D plots in an (n x 2) matix
    multiple datasets can be passed as a list
    
    Can take ImageData, AquistionData or numpy.ndarray as input
    '''
    if(isinstance(datacontainers, list)) is False:
        datacontainers = [datacontainers]

    if titles is not None:
        if(isinstance(titles, list)) is False:
            titles = [titles]
            

    
    nplots = len(datacontainers)
    rows = int(round((nplots+0.5)/2.0))

    fig, (ax) = plt.subplots(rows, 2,figsize=(15,15))

    axes = ax.flatten() 

    range_min = float("inf")
    range_max = 0
    
    if fix_range == True:
        for i in range(nplots):
            if type(datacontainers[i]) is numpy.ndarray:
                dc = datacontainers[i]
            else:
                dc = datacontainers[i].as_array()
                
            range_min = min(range_min, numpy.amin(dc))
            range_max = max(range_max, numpy.amax(dc))
        
    for i in range(rows*2):
        axes[i].set_visible(False)

    for i in range(nplots):
        axes[i].set_visible(True)
        
        if titles is not None:
            axes[i].set_title(titles[i])
       
        if axis_labels is not None:
            axes[i].set_ylabel(axis_labels[1])
            axes[i].set_xlabel(axis_labels[0]) 
            
        if type(datacontainers[i]) is numpy.ndarray:
            dc = datacontainers[i]          
        else:
            dc = datacontainers[i].as_array()
            
            if axis_labels is None:
                axes[i].set_ylabel(datacontainers[i].dimension_labels[0])
                axes[i].set_xlabel(datacontainers[i].dimension_labels[1])        
        
        
        sp = axes[i].imshow(dc, cmap=cmap, origin='upper', extent=(0,dc.shape[1],dc.shape[0],0))
    
        
        im_ratio = dc.shape[0]/dc.shape[1]
        
        if stretch_y ==True:   
            axes[i].set_aspect(1/im_ratio)
            im_ratio = 1
            
        plt.colorbar(sp, ax=axes[i],fraction=0.0467*im_ratio, pad=0.02)
        
        if fix_range == True:
            sp.set_clim(range_min,range_max)
    plt.show()


def channel_to_energy(channel):
    # Convert from channel number to energy using calibration linear fit
    m = 0.2786
    c = 0.8575
    # Add on the offset due to using a restricted number of channel (varies based on user choice)
    shifted_channel = channel + 100
    energy = (shifted_channel * m) + c
    energy = format(energy,".3f")
    return energy


def show2D(x, title='', **kwargs):
    
    cmap = kwargs.get('cmap', 'gray')
    font_size = kwargs.get('font_size', [12, 12])
    minmax = (kwargs.get('minmax', (x.as_array().min(),x.as_array().max())))
    
    # get numpy array
    tmp = x.as_array()
      
    # labels for x, y      
    labels = kwargs.get('labels', ['x','y']) 
    
    
    # defautl figure_size
    figure_size = kwargs.get('figure_size', (10,5)) 
    
    # show 2D via plt
    fig, ax = plt.subplots(figsize = figure_size)  
    im = ax.imshow(tmp, cmap = cmap, vmin=min(minmax), vmax=max(minmax))
    ax.set_title(title, fontsize = font_size[0])
    ax.set_xlabel(labels[0], fontsize = font_size[1])
    ax.set_ylabel(labels[1], fontsize = font_size[1]) 
    divider = make_axes_locatable(ax) 
    cax1 = divider.append_axes("right", size="5%", pad=0.1)  
    fig.colorbar(im, ax=ax, cax = cax1)
    plt.show()    
    

    
    
def show3D(x, title , **kwargs):
        
    # show slices for 3D
    show_slices = kwargs.get('show_slices', [int(i/2) for i in x.shape])
    
    # defautl figure_size
    figure_size = kwargs.get('figure_size', (10,5))    
    
    # font size of title and labels
    cmap = kwargs.get('cmap', 'gray')
    font_size = kwargs.get('font_size', [12, 12])

    # Default minmax scaling
    minmax = (kwargs.get('minmax', (x.as_array().min(),x.as_array().max())))
    
    labels = kwargs.get('labels', ['x','y','z'])     
            
    title_subplot = kwargs.get('title_subplot',['Axial','Coronal','Sagittal'])

    fig, axs = plt.subplots(1, 3, figsize = figure_size)
    
    tmp = x.as_array()
    
    im1 = axs[0].imshow(tmp[show_slices[0],:,:], cmap=cmap, vmin=min(minmax), vmax=max(minmax))
    axs[0].set_title(title_subplot[0], fontsize = font_size[0])
    axs[0].set_xlabel(labels[0], fontsize = font_size[1])
    axs[0].set_ylabel(labels[1], fontsize = font_size[1])
    divider = make_axes_locatable(axs[0]) 
    cax1 = divider.append_axes("right", size="5%", pad=0.1)      
    fig.colorbar(im1, ax=axs[0], cax = cax1)   
    
    im2 = axs[1].imshow(tmp[:,show_slices[1],:], cmap=cmap, vmin=min(minmax), vmax=max(minmax))
    axs[1].set_title(title_subplot[1], fontsize = font_size[0])
    axs[1].set_xlabel(labels[0], fontsize = font_size[1])
    axs[1].set_ylabel(labels[2], fontsize = font_size[1])
    divider = make_axes_locatable(axs[1])  
    cax1 = divider.append_axes("right", size="5%", pad=0.1)       
    fig.colorbar(im2, ax=axs[1], cax = cax1)   

    im3 = axs[2].imshow(tmp[:,:,show_slices[2]], cmap=cmap, vmin=min(minmax), vmax=max(minmax))
    axs[2].set_title(title_subplot[2], fontsize = font_size[0]) 
    axs[2].set_xlabel(labels[1], fontsize = font_size[1])
    axs[2].set_ylabel(labels[2], fontsize = font_size[1])
    divider = make_axes_locatable(axs[2])  
    cax1 = divider.append_axes("right", size="5%", pad=0.1) 
    fig.colorbar(im3, ax=axs[2], cax = cax1)       
    
    fig.suptitle(title, fontsize = font_size[0])
    plt.tight_layout(h_pad=1)
    plt.show()
    
         
def show2D_channels(x, title, show_channels = [1], **kwargs):
    
    # defautl figure_size
    figure_size = kwargs.get('figure_size', (10,5))    
    
    # font size of title and labels
    cmap = kwargs.get('cmap', 'gray')
    font_size = kwargs.get('font_size', [12, 12])
    
    labels = kwargs.get('labels', ['x','y'])

    # Default minmax scaling
    minmax = (kwargs.get('minmax', (x.as_array().min(),x.as_array().max()))) 
    
    if len(show_channels)==1:
        show2D(x.subset(channel=show_channels[0]), title + ' Energy {}'.format(channel_to_energy(show_channels[0])) + " keV", **kwargs)        
    else:
        
        fig, axs = plt.subplots(1, len(show_channels), sharey=True, figsize = figure_size)    
    
        for i in range(len(show_channels)):
            im = axs[i].imshow(x.subset(channel=show_channels[i]).as_array(), cmap = cmap, vmin=min(minmax), vmax=max(minmax))
            axs[i].set_title('Energy {}'.format(channel_to_energy(show_channels[i])) + "keV", fontsize = font_size[0])
            axs[i].set_xlabel(labels[0], fontsize = font_size[1])
            divider = make_axes_locatable(axs[i])
            cax1 = divider.append_axes("right", size="5%", pad=0.1)    
            fig.colorbar(im, ax=axs[i], cax = cax1)
        axs[0].set_ylabel(labels[1], fontsize = font_size[1]) 
        fig.suptitle(title, fontsize = font_size[0])
        plt.tight_layout(h_pad=1)
    plt.show()
        
def show3D_channels(x, title = None, show_channels = 0, **kwargs):
    
    show3D(x.subset(channel=show_channels), title + ' Energy {}'.format(channel_to_energy(show_channels))  + " keV", **kwargs)        
        
def show(x, title = None, show_channels = [1], **kwargs):
    
    sz = len(x.shape)
    ch_num = x.geometry.channels
    
    if ch_num == 1:
        
        if sz == 2:
            show2D(x, title, **kwargs)
        elif sz == 3:
            show3D(x, title, **kwargs)
            
    elif ch_num>1:
        
        if len(x.shape[1:]) == 2:
            show2D_channels(x, title, show_channels,  **kwargs)
            
        elif len(x.shape[1:]) == 3:
            show3D_channels(x, title, show_channels,  **kwargs)  
    plt.show()
            
            
        
            
if __name__ == '__main__':         
    
    from ccpi.framework import TestData, ImageData
    import os
    import sys
    
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data = loader.load(TestData.PEPPERS, size=(256,256))
    ig = data.geometry
    
    show2D(data)
       
    if False:
    
        N = 100
        ig2D = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        ig3D = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_num_z=N)
        
        ch_number = 10
        ig2D_ch = ImageGeometry(voxel_num_x=N, voxel_num_y=N, channels = ch_number)
        ig3D_ch = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_num_z=N, channels = ch_number)
        
        x2D = ig2D.allocate('random_int')
        x2D_ch = ig2D_ch.allocate('random_int')
        x3D = ig3D.allocate('random_int')
        x3D_ch = ig3D_ch.allocate('random_int')         
         
        #%%
        ###############################################################################           
        # test 2D cases
        show(x2D)
        show(x2D, title = '2D no font')
        show(x2D, title = '2D with font', font_size = (50, 30))
        show(x2D, title = '2D with font/fig_size', font_size = (20, 10), figure_size = (10,10))
        show(x2D, title = '2D with font/fig_size', 
                  font_size = (20, 10), 
                  figure_size = (10,10),
                  labels = ['xxx','yyy'])
        ###############################################################################
        
        
        #%%
        ###############################################################################
        # test 3D cases
        show(x3D)
        show(x3D, title = '2D no font')
        show(x3D, title = '2D with font', font_size = (50, 30))
        show(x3D, title = '2D with font/fig_size', font_size = (20, 20), figure_size = (10,4))
        show(x3D, title = '2D with font/fig_size', 
                  font_size = (20, 10), 
                  figure_size = (10,4),
                  labels = ['xxx','yyy','zzz'])
        ###############################################################################
        #%%
        
        ###############################################################################
        # test 2D case + channel
        show(x2D_ch, show_channels = [1, 2, 5])
        
        ###############################################################################
                
          