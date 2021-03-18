# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy
from cil.framework import ImageGeometry, AcquisitionData, ImageData, DataContainer, BlockDataContainer
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PlotData(object):
    def __init__(self, data, title, axis_labels, origin):
        self.data = data
        self.title = title
        self.axis_labels = axis_labels
        self.origin = origin
        self.range = None

def plotter2D(datacontainers, title=None, slice_list=None, fix_range=False, axis_labels=None, origin='lower-left', cmap='gray', num_cols=2, size=(15,15)):
    r'''This function plots 2D slices from cil DataContainer types

    Plots 1 or more 2D plots in an (n x num_cols) matix
    Can plot multiple slices from one 3D dataset, or compare multiple datasets
    Inputs can be single arguments or list of arguments that will be sequentally applied to subplots
    If no slice_list is passed a 3D dataset will display the centre slice of the outer dimension, a 4D dataset will show the centre slices of the two outer dimension.

    :param datacontainers: The DataContainers to be displayed.
    :type datacontainers: ImageData, AcquisitionData, list of  Image/AcquisitionData, BlockDataContainer
    :param title: The title for each figure
    :type title: string, list of strings
    :param slice_list: The slices to show. A list of intergers will show slices for the outer dimension. For 3D datacontainers single slice: (direction, index). For 4D datacontainers two slices: [(direction0, index),(direction1, index)].
    :type slice_list: tuple, int, list of tuples, list of ints
    :param fix_range: Sets the display range of the data. `True` sets all plots to the global (min, max). 
    :type fix_range: boolian, tuple, list of tuples
    :param axis_labels: The axis labels for each figure e.g. ('x','y')
    :type axis_labels: tuple, list of tuples
    :param origin: Sets the display origin. 'lower/upper-left/right'
    :type origin: string, list of strings
    :param cmap: Sets the colour map of the plot (see matplotlib.pyplot)
    :param num_cols: Sets the number of columns of subplots to display
    :type num_cols: int
    :param size: Figure size in inches
    :type size: tuple of floats
     '''

    #get number of subplots, number of input datasets, or number of slices requested
    if isinstance(datacontainers, (list, BlockDataContainer)):
        num_plots = len(datacontainers)
    else:
        dim = len(datacontainers.shape)

        if slice_list is None or dim == 2:
            num_plots = 1
        elif type(slice_list) is tuple:
            num_plots = 1
        elif dim == 4 and type(slice_list[0]) is tuple:
            num_plots = 1  
        else:
            num_plots = len(slice_list)       

    subplots = []

    #range needs subsetted data
    range_min = float("inf")
    range_max = -range_min

    #set up, all inputs can be 1 or num_plots
    for i in range(num_plots):

        #get data per subplot, subset where required
        if isinstance(datacontainers, (list, BlockDataContainer)):
            data = datacontainers[i]
        else:
            data = datacontainers

        if len(data.shape) ==4:

            if slice_list is None or type(slice_list) is tuple or type(slice_list[0]) is tuple: #none, (direction, ind) or [(direction0, ind), (direction1, ind)] apply to all datasets
                slice_requested = slice_list
            elif type(slice_list[i]) == int or len(slice_list[i]) > 1: # [ind0, ind1, ind2] of direction0, or [[(direction0, ind), (direction1, ind)],[(direction0, ind), (direction1, ind)]]
                slice_requested = slice_list[i]
            else:
                slice_requested = slice_list[i][0] # [[(direction0, ind)],[(direction0, ind)]]

            cut_axis = [0,1]
            cut_slices = [data.shape[0]//2, data.shape[1]//2]

            if type(slice_requested) is int:
                #use axis 0, slice val
                cut_slices[0] = slice_requested
            elif type(slice_requested) is tuple:
                #get axis ind,
                # if 0 default 1
                # if 1 default 0
                axis = slice_requested[0]
                if slice_requested[0] is str:
                    axis = data.dimension_labels.index(axis)

                if axis == 0:
                    cut_axis[0] = slice_requested[0]
                    cut_slices[0] = slice_requested[1]
                else:
                    cut_axis[1] = slice_requested[0]
                    cut_slices[1] = slice_requested[1]

            elif type(slice_requested) is list:
                #use full input
                cut_axis[0] = slice_requested[0][0]
                cut_axis[1] = slice_requested[1][0]
                cut_slices[0] = slice_requested[0][1]
                cut_slices[1] = slice_requested[1][1]

                if cut_axis[0] > cut_axis[1]:
                    cut_axis.reverse()
                    cut_slices.reverse()

            try:
                if hasattr(data, 'subset'):
                    if type(cut_axis[0]) is int:
                        cut_axis[0] = data.dimension_labels[cut_axis[0]]
                    if type(cut_axis[1]) is int:
                        cut_axis[1] = data.dimension_labels[cut_axis[1]]

                    temp_dict = {cut_axis[0]:cut_slices[0], cut_axis[1]:cut_slices[1]}
                    plot_data = data.subset(**temp_dict, force=True)
                elif hasattr(data,'as_array'):
                    plot_data = data.as_array().take(indices=cut_slices[1], axis=cut_axis[1])
                    plot_data = plot_data.take(indices=cut_slices[0], axis=cut_axis[0])
                else:
                    plot_data = data.take(indices=cut_slices[1], axis=cut_axis[1])
                    plot_data = plot_data.take(indices=cut_slices[0], axis=cut_axis[0])

            except:
                raise TypeError("Unable to slice input data. Could not obtain 2D slice {0} from {1} with shape {2}.\n\
                          Pass either correct slice information or a 2D array".format(slice_requested, type(data), data.shape))

            subtitle = "direction: ({0},{1}), slice: ({2},{3})".format(*cut_axis, * cut_slices)

                
        elif len(data.shape) == 3:
            #get slice list per subplot
            if type(slice_list) is list:            #[(direction, ind), (direction, ind)],  [ind0, ind1, ind2] of direction0
                slice_requested = slice_list[i]
            else:                                   #(direction, ind) single tuple apply to all datasets
                slice_requested = slice_list

            #default axis 0, centre slice
            cut_slice = data.shape[0]//2
            cut_axis = 0
             
            if type(slice_requested) is int:
                #use axis 0, slice val
                cut_slice = slice_requested             
            if type(slice_requested) is tuple:
                cut_slice = slice_requested[1]
                cut_axis = slice_requested[0]

            try:
                if hasattr(data, 'subset'):
                    if type(cut_axis) is int:
                        cut_axis = data.dimension_labels[cut_axis]
                    temp_dict = {cut_axis:cut_slice}
                    plot_data = data.subset(**temp_dict, force=True)
                elif hasattr(data,'as_array'):
                    plot_data = data.as_array().take(indices=cut_slice, axis=cut_axis)
                else:
                    plot_data = data.take(indices=cut_slice, axis=cut_axis)
            except:
                raise TypeError("Unable to slice input data. Could not obtain 2D slice {0} from {1} with shape {2}.\n\
                          Pass either correct slice information or a 2D array".format(slice_requested, type(data), data.shape))

            subtitle = "direction: {0}, slice: {1}".format(cut_axis,cut_slice)  
        else:
            plot_data = data
            subtitle = None


        #check dataset is now 2D
        if len(plot_data.shape) != 2:
            raise TypeError("Unable to slice input data. Could not obtain 2D slice {0} from {1} with shape {2}.\n\
                             Pass either correct slice information or a 2D array".format(slice_requested, type(data), data.shape))
       
        #get axis labels per subplot
        if type(axis_labels) is list: 
            plot_axis_labels = axis_labels[i]
        else:
            plot_axis_labels = axis_labels

        if plot_axis_labels is None and hasattr(plot_data,'dimension_labels'):
                plot_axis_labels = (plot_data.dimension_labels[1],plot_data.dimension_labels[0])

        #get min/max of subsetted data
        range_min = min(range_min, plot_data.min())
        range_max = max(range_max, plot_data.max())

        #get title per subplot
        if isinstance(title, list):
            if title[i] is None:
                plot_title = ''
            else:
                plot_title = title[i]
        else:
            if title is None:
                plot_title = ''
            else:
                plot_title = title

        if subtitle is not None:
            plot_title += '\n' + subtitle

        #get origin per subplot
        if isinstance(origin, list):
            plot_origin = origin[i]
        else:
            plot_origin = origin

        subplots.append(PlotData(plot_data,plot_title,plot_axis_labels, plot_origin))

    #set range per subplot
    for i, subplot in enumerate(subplots):
        if fix_range == False:
            pass
        elif fix_range == True:
            subplot.range = (range_min,range_max)
        elif type(fix_range) is list:
            subplot.range = fix_range[i]
        else:
            subplot.range = (fix_range[0], fix_range[1])                
        
    #create plots
    if num_plots < num_cols:
        num_cols = num_plots
        
    num_rows = int(round((num_plots+0.5)/num_cols))
    fig, (ax) = plt.subplots(num_rows, num_cols, figsize=size)
    axes = ax.flatten() 

    #set up plots
    for i in range(num_rows*num_cols):
        axes[i].set_visible(False)

    for i, subplot in enumerate(subplots):

        axes[i].set_visible(True)
        axes[i].set_title(subplot.title)

        if subplot.axis_labels is not None:
            axes[i].set_ylabel(subplot.axis_labels[1])
            axes[i].set_xlabel(subplot.axis_labels[0])  

        #set origin
        shape_v = [0,subplot.data.shape[0]]
        shape_h = [0,subplot.data.shape[1]]

        if type(subplot.data) != numpy.ndarray:
            data = subplot.data.as_array() 
        else:
            data = subplot.data
            
        data_origin='lower'

        if 'upper' in subplot.origin:
            shape_v.reverse()
            data_origin='upper'

        if 'right' in subplot.origin:
            shape_h.reverse()
            data = numpy.flip(data,1)

        extent = (*shape_h,*shape_v)
        
        sp = axes[i].imshow(data, cmap=cmap, origin=data_origin, extent=extent)

        im_ratio = subplot.data.shape[0]/subplot.data.shape[1]

        y_axes2 = False
        if isinstance(subplot.data,(AcquisitionData)):
            if axes[i].get_ylabel() == 'angle':
                locs = axes[i].get_yticks()
                location_new = locs[0:-1].astype(int)

                ang = subplot.data.geometry.config.angles

                labels_new = [str(i) for i in numpy.take(ang.angle_data, location_new)]
                axes[i].set_yticklabels(labels_new)
                
                axes[i].set_ylabel('angle / ' + str(ang.angle_unit))

                y_axes2 = axes[i].axes.secondary_yaxis('right')
                y_axes2.set_ylabel('angle / index')
        
                if subplot.data.shape[0] < subplot.data.shape[1]//2:
                    axes[i].set_aspect(1/im_ratio)
                    im_ratio = 1

        if y_axes2: 
            scale = 0.041*im_ratio
            pad = 0.12
        else:
            scale = 0.0467*im_ratio
            pad = 0.02

        plt.tight_layout()
        plt.colorbar(sp, orientation='vertical', ax=axes[i],fraction=scale, pad=pad)

        if subplot.range is not None:
            sp.set_clim(subplot.range[0],subplot.range[1])
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
    
    from cil.framework import ImageData
    from cil.utilities import dataexample
    import os
    import sys
    
    data = dataexample.PEPPERS.get(size=(256,256))
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
                
# %%
