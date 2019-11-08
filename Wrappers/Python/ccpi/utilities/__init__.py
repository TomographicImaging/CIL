# imports for plotting
from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy



def psnr(img1, img2, data_range=1):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 1000
    return 20 * numpy.log10(data_range / numpy.sqrt(mse))


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