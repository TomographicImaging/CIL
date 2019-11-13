# imports for plotting
from __future__ import print_function, division
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy

from IPython.display import HTML
import random


def display_slice(container, direction, title, cmap, minmax, size, axis_labels):
    
        
    def get_slice_3D(x):
        
        if direction == 0:
            img = container[x]
            x_lim = container.shape[2]
            y_lim = container.shape[1]
            x_label = axis_labels[2]
            y_label = axis_labels[1] 
            
        elif direction == 1:
            img = container[:,x,:]
            x_lim = container.shape[2]
            y_lim = container.shape[0] 
            x_label = axis_labels[2]
            y_label = axis_labels[0]             
            
        elif direction == 2:
            img = container[:,:,x]
            x_lim = container.shape[1]
            y_lim = container.shape[0]    
            x_label = axis_labels[1]
            y_label = axis_labels[0]             
        
        if size is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=size)
        
        if isinstance(title, (list, tuple)):
            dtitle = title[x]
        else:
            dtitle = title
        
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=(1,.05), height_ratios=(1,))
        # image
        ax = fig.add_subplot(gs[0, 0])
      
        ax.set_xlabel(x_label)     
        ax.set_ylabel(y_label)
 
        aximg = ax.imshow(img, cmap=cmap, origin='upper', extent=(0,x_lim,y_lim,0))
        aximg.set_clim(minmax)
        ax.set_title(dtitle + " {}".format(x))
        # colorbar
        ax = fig.add_subplot(gs[0, 1])
        plt.colorbar(aximg, cax=ax)
        plt.tight_layout()
        plt.show(fig)
        
    return get_slice_3D

    
def islicer(data, direction, title="", slice_number=None, cmap='gray', minmax=None, size=None, axis_labels=None):

    '''Creates an interactive integer slider that slices a 3D volume along direction
    
    :param data: DataContainer or numpy array
    :param direction: slice direction, int, should be 0,1,2 or the axis label
    :param title: optional title for the display
    :slice_number: int start slice number, optional. If None defaults to center slice
    :param cmap: matplotlib color map
    :param minmax: colorbar min and max values, defaults to min max of container
    :param size: int or tuple specifying the figure size in inch. If int it specifies the width and scales the height keeping the standard matplotlib aspect ratio 
    '''
    
    if axis_labels is None:
        if hasattr(data, "dimension_labels"):
            axis_labels = [data.dimension_labels[0],data.dimension_labels[1],data.dimension_labels[2]]
        else:
            axis_labels = ['X', 'Y', 'Z']

    
    if hasattr(data, "as_array"):
        container = data.as_array()
        
        if not isinstance (direction, int):
            if direction in data.dimension_labels.values():
                direction = data.get_dimension_axis(direction)                             

    elif isinstance (data, numpy.ndarray):
        container = data
        
    if slice_number is None:
        slice_number = int(data.shape[direction]/2)
        
    slider = widgets.IntSlider(min=0, max=data.shape[direction]-1, step=1, 
                             value=slice_number, continuous_update=False, description=axis_labels[direction])

    if minmax is None:
        amax = container.max()
        amin = container.min()
    else:
        amin = min(minmax)
        amax = max(minmax)
    
    if isinstance (size, (int, float)):
        default_ratio = 6./8.
        size = ( size , size * default_ratio )
    
    interact(display_slice(container, 
                           direction, 
                           title=title, 
                           cmap=cmap, 
                           minmax=(amin, amax),
                           size=size, axis_labels=axis_labels),
             x=slider);
    
    return slider
    

def link_islicer(*args):
    '''links islicers IntSlider widgets'''
    linked = [(widg, 'value') for widg in args]
    # link pair-wise
    pairs = [(linked[i+1],linked[i]) for i in range(len(linked)-1)]
    for pair in pairs:
        widgets.link(*pair)


# https://stackoverflow.com/questions/31517194/how-to-hide-one-specific-cell-input-or-output-in-ipython-notebook/52664156

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)    