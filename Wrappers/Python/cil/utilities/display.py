#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
# Kyle Pidgeon (UKRI-STFC)


#%%
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageData, DataContainer, BlockDataContainer
from cil.framework.labels import AcquisitionType
import numpy as np
import warnings

import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle

CB_PALETTE = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']

class _PlotData(object):
    def __init__(self, data, title, axis_labels, origin):
        self.data = data
        self.title = title
        self.axis_labels = axis_labels
        self.origin = origin
        self.range = None

def set_origin(data, origin):
    shape_v = [0, data.shape[0]]
    shape_h = [0, data.shape[1]]

    if type(data) != np.ndarray:
        data = data.as_array()

    data_origin='lower'

    if 'upper' in origin:
        shape_v.reverse()
        data_origin='upper'

    if 'right' in origin:
        shape_h.reverse()
        data = np.flip(data,1)

    extent = (*shape_h,*shape_v)
    return data, data_origin, extent

class show_base(object):
    def save(self,filename, **kwargs):
        '''
        Saves the image as a `.png` using matplotlib.figure.savefig()

        matplotlib kwargs can be passed, refer to documentation
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        '''
        file,extension = os.path.splitext(os.path.abspath(filename))
        extension = extension.strip('.')

        extensions = plt.gcf().canvas.get_supported_filetypes()
        extensions = [i for i in extensions.keys()]

        format = kwargs.get('format',None)

        if format is None:
            if extension == '':
                extension = 'png'
        else:
            extension = format

        if extension not in extensions:
            raise ValueError("Extension not valid. Got {0}, backend supports {1}".format(extension,extensions))

        try:
            path_full = file+'.'+extension
            self.figure.set_tight_layout(True)
            self.figure.set_facecolor('w')
            self.figure.savefig(path_full, bbox_inches='tight',**kwargs)
            print("Saved image as {}".format(path_full))
        except PermissionError:
            print("Unable to save image. Permissions denied: {}".format(path_full))
        except:
            print("Unable to save image")


class show1D(show_base):
    """
    This creates and displays 1D plots of pixel values by slicing
    multi-dimensional data.

    The behaviour is as follows: if provided multiple datasets and a single
    slice set (see first example below), one line plot will be generated
    per dataset; if provided a single dataset and multiple sets of slices
    (see second example below), one line plot will be generated per slice
    set;  if provided multiple datasets and multiple slice sets, the
    :math:`i`-th set of slices will apply to the :math:`i`-th dataset, with
    a line plot generated in each case.

    Parameters
    ----------
    data : DataContainer, list of DataContainer, tuple of DataContainer
        Multi-dimensional data to be reduced to 1D.
    slice_list : tuple, list of tuple or list of list of tuple, default=None
        A tuple of (dimension, coordinate) pair, or a list, or nested list, of
        such pairs for slicing `data` (default is None, which is only valid when 1D
        data is passed)
    label : 'default', str, list of str, None, default='default'
        Label(s) to use in the plot's legend. Use `None` to suppress legend.
    title : str, default None
        A title for the plot
    line_colours : str, list of str, default=None
        Colour(s) for each line plot
    line_styles : {"-","--","-.",":"}, list of {"-","--","-.",":"}, default=None
        Linestyle(s) for each line plot
    axis_labels : tuple of str, list of str, default=('Index','Value')
        Axis labels in the form (x_axis_label,y_axis_label)
    size : tuple, default=(8,6)
        The size of the figure

    Attributes
    ----------
    figure : matplotlib.figure.Figure

    Examples
    --------

    This example creates two 2D datasets (images), and uses the provided
    slicing information to generate two plots on the same axis,
    corresponding to the two datasets.

    >>> from cil.utilities.display import show1D
    >>> from cil.utilities.dataexample import PEPPERS
    >>> data = PEPPERS.get()
    >>> data_channel0 = data.get_slice(channel=0)
    >>> data_channel1 = data.get_slice(channel=1)
    >>> show1D([data_channel0, data_channel1], slice_list=[('horizontal_x', 256)],
    ...        label=['Channel 0', 'Channel 1'], line_styles=["--", "-"])

    The following example uses two sets of slicing information applied to a
    single dataset, resulting in two separate plots.

    >>> from cil.utilities.display import show1D
    >>> from cil.utilities.dataexample import PEPPERS
    >>> data = PEPPERS.get()
    >>> slices = [[('channel', 0), ('horizontal_x', 256)], [('channel', 1), ('horizontal_y', 256)]]
    >>> show1D(data, slice_list=slices, title=['Channel 0', 'Channel 1'])
    """

    def __init__(self, data, slice_list=None, label='default', title=None,
                 line_colours=None, line_styles=None, axis_labels=('Index', 'Value'),
                 size=(8,6)):

        self.figure = self._show1d(data, slice_list, labels=label, title=title,
                                   line_colours=line_colours, line_styles=line_styles,
                                   axis_labels=axis_labels, plot_size=size)

    def _extract_vector(self, data, coords):
        """
        Extracts a 1D vector by slicing multi-dimensional data using the
        coordinates provided.

        Parameters
        ----------
        data : DataContainer or numpy.ndarray
            Multi-dimensional data to be reduced to 1D.
        coords : dict
            The dimensions and coordinates used for slicing. If `data` is a
            DataContainer, this should comprise dimensions from
            `data.dimension_labels`. If `data` is a numpy.ndarray, integers
            representing the axes should be used instead.

        Returns
        -------
        numpy.ndarray
            The 1-dimensional pixel flux data extracted from `data`.
        """
        vector = None
        possible_dimensions = None

        if isinstance(data, np.ndarray):
            possible_dimensions = [i for i in range(len(data.shape))]
            if len(possible_dimensions) == 1:
                return data
        elif isinstance(data, DataContainer):
            possible_dimensions = data.dimension_labels
            if len(possible_dimensions) == 1:
                return data.as_array()

        if coords is None:
            raise TypeError(f'Must provide slicing coordinates for multi-dimensional data')

        remaining_dimensions = set(possible_dimensions) - set(coords.keys())
        if len(remaining_dimensions) > 1:
            raise ValueError(f'One remaining dimension required, ' \
                            f'found {len(remaining_dimensions)}: {remaining_dimensions}')

        if isinstance(data, np.ndarray):
            s = data
            for d, i in coords.items():
                if d not in possible_dimensions:
                    raise ValueError(f'Unexpected key "{d}", not in ' \
                                    f'{possible_dimensions}')
                else:
                    s = s.take(indices=i, axis=d)

            vector = s

        elif isinstance(data, DataContainer):
            sliceme = {}
            for k,v in coords.items():
                if k not in possible_dimensions:
                    raise ValueError(f'Unexpected key "{k}", not in ' \
                                    f'{possible_dimensions}')
                else:
                    sliceme[k] = v

            if isinstance(data, AcquisitionData) or isinstance(data, ImageData):
                sliceme['force'] = True

            vector = data.get_slice(**sliceme).as_array()

        return vector

    def _plot_slice(self, ax, data, slice_list=None,
                   label=None, line_colour=None, line_style=None):
        """
        Creates 1D plots of pixel flux from multi-dimensional data and slicing information.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to draw on
        data : DataContainer
            The data to be sliced and plotted
        slice_list : tuple or list of tuples, optional
            (dimension, coordinate) pairs for slicing `data` (default is
            None, which is only valid when 1D data is passed)
        label : str, default=None
            Label to use in the plot's legend
        line_colour : str, default=None
            Colour of the line plot
        line_style : {"-","--","-.",":"}, default=None
            Linestyle to pass to `matplotlib.axes.Axes.plot`
        """

        is_1d = False
        if len(data.shape) == 1:
            is_1d = True

        if isinstance(slice_list, tuple):
            slice_list = [slice_list]

        dims = {}
        if not is_1d:
            try:
                for el in slice_list:
                    dims[el[0]] = el[1]
            except TypeError:
                raise TypeError(f'Expected tuple or list of tuples for slicing, ' \
                                f'received {type(slice_list)}')

        arr = self._extract_vector(data, dims)
        ax.plot(arr, color=line_colour, ls=line_style, label=label)


    def _show1d(self, data, slice_list=None, labels='default', title=None, line_colours=None,
                line_styles=None, axis_labels=('Pixel', 'Pixel value'), plot_size=(8,6)):
        """
        Displays 1D plots of pixel flux from multi-dimensional data and
        slicing information.

        Parameters
        ----------
        data : DataContainer, list of DataContainer or tuple of
        DataContainer
            The data to be sliced and plotted
        slice_list : tuple, list of tuple, or list of list of tuple, optional
            A (dimension, coordinate) pair or a list, or nested list, of such
            pairs for slicing `data` (default is None, which is only valid when 1D
            data is passed)
        labels : 'default', str, list of str, None, default='default'
            Label(s) to use in the plot's legend. Use `None` to suppress legend.
        titles : str, default=None
            A title for the plot
        line_colours : str, list of str, default=None
            Colour(s) for each line plot
        line_styles : {"-","--","-.",":"}, list of {"-","--","-.",":"}, default=None
            Linestyle(s) for each line plot
        axis_labels : tuple of str, list of str, default=('Index','Value')
            Axis labels in the form (x_axis_label,y_axis_label)
        num_cols : int, default=3
            The number of columns in the grid of subplots produced in the
            case of multiple plots
        plot_size : tuple, default=(8,6)
            The size of the figure

        Returns
        -------
        matplotlib.figure.Figure
            The figure created to plot the 1D data
        """

        fig = plt.figure(figsize=plot_size)
        ax = fig.add_subplot(1, 1, 1)

        num_data = 1 if isinstance(data, DataContainer) else len(data)
        colour_cyc = cycle(CB_PALETTE)
        ls_cyc = cycle(["-","--","-.",":"])
        _lbls = labels

        if slice_list is None or isinstance(slice_list, tuple) or isinstance(slice_list[0], tuple):

            for i in range(num_data):
                _data = data if isinstance(data, DataContainer) else data[i]
                _cl = next(colour_cyc) if line_colours is None else line_colours[i]
                _ls = next(ls_cyc) if line_styles is None else line_styles[i]
                if labels is None:
                    _lbl = None
                elif labels == 'default':
                    _lbl = f'Dataset {i}'
                else:
                    _lbl = labels[i]
                self._plot_slice(ax, _data, slice_list, label=_lbl,
                                line_colour=_cl, line_style=_ls)

        elif isinstance(slice_list[0], list):

            if labels == 'default' or labels is None:
                _lbls =  [None]*(len(slice_list)*num_data)

            if num_data == 1:
                for i, sl in enumerate(slice_list):
                    _cl = next(colour_cyc) if line_colours is None else line_colours[i]
                    _ls = next(ls_cyc) if line_styles is None else line_styles[i]
                    if labels == 'default':
                        _lbls[i] = ', '.join(f'{c[0]}={c[1]}' for c in sl)
                    self._plot_slice(ax, data, sl, label=_lbls[i], line_colour=_cl,
                                     line_style=_ls)
            else:
                for i, sl in enumerate(slice_list):
                    _cl = next(colour_cyc) if line_colours is None else line_colours[i]
                    _ls = next(ls_cyc) if line_styles is None else line_styles[i]
                    if labels == 'default':
                        _lbls[i] = f'Dataset {i}, ' + \
                                   ', '.join(f'{c[0]}={c[1]}' for c in sl)
                    self._plot_slice(ax, data[i], sl, label=_lbls[i], line_colour=_cl,
                                     line_style=_ls)

        else:
            raise TypeError(f'Unexpected type for slice_list: {type(slice_list)}, expected: (tuple, list of tuples, list of list of tuples)')

        ax.set_title(title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if labels is not None:
            fig.legend(loc='upper left', bbox_to_anchor=(1., 0., 1., 1.))
        plt.tight_layout()
        fig2 = plt.gcf()
        return fig2


class show2D(show_base):
    '''This plots 2D slices from cil DataContainer types.

    Plots 1 or more 2D plots in an (n x num_cols) matrix.
    Can plot multiple slices from one 3D dataset, or compare multiple datasets
    Inputs can be single arguments or list of arguments that will be sequentially applied to subplots
    If no slice_list is passed a 3D dataset will display the centre slice of the outer dimension, a 4D dataset will show the centre slices of the two outer dimension.


    Parameters
    ----------
    datacontainers: ImageData, AcquisitionData, list of  ImageData / AcquisitionData, BlockDataContainer
        The DataContainers to be displayed
    title: string, list of strings, optional
        The title for each figure
    slice_list: tuple, int, list of tuples, list of ints, optional
        The slices to show. A list of integers will show slices for the outer dimension. For 3D datacontainers single slice: (direction, index). For 4D datacontainers two slices: [(direction0, index),(direction1, index)].
    fix_range: boolean, tuple, list of tuples
        Sets the display range of the data. `True` sets all plots to the global (min, max).
    axis_labels: tuple, list of tuples, optional
        The axis labels for each figure e.g. ('x','y')
    origin: string, list of strings
        Sets the display origin. 'lower/upper-left/right'
    cmap: str, list or tuple of strings
        Sets the colour map of the plot (see matplotlib.pyplot). If passed a list or tuple of the
        length of datacontainers, allows to set a different color map for each datacontainer.
    num_cols: int
        Sets the number of columns of subplots to display
    size: tuple
        Figure size in inches

    Returns
    -------
    matplotlib.figure.Figure
        returns a matplotlib.pyplot figure object
    '''

    def __init__(self,datacontainers, title=None, slice_list=None, fix_range=False, axis_labels=None, origin='lower-left', cmap='gray', num_cols=2, size=(15,15)):

        self.figure = self.__show2D(datacontainers, title=title, slice_list=slice_list, fix_range=fix_range, axis_labels=axis_labels, origin=origin, cmap=cmap, num_cols=num_cols, size=size)

    def __show2D(self,datacontainers, title=None, slice_list=None, fix_range=False, axis_labels=None, origin='lower-left', cmap='gray', num_cols=2, size=(15,15)):

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
                    if hasattr(data, 'get_slice'):
                        if type(cut_axis[0]) is int:
                            cut_axis[0] = data.dimension_labels[cut_axis[0]]
                        if type(cut_axis[1]) is int:
                            cut_axis[1] = data.dimension_labels[cut_axis[1]]

                        temp_dict = {cut_axis[0]:cut_slices[0], cut_axis[1]:cut_slices[1]}
                        if type(data) == DataContainer:
                            plot_data = data.get_slice(**temp_dict)
                        else:
                            plot_data = data.get_slice(**temp_dict, force=True)
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
                    if hasattr(data, 'get_slice'):
                        if type(cut_axis) is int:
                            cut_axis = data.dimension_labels[cut_axis]
                        temp_dict = {cut_axis:cut_slice}
                        if type(data) == DataContainer:
                            plot_data = data.get_slice(**temp_dict)
                        else:
                            plot_data = data.get_slice(**temp_dict, force=True)
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

            subplots.append(_PlotData(plot_data,plot_title,plot_axis_labels, plot_origin))

        #set range per subplot
        for i, subplot in enumerate(subplots):
            if fix_range is False:
                pass
            elif fix_range is True:
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
            data, data_origin, extent = set_origin(subplot.data, subplot.origin)
            if isinstance(cmap, (list, tuple)):
                dcmap = cmap[i]
            else:
                dcmap = cmap
            sp = axes[i].imshow(data, cmap=dcmap, origin=data_origin, extent=extent)

            im_ratio = subplot.data.shape[0]/subplot.data.shape[1]

            y_axes2 = False
            if isinstance(subplot.data,(AcquisitionData)):
                if axes[i].get_ylabel() == 'angle':
                    locs = axes[i].get_yticks()
                    location_new = locs[0:-1].astype(int)

                    ang = subplot.data.geometry.config.angles

                    labels_new = ["{:.2f}".format(i) for i in np.take(ang.angle_data, location_new)]
                    axes[i].set_yticks(location_new, labels=labels_new)

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

            plt.colorbar(sp, orientation='vertical', ax=axes[i],fraction=scale, pad=pad)

            if subplot.range is not None:
                sp.set_clim(subplot.range[0],subplot.range[1])

        fig.set_tight_layout(True)
        fig.set_facecolor('w')

        #plt.show() creates a new figure so we save a copy to return
        fig2 = plt.gcf()
        plt.show()
        return fig2

def plotter2D(datacontainers, title=None, slice_list=None, fix_range=False, axis_labels=None, origin='lower-left', cmap='gray', num_cols=2, size=(15,15)):
    '''Alias of show2D'''
    return show2D(datacontainers, title=title, slice_list=slice_list, fix_range=fix_range, axis_labels=axis_labels, origin=origin, cmap=cmap, num_cols=num_cols, size=size)

class _Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

class _ShowGeometry(object):
    def __init__(self, acquisition_geometry, image_geometry=None):
        if AcquisitionType.DIM2 & acquisition_geometry.dimension:
            self.ndim = 2
            sys = acquisition_geometry.config.system
            if acquisition_geometry.geom_type == 'cone':
                ag_temp = AcquisitionGeometry.create_Cone3D([*sys.source.position,0], [*sys.detector.position,0], [*sys.detector.direction_x,0],[0,0,1],[*sys.rotation_axis.position,0],[0,0,1])
            else:
                ag_temp = AcquisitionGeometry.create_Parallel3D([*sys.ray.direction,0], [*sys.detector.position,0], [*sys.detector.direction_x,0],[0,0,1],[*sys.rotation_axis.position,0],[0,0,1])


            ag_temp.config.panel = acquisition_geometry.config.panel
            ag_temp.set_angles(acquisition_geometry.angles)
            ag_temp.set_labels(['vertical', *acquisition_geometry.dimension_labels])

            self.acquisition_geometry = ag_temp

        elif acquisition_geometry.channels > 1:
            self.ndim = 3
            self.acquisition_geometry = acquisition_geometry.get_slice(channel=0)
        else:
            self.acquisition_geometry = acquisition_geometry
            self.ndim = 3

        if image_geometry is None:
            self.image_geometry=self.acquisition_geometry.get_ImageGeometry()
        else:
            self.image_geometry = image_geometry


        len1 = self.acquisition_geometry.config.panel.num_pixels[0] * self.acquisition_geometry.config.panel.pixel_size[0]
        len2 = self.acquisition_geometry.config.panel.num_pixels[1] * self.acquisition_geometry.config.panel.pixel_size[1]
        self.scale = max(len1,len2)/5


        self.handles = []
        self.labels = []

    def draw(self, elev=35, azim=35, view_distance=10, grid=False, figsize=(10,10), fontsize=10):

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.text_options = {   'horizontalalignment': 'center',
                                'verticalalignment': 'center',
                                'fontsize': fontsize }


        self.display_world()

        if self.acquisition_geometry.geom_type == 'cone':
            self.display_source()
        else:
            self.display_ray()

        self.display_object()

        self.display_detector()


        if grid is False:
            self.ax.set_axis_off()

        self.ax.view_init(elev=elev, azim=azim)
        self.ax.dist = view_distance

        #to force aspect ratio 1:1:1
        world_limits = self.ax.get_w_lims()
        self.ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

        l = self.ax.plot(np.NaN, np.NaN, '-', color='none', label='')[0]

        for i in range(3):
            self.handles.insert(2,l)
            self.labels.insert(2,'')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ax.legend(self.handles, self.labels, loc='upper left', bbox_to_anchor= (0, 1), ncol=3,
                        borderaxespad=0, frameon=False,fontsize=self.text_options.get('fontsize'))


        self.fig.set_tight_layout(True)
        self.fig.set_facecolor('w')

        #plt.show() creates a new figure so we save a copy to return
        fig2 = plt.gcf()
        plt.show()
        return fig2

    def display_world(self):

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')

        if self.ndim == 3:
            self.ax.set_zlabel('Z axis')
        else:
            self.ax.set_zticks([])

        #origin and coordinate frame
        Oo = np.zeros(3)
        self.ax.scatter3D(*Oo, marker='o', alpha=1,color='k',lw=1)
        h = mlines.Line2D([], [], color='k',linestyle='solid', markersize=12, label='world coordinate system')

        labels = ['$x$','$y$','$z$']
        for i in range(self.ndim):
            axis = np.zeros(3)
            axis[i] = 1 * self.scale

            a = _Arrow3D(*zip(Oo,axis*2), mutation_scale=20,lw=1, arrowstyle="->", color="k")
            self.ax.add_artist(a)

            self.ax.text(*(axis*2.2),labels[i], **self.text_options)

        self.handles.append(h)
        self.labels.append(h.get_label())

    def detector_vertex(self):
        # detector corners
        det_size  = (np.array(self.acquisition_geometry.config.panel.num_pixels) * np.array(self.acquisition_geometry.config.panel.pixel_size))/2
        det_rows_dir = self.acquisition_geometry.config.system.detector.direction_x

        if self.ndim == 3:
            det_v = self.acquisition_geometry.config.system.detector.direction_y * det_size[1]
            det_h = det_rows_dir * det_size[0]

            rt = det_h + det_v + self.acquisition_geometry.config.system.detector.position
            lt = -det_h + det_v + self.acquisition_geometry.config.system.detector.position
            lb = -det_h - det_v + self.acquisition_geometry.config.system.detector.position
            rb = det_h - det_v + self.acquisition_geometry.config.system.detector.position

            return [rb, lb, lt, rt]

        else:
            det_h = det_rows_dir * det_size[0]
            r = det_h  + self.acquisition_geometry.config.system.detector.position
            l = -det_h  + self.acquisition_geometry.config.system.detector.position
            return [r, l]


    def display_detector(self):

        do = self.acquisition_geometry.config.system.detector.position
        det = self.detector_vertex()

        #mark data origin
        if 'right' in self.acquisition_geometry.config.panel.origin:
            if self.ndim==2 or 'bottom' in self.acquisition_geometry.config.panel.origin:
                pix0 = det[0]
            else:
                pix0 = det[3]
        else:
            if self.ndim==2 or 'bottom' in self.acquisition_geometry.config.panel.origin:
                pix0 = det[1]
            else:
                pix0 = det[2]

        det_rows_dir = self.acquisition_geometry.config.system.detector.direction_x

        x = _Arrow3D(*zip(do, self.scale * det_rows_dir + do), mutation_scale=20,lw=1, arrowstyle="-|>", color="b")
        self.ax.add_artist(x)
        self.ax.text(*(1.2 * self.scale * det_rows_dir + do),r'$D_x$', **self.text_options)

        if self.ndim == 3:
            det_col_dir = self.acquisition_geometry.config.system.detector.direction_y
            y = _Arrow3D(*zip(do, self.scale * det_col_dir + do), mutation_scale=20,lw=1, arrowstyle="-|>", color="b")
            self.ax.add_artist(y)
            self.ax.text(*(1.2 * self.scale * det_col_dir + do),r'$D_y$', **self.text_options)

        handles=[
            self.ax.scatter3D(*do, marker='o', alpha=1,color='b',lw=1, label='detector position'),
            mlines.Line2D([], [], color='b',linestyle='solid', markersize=12, label='detector direction'),
            self.ax.plot3D(*zip(*det, det[0]), color='b',ls='dotted',alpha=1, label='detector')[0],
            self.ax.scatter3D(*pix0, marker='x', alpha=1,color='b',lw=1,s=50, label='data origin (pixel 0)'),
        ]

        for x in handles:
            self.handles.append(x)
            self.labels.append(x.get_label())

    def display_object(self):

        ro = self.acquisition_geometry.config.system.rotation_axis.position
        h0 = self.ax.scatter3D(*ro, marker='o', alpha=1,color='r',lw=1,label='rotation axis position')
        self.handles.append(h0)
        self.labels.append(h0.get_label())

        if self.ndim == 3:
            # rotate axis arrow
            r1 = ro +  self.acquisition_geometry.config.system.rotation_axis.direction * self.scale * 2
            arrow3 = _Arrow3D(*zip(ro,r1), mutation_scale=20,lw=1, arrowstyle="-|>", color="r")
            self.ax.add_artist(arrow3)

            a = self.acquisition_geometry.config.system.rotation_axis.direction


            # draw reco
            x = np.array([self.image_geometry.get_min_x(), self.image_geometry.get_max_x()])
            y = np.array([self.image_geometry.get_min_y(), self.image_geometry.get_max_y()])
            z = np.array([self.image_geometry.get_min_z(), self.image_geometry.get_max_z()])

            combos = [
                ((x[0],y[0],z[0]),(x[0],y[1],z[0])),
                ((x[0],y[1],z[0]),(x[1],y[1],z[0])),
                ((x[1],y[1],z[0]),(x[1],y[0],z[0])),
                ((x[1],y[0],z[0]),(x[0],y[0],z[0])),
                ((x[0],y[0],z[1]),(x[0],y[1],z[1])),
                ((x[0],y[1],z[1]),(x[1],y[1],z[1])),
                ((x[1],y[1],z[1]),(x[1],y[0],z[1])),
                ((x[1],y[0],z[1]),(x[0],y[0],z[1])),
                ((x[0],y[0],z[0]),(x[0],y[0],z[1])),
                ((x[0],y[1],z[0]),(x[0],y[1],z[1])),
                ((x[1],y[1],z[0]),(x[1],y[1],z[1])),
                ((x[1],y[0],z[0]),(x[1],y[0],z[1])),
            ]

            if np.allclose(a,[0,0,1]):
                axis_rotation = np.eye(3)
            elif np.allclose(a,[0,0,-1]):
                axis_rotation = np.eye(3)
                axis_rotation[1][1] = -1
                axis_rotation[2][2] = -1
            else:
                vx = np.array([[0, 0, -a[0]], [0, 0, -a[1]], [a[0], a[1], 0]])
                axis_rotation = np.eye(3) + vx + vx.dot(vx) *  1 / (1 + a[2])

            rotation_matrix = np.matrix.transpose(axis_rotation)

            count = 0
            for x in combos:
                s = rotation_matrix.dot(np.asarray(x[0]).reshape(3,1))
                e = rotation_matrix.dot(np.asarray(x[1]).reshape(3,1))

                x_data = float(s[0]) + ro[0], float(e[0]) + ro[0]
                y_data = float(s[1]) + ro[1], float(e[1]) + ro[1]
                z_data = float(s[2]) + ro[2], float(e[2]) + ro[2]

                self.ax.plot3D(x_data,y_data,z_data, color="r",ls='dotted',alpha=1)

                if count == 0:
                    vox0=(x_data[0],y_data[0],z_data[0])
                count+=1
        else:
            # draw square
            x = [self.image_geometry.get_min_x(), self.image_geometry.get_max_x()]
            y = [self.image_geometry.get_min_y(), self.image_geometry.get_max_y()]
            vertex = np.array([(x[0],y[0],0),(x[0],y[1],0),(x[1],y[1],0),(x[1],y[0],0)]) + ro
            self.ax.plot3D(*zip(*vertex, vertex[0]), color='r',ls='dotted',alpha=1)
            vox0=vertex[0]
            rotation_matrix = np.eye(3)

        #rotation direction
        points = 36
        x = [None]*points
        y = [None]*points
        z = [None]*points

        for i in range(points):
            theta = i * (np.pi * 1.8) /36
            point_i = np.array([np.sin(theta),-np.cos(theta),0]).reshape(3,1)
            point_rot = -self.scale*0.5*rotation_matrix.dot(point_i)

            x[i] = float(point_rot[0] + ro[0])
            y[i] = float(point_rot[1] + ro[1])
            z[i] = float(point_rot[2] + ro[2])

        self.ax.plot3D(x,y,z, color='r',ls="dashed",alpha=1)
        arrow4 = _Arrow3D(x[-2:],y[-2:],z[-2:],mutation_scale=20,lw=1, arrowstyle="-|>", color="r")
        self.ax.add_artist(arrow4)

        handles = [
            mlines.Line2D([], [], color='r',linestyle='solid', markersize=12, label='rotation axis direction'),
            mlines.Line2D([], [], color='r',linestyle='dotted', markersize=15, label='image geometry'),
            self.ax.scatter3D(*vox0, marker='x', alpha=1,color='r',lw=1,s=50, label='data origin (voxel 0)'),
            mlines.Line2D([], [], color='r',linestyle='dashed', markersize=12, label=r'rotation direction $\theta$')
        ]

        for x in handles:
            self.handles.append(x)
            self.labels.append(x.get_label())

    def display_source(self):

        so = self.acquisition_geometry.config.system.source.position
        det = self.detector_vertex()

        for i in range(len(det)):
            self.ax.plot3D(*zip(so,det[i]), color='#D4BD72',ls="dashed",alpha=0.4)

        self.ax.plot3D(*zip(so,self.acquisition_geometry.config.system.detector.position), color='#D4BD72',ls="solid",alpha=1)[0],

        h0 = self.ax.scatter3D(*so, marker='*', alpha=1,color='#D4BD72',lw=1, label='source position', s=100)

        self.handles.append(h0)
        self.labels.append(h0.get_label())

    def display_ray(self):

        det = self.detector_vertex()
        det.append(self.acquisition_geometry.config.system.detector.position)

        dist = np.sqrt(np.sum(self.acquisition_geometry.config.system.detector.position**2))*2

        if dist < 0.01:
            dist = self.acquisition_geometry.config.panel.num_pixels[0] * self.acquisition_geometry.config.panel.pixel_size[0]

        rays = det - self.acquisition_geometry.config.system.ray.direction*dist

        for i in range(len(rays)):
            h0 = self.ax.plot3D(*zip(rays[i],det[i]), color='#D4BD72',ls="dashed",alpha=0.4, label='ray direction')[0]
            arrow = _Arrow3D(*zip(rays[i],rays[i]+self.acquisition_geometry.config.system.ray.direction*self.scale ),mutation_scale=20,lw=1, arrowstyle="-|>", color="#D4BD72")
            self.ax.add_artist(arrow)

        self.handles.append(h0)
        self.labels.append(h0.get_label())

class show_geometry(show_base):
    '''
    Displays a schematic of the acquisition geometry
    for 2D geometries elevation and azimuthal cannot be changed


    Parameters
    ----------
    acquisition_geometry: AcquisitionGeometry
        CIL acquisition geometry
    image_geometry: ImageGeometry, optional
        CIL image geometry
    elevation: float
        Camera elevation in degrees, 3D geometries only, default=20
    azimuthal: float
        Camera azimuthal in degrees, 3D geometries only, default=-35
    view_distance: float
        Camera view distance, default=10
    grid: boolean
        Show figure axis, default=False
    figsize: tuple (x, y)
        Set figure size (inches), default (10,10)
    fontsize: int
        Set fontsize, default 10

    Returns
    -------
    matplotlib.figure.Figure
        returns a matplotlib.pyplot figure object
    '''


    def __init__(self,acquisition_geometry, image_geometry=None, elevation=20, azimuthal=-35, view_distance=10, grid=False, figsize=(10,10), fontsize=10):
        if AcquisitionType.DIM2 & acquisition_geometry.dimension:
            elevation = 90
            azimuthal = 0

        self.display = _ShowGeometry(acquisition_geometry, image_geometry)
        self.figure = self.display.draw(elev=elevation, azim=azimuthal, view_distance=view_distance, grid=grid, figsize=figsize, fontsize=fontsize)
