#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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

from cil.framework import Processor, AcquisitionData
from cil.utilities.display import show2D
import numpy
import logging
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

class FluxNormaliser(Processor):
    r'''
    Flux normalisation based on float or region of interest

    This processor reads in an AcquisitionData and normalises it based on
    a float or array of float values, or a region of interest.
    
    The normalised image :math:`I_{norm}` is calculated from the original image
    :math:`I` by

    .. math:: I_{norm} = I\frac{n}{F}

    where :math:`F` is the flux and :math:`n` is the norm_value

    Parameters:
    -----------
    flux: float or list of floats (optional)
        The value to divide the image by. If flux is a list it must have length 
        equal to the number of angles in the dataset. If flux=None, calculate
        flux from the roi, default is None.
    
    roi: dict (optional)
        Dictionary describing the region of interest containing the background
        in the image. The roi is specified as `{‘axis_name1’:(start,stop), 
        ‘axis_name2’:(start,stop)}`, where the key is the axis name to calculate
        the flux from. If the dataset has an axis which is not specified in the 
        roi dictionary, all data from that axis will be used in the calculation.
        The image is divided by the mean value in the roi. If None, specify flux 
        directly, default is None.

    norm_value: float (optional)
        The value to multiply the image by. If None, the mean flux value will be used,
         either the flux provided or the mean value in the roi across all 
         projections, default is None.

    Returns:
    --------
    Output: AcquisitionData normalised by flux or mean intensity in roi

    Example
    -------
    >>> from cil.processors import FluxNormaliser
    >>> processor = FluxNormaliser(flux=10)
    >>> processor.set_input(data)
    >>> data_norm = processor.get_output()

    Example
    -------
    >>> from cil.processors import FluxNormaliser
    >>> processor = FluxNormaliser(flux=np.arange(1,2,(2-1)/(data.get_dimension_size('angle'))))
    >>> processor.set_input(data)
    >>> data_norm = processor.get_output()

    Example
    -------
    >>> from cil.processors import FluxNormaliser
    >>> processor = FluxNormaliser(flux=10)
    >>> processor.set_input(data)
    >>> data_norm = processor.get_output()

    Note
    ----
    The roi indices provided are start inclusive, stop exclusive.
    All elements along a dimension will be included if the axis does not appear 
    in the roi dictionary
    '''

    def __init__(self, flux=None, roi=None, norm_value=None):
            kwargs = {
                    'flux'  : flux,
                    'roi' : roi,
                    'roi_slice' : None,
                    'roi_axes' : None,
                    'norm_value' : norm_value
                    }
            super(FluxNormaliser, self).__init__(**kwargs)
            
    def check_input(self, dataset):
        
        if not (type(dataset), AcquisitionData):
            raise TypeError("Expected AcquistionData, found {}"
                            .format(type(dataset)))

        if self.roi is not None:
            if self.flux is not None:
                raise ValueError("Please specify either flux or roi, not both")

            else:
                if isinstance(self.roi, dict):
                    if not all (r in dataset.dimension_labels for r in self.roi):
                        raise ValueError("roi labels must be in the dataset dimension_labels, found {}"
                                        .format(str(self.roi)))

                    slc = [slice(None)]*len(dataset.shape)
                    axes=[]

                    for r in self.roi:
                        # only allow roi to be specified in horizontal and vertical
                        if (r != 'horizontal' and r != 'vertical'):
                            raise ValueError("roi must be 'horizontal' or 'vertical', found '{}'"
                                .format(str(r)))
                    
                    dimension_label_list = list(dataset.dimension_labels)
                    # loop through all dimensions in the dataset apart from angle
                    if 'angle' in dimension_label_list:
                        dimension_label_list.remove('angle')

                    for d in dimension_label_list:
                        # check the dimension is in the user specified roi
                        if d in self.roi:
                            # check indices are ints
                            if not all(isinstance(i, int) for i in self.roi[d]):
                                raise TypeError("roi values must be int, found {} and {}"
                                .format(str(type(self.roi[d][0])), str(type(self.roi[d][1]))))
                            # check indices are in range
                            elif (self.roi[d][0] >= self.roi[d][1]) or (self.roi[d][0] < 0) or self.roi[d][1] > dataset.get_dimension_size(d):
                                raise ValueError("roi values must be start > stop and between 0 and {}, found start={} and stop={} for direction '{}'"
                                .format(str(dataset.get_dimension_size(d)), str(self.roi[d][0]), str(self.roi[d][1]), d ))
                            # create slice
                            else:
                                ax = dataset.get_dimension_axis(d)
                                slc[ax] = slice(self.roi[d][0], self.roi[d][1])
                                axes.append(ax)
                            
                        # if the dimension is not in roi, average across the whole dimension
                        else:
                            ax = dataset.get_dimension_axis(d)
                            axes.append(ax)
                            self.roi.update({d:(0,dataset.get_dimension_size(d))})

                    self.flux = numpy.mean(dataset.array[tuple(slc)], axis=tuple(axes))
                    
                    dataset_range = numpy.max(dataset.array, axis=tuple(axes)) - numpy.min(dataset.array, axis=tuple(axes)) 

                    if (numpy.mean(self.flux) > dataset.mean()):
                        if numpy.mean(self.flux/dataset_range) < 0.9:
                            log.warning('Warning: mean value in selected roi is more than 10 percent of data range - may not represent the background')
                    else:
                        if numpy.mean(self.flux/dataset_range) > 0.1:
                            log.warning('Warning: mean value in selected roi is more than 10 percent of data range - may not represent the background')

                    self.roi_slice = slc
                    self.roi_axes = axes
                    
                else:
                    raise TypeError("roi must be a dictionary, found {}"
                    .format(str(type(self.roi))))
        else:
            if self.flux is None:
                raise ValueError("Please specify flux or roi")

        flux_size = (numpy.shape(self.flux))
        if len(flux_size) > 0:
            data_size = numpy.shape(dataset.geometry.angles)
            if data_size != flux_size:
                raise ValueError("Flux must be a scalar or array with length \
                                    \n = data.geometry.angles, found {} and {}"
                                    .format(flux_size, data_size))
            if numpy.any(self.flux==0):
                raise ValueError('Flux value can\'t be 0, provide a different flux\
                                  or region of interest with non-zero values')
        else:
            if self.flux==0:
                raise ValueError('Flux value can\'t be 0, provide a different flux\
                                  or region of interest with non-zero values')
            
        if self.norm_value is None:
            self.norm_value = numpy.mean(self.flux)
            
        return True

    def preview_configuration(self, angle=None, channel=None, log=False):
        '''
        Preview the FluxNormalisation processor configuration for roi mode.
        Plots the region of interest on the image and the mean, maximum and 
        minimum intensity in the roi.
        
        Parameters:
        -----------
        angle: float or str (optional)
            Angle to plot, default=None displays the data with the minimum
            and maximum pixel values in the roi, otherwise the angle to display 
            can be specified as a float and the closest angle will be displayed.
            For 2D data, the roi is plotted on the sinogram.

        channel: int (optional)
            The channel to plot, default=None displays the central channel if
            the data has channels

        log: bool (optional)
            If True, plot the image with a log scale, default is False
        '''
        if self.roi_slice is None:
            raise ValueError('Preview available with roi, run `processor= FluxNormaliser(roi=roi)` then `set_input(data)`')
        else:
            data = self.get_input()
            
            min = numpy.min(data.array[tuple(self.roi_slice)], axis=tuple(self.roi_axes))
            max = numpy.max(data.array[tuple(self.roi_slice)], axis=tuple(self.roi_axes))
            plt.figure()
            if data.geometry.dimension == '3D':
                if angle is None:
                    if 'angle' in data.dimension_labels:
                        self._plot_slice_roi(angle_index=numpy.argmin(min), channel_index=channel, log=log, ax=221)
                        self._plot_slice_roi(angle_index=numpy.argmax(max), channel_index=channel, log=log, ax=222)
                    else:
                        self._plot_slice_roi(log=log, channel_index=channel, ax=211)
                else:
                    if 'angle' in data.dimension_labels:
                        angle_index = numpy.argmin(numpy.abs(angle-data.geometry.angles))
                        self._plot_slice_roi(angle_index=angle_index, channel_index=channel, log=log, ax=211)
                    else:
                        self._plot_slice_roi(log=log, channel_index=channel, ax=211)
                        
            # if data is 2D plot roi on all angles
            elif data.geometry.dimension == '2D':
                if angle is None:
                    self._plot_slice_roi(channel_index=channel, log=log, ax=211)
                else:
                    raise ValueError("Cannot plot ROI for a single angle on 2D data, please specify angle=None to plot ROI on the sinogram")
            
            plt.subplot(212)
            if len(data.geometry.angles)==1:
                plt.plot(data.geometry.angles, self.flux, '.r', label='Mean')
                plt.plot(data.geometry.angles, min,'.k', label='Minimum')
                plt.plot(data.geometry.angles, max,'.k', label='Maximum')
            else:
                plt.plot(data.geometry.angles, self.flux, 'r', label='Mean')
                plt.plot(data.geometry.angles, min,'--k', label='Minimum')
                plt.plot(data.geometry.angles, max,'--k', label='Maximum')
            plt.legend()
            plt.xlabel('Angle')
            plt.ylabel('Intensity in roi')
            plt.grid()
            plt.tight_layout()
            plt.show()
            
    def _plot_slice_roi(self, angle_index=None, channel_index=None, log=False, ax=111):
        
        data = self.get_input()
        if angle_index is not None and 'angle' in data.dimension_labels:
            data_slice = data.get_slice(angle=angle_index)
        else:
            data_slice = data
        
        if 'channel' in data.dimension_labels:
            if channel_index is None:
                channel_index = int(data_slice.get_dimension_size('channel')/2)
            data_slice = data_slice.get_slice(channel=channel_index)
        else:
            if channel_index is not None:
                raise ValueError("Channel not found")

        if len(data_slice.shape) != 2:
            raise ValueError("Data shape not compatible with preview_configuration(), data must have at least two of 'horizontal', 'vertical' and 'angle'")

        extent = [0, data_slice.shape[1], 0, data_slice.shape[0]]
        if 'angle' in data_slice.dimension_labels:
            min_angle = data_slice.geometry.angles[0]
            max_angle = data_slice.geometry.angles[-1]
            for i, d in enumerate(data_slice.dimension_labels):
                if d !='angle':
                    extent[i*2]=min_angle
                    extent[i*2+1]=max_angle

        plt.subplot(ax)
        if log:
            im = plt.imshow(numpy.log(data_slice.array), cmap='gray',aspect='equal', origin='lower', extent=extent)
            plt.gcf().colorbar(im, ax=plt.gca())
        else:
            im = plt.imshow(data_slice.array, cmap='gray',aspect='equal', origin='lower', extent=extent)
            plt.gcf().colorbar(im, ax=plt.gca())

        h = data_slice.dimension_labels[1]
        v = data_slice.dimension_labels[0]

        if h == 'angle':
            h_min = min_angle
            h_max = max_angle
        else:
            h_min = self.roi[h][0]
            h_max = self.roi[h][1]

        if v == 'angle':
            v_min = min_angle
            v_max = max_angle
        else:
            v_min = self.roi[v][0]
            v_max = self.roi[v][1]

        plt.plot([h_min, h_max],[v_min, v_min],'--r')
        plt.plot([h_min, h_max],[v_max, v_max],'--r')

        plt.plot([h_min, h_min],[v_min, v_max],'--r')
        plt.plot([h_max, h_max],[v_min, v_max],'--r')
        
        title = 'ROI'
        if angle_index is not None:
            title += ' angle = ' + str(data.geometry.angles[angle_index])
        if channel_index is not None:
            title += ' channel = ' + str(channel_index)
        plt.title(title)

        plt.xlabel(h)
        plt.ylabel(v)

    def process(self, out=None):
        
        data = self.get_input()

        if out is None:
            out = data.copy()

        flux_size = (numpy.shape(self.flux))
        f = self.flux
        
        if 'angle' in data.dimension_labels:
            proj_axis = data.get_dimension_axis('angle')
            slice_proj = [slice(None)]*len(data.shape)
            slice_proj[proj_axis] = 0
        
            for i in range(len(data.geometry.angles)):
                if len(flux_size) > 0:
                    f = self.flux[i]
                slice_proj[proj_axis] = i
                out.array[tuple(slice_proj)] = data.array[tuple(slice_proj)]*self.norm_value/f
        else:
            out.array = data.array*self.norm_value/f 

        return out
