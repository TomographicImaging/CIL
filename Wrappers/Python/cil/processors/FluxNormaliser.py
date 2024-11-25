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
import numpy
import logging
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

class FluxNormaliser(Processor):
    r'''
    Flux normalisation based on float or region of interest

    This processor reads in an AcquisitionData and normalises it by flux from
    a float or array of float values, or the mean flux in a region of interest.

    Parameters:
    -----------
    flux: float or list of floats, optional
        The value to divide the image by. If flux is a list it must have length 
        equal to the number of projections in the dataset. If flux=None, calculate
        flux from the roi.
    
    roi: dict, optional
        Dictionary describing the region of interest containing the background
        in the image. The roi is specified as `{'horizontal':(start,stop), 
        'vertical':(start,stop)}`. If an axis is not specified in the roi dictionary, 
        the full range will be used.

    target: {'mean', 'first'} or float, default='mean'
        The value to scale the normalised data with. If float, the data is scaled 
        to the float value. If string 'mean' the data is scaled to the mean value 
        of the input flux or flux across all rois. If 'first' the data is scaled to 
        the first input flux value or the flux in the roi of the first projection.
        Default is 'mean'

    Returns:
    --------
    Output: AcquisitionData normalised by flux

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
    >>> processor = FluxNormaliser(roi={'horizontal':(5, 15)})
    >>> processor.set_input(data)
    >>> data_norm = processor.get_output()

    Note
    ----
    The roi indices provided are start inclusive, stop exclusive.
    All elements along a dimension will be included if the axis does not appear 
    in the roi dictionary
    '''

    def __init__(self, flux=None, roi=None, target='mean'):
            
            kwargs = {
                    'flux'  : flux,
                    'roi' : roi,
                    'roi_slice' : None,
                    'roi_axes' : None,
                    'target' : target,
                    'target_value' : None,
                    'v_size' : 1,
                    'v_axis' : None,
                    'h_size' : 1,
                    'h_axis' : None
                    }
            super(FluxNormaliser, self).__init__(**kwargs)
            
    def check_input(self, dataset):

        if self.roi is not None and self.flux is not None:
            raise ValueError("Please specify either flux or roi, not both")
        if self.roi is None and self.flux is None:
            raise ValueError("Please specify either flux or roi, found None")
        
        if not (type(dataset), AcquisitionData):
            raise TypeError("Expected AcquistionData, found {}"
                            .format(type(dataset)))
        
        image_axes = 0
        if 'vertical' in dataset.dimension_labels:
            v_axis = dataset.get_dimension_axis('vertical')
            self.v_size = dataset.get_dimension_size('vertical')
            image_axes += 1

        if 'horizontal' in dataset.dimension_labels:
            self.h_axis = dataset.get_dimension_axis('horizontal')
            self.h_size = dataset.get_dimension_size('horizontal')
            image_axes += 1

        if (( self.h_axis is not None)  and (self.h_axis < (len(dataset.shape)-image_axes))) or \
            ((self.v_axis is not None) and self.v_axis < (len(dataset.shape)-image_axes)):
            raise ValueError('Projections must be the last two axes of the dataset')

        return True

    def _calculate_flux(self):
        '''
        Function to calculate flux from a region of interest in the data. If the 
        flux is already provided as an array, convert the array to float 32 and
        check the size matches the number of projections 
        '''

        dataset = self.get_input()
        if dataset is None:
            raise ValueError('Data not found, please run `set_input(data)`')
        
        # Calculate the flux from the roi in the data
        if self.flux is None:

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
                    
                for d in ['horizontal', 'vertical']:
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
                    # if a projection dimension isn't in the roi, use the whole axis
                    else:
                        if d in dataset.dimension_labels:
                            ax = dataset.get_dimension_axis(d)
                            axes.append(ax)
                            self.roi.update({d:(0,dataset.get_dimension_size(d))})

                self.flux = numpy.mean(dataset.array[tuple(slc)], axis=tuple(axes))
                
                # Warn if the flux is more than 10% of the dataset range
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
        
        

        # convert flux array to float32
        self.flux = numpy.array(self.flux, dtype=numpy.float32, ndmin=1)

        # check flux array is the right size
        flux_size_flat = len(self.flux.ravel())
        if flux_size_flat > 1:
            data_size_flat = len(dataset.geometry.angles)*dataset.geometry.channels
            if data_size_flat != flux_size_flat:
                raise ValueError("Flux must be a scalar or array with length \
                                    \n = number of projections, found {} and {}"
                                    .format(flux_size_flat, data_size_flat))
            
        # check if flux array contains 0s
        if 0 in self.flux:
            raise ValueError('Flux value can\'t be 0, provide a different flux\
                                or region of interest with non-zero values')
          
    def _calculate_target(self):
        '''
        Calculate the target value for the normalisation
        '''

        if self.flux is None:
            raise ValueError('Flux not found')
            
        if isinstance(self.target, (int,float)):
            self.target_value = self.target
        elif isinstance(self.target, str):
            if self.target == 'first':
                if len(numpy.shape(self.flux)) > 0 :
                    self.target_value = self.flux.flat[0]
                else:
                    self.target_value = self.flux
            elif self.target == 'mean':
                self.target_value = numpy.mean(self.flux.ravel())
            else:
                raise ValueError("Target string not recognised, found {}, expected 'first' or 'mean'"
                                 .format(self.target))
        else:
            raise TypeError("Target must be string or a number, found {}"
                            .format(type(self.target)))
            
    def preview_configuration(self, angle=None, channel=None, log=False):
        '''
        Preview the FluxNormalisation processor configuration for roi mode.
        Plots the region of interest on the image and the mean, maximum and 
        minimum intensity in the roi.
        
        Parameters:
        -----------
        angle: float, optional
            Angle to plot, default=None displays the data with the minimum
            and maximum pixel values in the roi, otherwise the angle to display 
            can be specified as a float and the closest angle will be displayed.
            For 2D data, the roi is plotted on the sinogram.

        channel: int, optional
            The channel to plot, default=None displays the central channel if
            the data has channels

        log: bool, default=False
            If True, plot the image with a log scale, default is False
        '''
        self._calculate_flux()
        if self.roi_slice is None:
            raise ValueError('Preview available with roi, run `processor= FluxNormaliser(roi=roi)` then `set_input(data)`')
        else:
            
            data = self.get_input()

            min = numpy.min(data.array[tuple(self.roi_slice)], axis=tuple(self.roi_axes))
            max = numpy.max(data.array[tuple(self.roi_slice)], axis=tuple(self.roi_axes))
            if 'channel' in data.dimension_labels:
                if channel is None:
                    channel = int(data.get_dimension_size('channel')/2)
                channel_axis = data.get_dimension_axis('channel')
                flux_array = self.flux.take(indices=channel, axis=channel_axis)
                min = min.take(indices=channel, axis=channel_axis)
                max = max.take(indices=channel, axis=channel_axis)
            else:
                if channel is not None:
                    raise ValueError("Channel not found")
                else:
                    flux_array = self.flux
        
            plt.figure(figsize=(8,8))
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
                plt.plot(data.geometry.angles, flux_array, '.r', label='Mean')
                plt.plot(data.geometry.angles, min,'.k', label='Minimum')
                plt.plot(data.geometry.angles, max,'.k', label='Maximum')
            else:
                plt.plot(data.geometry.angles, flux_array, 'r', label='Mean')
                plt.plot(data.geometry.angles, min,'--k', label='Minimum')
                plt.plot(data.geometry.angles, max,'--k', label='Maximum')

            plt.legend()
            plt.xlabel('angle')
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
            data_slice = data_slice.get_slice(channel=channel_index)

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

        ax1 = plt.subplot(ax)
        if log:
            im = ax1.imshow(numpy.log(data_slice.array), cmap='gray',aspect='equal', origin='lower', extent=extent)
            plt.gcf().colorbar(im, ax=ax1)
        else:
            im = ax1.imshow(data_slice.array, cmap='gray',aspect='equal', origin='lower', extent=extent)
            plt.gcf().colorbar(im, ax=ax1)

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

        ax1.plot([h_min, h_max],[v_min, v_min],'--r')
        ax1.plot([h_min, h_max],[v_max, v_max],'--r')

        ax1.plot([h_min, h_min],[v_min, v_max],'--r')
        ax1.plot([h_max, h_max],[v_min, v_max],'--r')
        
        title = 'ROI'
        if angle_index is not None:
            title += ' angle = ' + str(data.geometry.angles[angle_index])
        if channel_index is not None:
            title += ' channel = ' + str(channel_index)
        ax1.set_title(title)

        ax1.set_xlabel(h)
        ax1.set_ylabel(v)

    def process(self, out=None):
        self._calculate_flux()
        self._calculate_target()
        data = self.get_input()

        if out is None:
            out = data.copy()

        proj_size = self.v_size*self.h_size
        num_proj = int(data.array.size / proj_size)
        
        f = self.flux
        for i in range(num_proj):
            arr_proj = data.array.flat[i*proj_size:(i+1)*proj_size]
            if len(self.flux.flat) > 1:
                f = self.flux.flat[i] 
            arr_proj *= self.target_value/f
            out.array.flat[i*proj_size:(i+1)*proj_size] = arr_proj

        return out