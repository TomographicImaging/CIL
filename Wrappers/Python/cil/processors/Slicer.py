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
from cil.framework import DataProcessor, AcquisitionData, ImageData, DataContainer, AcquisitionGeometry, ImageGeometry
import numpy
import warnings


class Slicer(DataProcessor):

    r'''Processor slices an array and returns new geometry.
        
    :param roi: region-of-interest to slice, specified as a dictionary containing tuple (Start, Stop, Step)
    :type roi: dict
    :param force: enforce slicing even if the returned geometry is not meaningful
    :type force: bool, default=False
    :return: returns a DataContainer (AcquisitionData or ImageData) object with an updated AcquisitionGeometry or ImageGeometry where applicable
    :rtype: DataContainer (AcquisitionData or ImageData)


    Start inclusive, Stop exclusive
    
    -1 is a shortcut to include all elements along the specified axis
    
    Input is used to contruct Python slice object using slice() function
    
    Start: Starting index where the slicing of object starts
    Stop: Ending index where the slicing of object stops.
    Step: Determines the increment between each index for slicing
    
    Both Start and Step are optional
    
    Supports negative indexing
                        
    '''

    def __init__(self,
                 roi = None,
                 force = False):

        kwargs = {'roi': roi,
                  'force': force}

        super(Slicer, self).__init__(**kwargs)
    

    def check_input(self, data):
        
        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        elif (data.geometry == None):
            raise ValueError('Geometry is not defined.')
        elif (self.roi == None):
            raise ValueError('Please, specify roi')
        else:
            return True 
    

    def process(self, out=None):

        data = self.get_input()
        ndim = data.number_of_dimensions

        geometry_0 = data.geometry
        geometry = geometry_0.copy()

        dimension_labels = list(geometry_0.dimension_labels)
        
        if self.roi != None:
            for key in self.roi.keys():
                if key not in data.dimension_labels:
                    raise ValueError('Wrong label is specified for roi, expected {}.'.format(data.dimension_labels))
        
        slice_object = self._construct_slice_object(self.roi, data.shape, dimension_labels)

        
        for key in self.roi.keys():
            idx = data.dimension_labels.index(key)
            n_elements = numpy.int32(numpy.ceil((slice_object[idx].stop - slice_object[idx].start) / numpy.abs(slice_object[idx].step)))
            
            if (isinstance(data, ImageData)):

                if key == 'channel':
                    if  n_elements > 1:
                        geometry.channels = n_elements
                    else:
                        geometry = geometry.get_slice(channel=slice_object[idx].start)
                elif key == 'vertical':
                    if n_elements > 1:
                        geometry.voxel_num_z = n_elements
                    else:
                        geometry = geometry.get_slice(vertical=slice_object[idx].start)
                elif key == 'horizontal_x':
                    if n_elements > 1:
                        geometry.voxel_num_x = n_elements
                    else:
                        geometry = geometry.get_slice(horizontal_x=slice_object[idx].start)
                elif key == 'horizontal_y':
                    if n_elements > 1:
                        geometry.voxel_num_y = n_elements
                    else:
                        geometry = geometry.get_slice(horizontal_y=slice_object[idx].start)
            
            # if AcquisitionData
            else:
                if key == 'channel':
                    if n_elements > 1:
                        geometry.set_channels(num_channels=n_elements)
                    else:
                        geometry = geometry.get_slice(channel=slice_object[idx].start)
                elif key == 'angle':
                    if n_elements > 1:
                        geometry.config.angles.angle_data = geometry_0.config.angles.angle_data[slice_object[idx]]
                    else:
                        geometry = geometry.get_slice(angle=slice_object[idx].start)
                elif key == 'vertical':
                    if n_elements > 1:
                        geometry.config.panel.num_pixels[1] = n_elements
                    else:
                        geometry = geometry.get_slice(vertical=slice_object[idx].start)
                elif key == 'horizontal':
                    if n_elements > 1:
                        geometry.config.panel.num_pixels[0] = n_elements
                    else:
                        geometry = geometry.get_slice(horizontal=slice_object[idx].start)
        
        if geometry is not None:
            data_sliced = geometry.allocate()
            data_sliced.fill(numpy.squeeze(data.as_array()[tuple(slice_object)]))
            if out == None:
                return data_sliced
            else:
                out = data_sliced
        else:
            if self.force == False:
                raise ValueError("Cannot calculate system geometry. Use 'force=True' to return DataContainer instead.")
            else:
                return DataContainer(numpy.squeeze(data.as_array()[tuple(slice_object)]), deep_copy=False, dimension_labels=dimension_labels, suppress_warning=True)

    def _construct_slice_object(self, roi, n_elements, dimension_labels):
        '''
        parse roi input
        here we construct slice() object to slice the actual array
        '''
        ndim = len(n_elements)
        slice_object = []
        # loop through dimensions
        for i in range(ndim):
            # given dimension number, get corresponding label
            label = dimension_labels[i]
            # '-1' shortcut = include all elements
            if (label in roi.keys()) and (roi[label] != -1):
                # start and step are optional
                if len(roi[label]) == 1:
                    start = 0
                    step = 1
                    if roi[label][0] != None:
                        if roi[label][0] < 0:
                            stop =  n_elements[i]+roi[label][0]
                        else:
                            stop = roi[label][0]
                    else:
                        stop =  n_elements[i]

                elif len(roi[label]) == 2 or len(roi[label]) == 3:
                    
                    if roi[label][0] != None:
                        if roi[label][0] < 0:
                            start = n_elements[i]+roi[label][0]
                        else:
                            start = roi[label][0]
                    else:
                        start = 0

                    if roi[label][1] != None:
                        if roi[label][1] < 0:
                            stop = n_elements[i]+roi[label][1]
                        else:
                            stop = roi[label][1]
                    else: 
                        stop = n_elements[i]
                    
                    if len(roi[label]) == 2:
                        step = 1

                    if len(roi[label]) == 3:
                        if roi[label][2] != None:
                            step = roi[label][2]
                        else:
                            step = 1
                else:
                    raise ValueError('roi is exected to have 1, 2 or 3 elements')
            else:
                step = 1
                start = 0
                stop = n_elements[i]
            slice_object.append(slice(start, stop, step))
        return slice_object

