#%%
# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cil.framework import DataProcessor, AcquisitionData, ImageData, DataContainer, AcquisitionGeometry, ImageGeometry
import numpy
import warnings


class Binner(DataProcessor):
    r'''Binner processor rebins (downsample) array and returns new geometry.
        
    :param roi: region-of-interest to bin, specified as a dictionary containing tuple (Start, Stop, Step)
    :type roi: dict
    :return: returns an AcquisitionData or ImageData object with an updated AcquisitionGeometry or ImageGeometry
    :rtype: AcquisitionData or ImageData
    '''
    
    '''
    Start inclusive, Stop exclusive
    
    -1 is a shortcut to include all elements along the specified dimension
    
    if only one number is provided, then it is interpreted as Stop
    
    if two numbers are provided, then they are interpreted as Start and Stop
    
    Start = None is equivalent to Start = 0
    Stop = None is equivalent to Stop = number of elements
    Step = None is equivalent to Step = 1
    
    You can specify negative Start and Stop.
    
    If Stop - Start is not multiple of Step, then 
    the resulted dimension will have (Stop - Start) // Step 
    elements, i.e. (Stop - Start) % Step elements will be ignored
    '''

    def __init__(self,
                 roi = None):

        kwargs = {'roi': roi}

        super(Binner, self).__init__(**kwargs)
    

    def check_input(self, data):
        
        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        elif (data.geometry == None):
            raise ValueError('Geometry is not defined.')
        elif (self.roi == None):
            raise ValueError('Prease, specify roi')
        else:
            return True 
    

    def process(self, out=None):

        data = self.get_input()
        ndim = len(data.dimension_labels)
        dimension_labels = data.dimension_labels
        
        geometry_0 = data.geometry
        geometry = geometry_0.copy()

        dimension_labels = list(geometry_0.dimension_labels)
        
        if self.roi != None:
            for key in self.roi.keys():
                if key not in data.dimension_labels.values():
                    raise ValueError('Wrong label is specified for roi, expected {}.'.format(data.dimension_labels.values()))
        
        roi_object = self._construct_roi_object(self.roi, data.shape, dimension_labels)

        
        for key in self.roi.keys():
            idx = data.get_dimension_axis(key)
            n_elements = (roi_object[idx][1] - roi_object[idx][0]) // roi_object[idx][2]
            
            if (isinstance(data, ImageData)):

                if key == 'channel':
                    geometry.channels = n_elements
                    geometry.channel_spacing *= roi_object[idx][2]
                    if n_elements <= 1:
                        dimension_labels.remove('channel')
                elif key == 'vertical':
                    geometry.voxel_num_z = n_elements
                    geometry.voxel_size_z *= roi_object[idx][2]
                    if n_elements <= 1:
                        dimension_labels.remove('vertical')
                elif key == 'horizontal_x':
                    geometry.voxel_num_x = n_elements
                    geometry.voxel_size_x *= roi_object[idx][2]
                    if n_elements <= 1:
                       dimension_labels.remove('horizontal_x')
                elif key == 'horizontal_y':
                    geometry.voxel_num_y = n_elements
                    geometry.voxel_size_y *= roi_object[idx][2]
                    if n_elements <= 1:
                        dimension_labels.remove('horizontal_y')
            
            # if AcquisitionData
            else:
                if key == 'channel':
                    geometry.set_channels(num_channels=n_elements)
                    if n_elements <= 1:
                        dimension_labels.remove('channel')
                elif key == 'angle':
                    shape = (n_elements, roi_object[idx][2])
                    geometry.config.angles.angle_data = geometry_0.config.angles.angle_data[roi_object[idx][0]:(roi_object[idx][0] + n_elements * roi_object[idx][2])].reshape(shape).mean(1)
                    if n_elements <= 1:
                        dimension_labels.remove('angle')
                elif key == 'vertical':
                    if n_elements > 1:
                        geometry.config.panel.num_pixels[1] = n_elements
                    else:
                        geometry = geometry.subset(vertical = (roi_object[idx][1] + roi_object[idx][0]) // 2)
                    geometry.config.panel.pixel_size[1] *= roi_object[idx][2]
                elif key == 'horizontal':
                    geometry.config.panel.num_pixels[0] = n_elements
                    geometry.config.panel.pixel_size[0] *= roi_object[idx][2]
                    if n_elements <= 1:
                        dimension_labels.remove('horizontal')
        
        geometry.dimension_labels = dimension_labels
        
        shape_object = []
        slice_object = []
        for i in range(ndim):
            n_pix = (roi_object[i][1] - roi_object[i][0]) // roi_object[i][2]
            shape_object.append(n_pix)
            shape_object.append(roi_object[i][2])
            slice_object.append(slice(roi_object[i][0], roi_object[i][0] + n_pix * roi_object[i][2]))
        
        shape_object = tuple(shape_object)
        slice_object = tuple(slice_object)

        data_resized = data.as_array()[slice_object].reshape(shape_object)

        mean_order = [-1, 1, 2, 3]

        for i in range(ndim):
            data_resized = data_resized.mean(mean_order[i])
            
        data_binned = geometry.allocate()
        data_binned.fill(numpy.squeeze(data_resized))
        if out == None:
            return data_binned
        else:
            out = data_binned
        

    def _construct_roi_object(self, roi, n_elements, dimension_labels):

        '''
        parse roi input
        here we parse input and calculate requested roi
        '''
        ndim = len(n_elements)
        roi_object = []
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
                            if roi[label][2] <= 0:
                                raise ValueError('Binning parameter has to be > 0')
                            else:
                                step = roi[label][2]
                        else:
                            step = 1
                else:
                    raise ValueError('roi is exected to have 1, 2 or 3 elements')
            else:
                step = 1
                start = 0
                stop = n_elements[i]
            roi_object.append((start, stop, step))
        return roi_object