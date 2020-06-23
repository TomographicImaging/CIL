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

from ccpi.framework import DataProcessor, AcquisitionData, ImageData
import warnings
import numpy, scipy


class MaskGenerator(DataProcessor):

    def __init__(self,
                 method = 'nan',
                 threshold_value = (None, None),
                 quantiles = (None, None),
                 threshold_factor = 3,
                 window = None,
                 axis = None):
        
        '''
        Constructor
        
        Input:
            
            method              
                - special_values    test element-wise for both inf and nan
                - nan               test element-wise for nan
                - inf               test element-wise for nan
                - threshold         test element-wise if array values are within boundaries
                                    given by threshold_values = (float,float). 
                                    You can secify only lower threshold value by setting another to None
                                    such as threshold_values = (float,None), then
                                    upper boundary will be amax(data). Similarly, to specify only upper 
                                    boundary, use threshold_values = (None,float). If both threshold_values
                                    are set to None, then original array will be returned.
                - quantile          test element-wise if array values are within boundaries
                                    given by quantiles = (q1,q2), q1,q2<=1. 
                                    You can secify only lower quantile value by setting another to None
                                    such as quantiles = (float,q2), then
                                    upper boundary will be amax(data). Similarly, to specify only upper 
                                    boundary, use quantiles = (None,q1). If both quantiles
                                    are set to None, then original array will be returned.
                - mean              test element-wise if 
                                    abs(A - mean(A)) < threshold_factor * std(A).
                                    Default value of threshold_factor is 3. If no axis is specified, 
                                    then operates over flattened array. Alternatively operates along axis specified 
                                    as dimension_label.
                - median            test element-wise if 
                                    abs(A - median(A)) < threshold_factor * scaled MAD(A),
                                    scaled MAD is defined as c*median(abs(A-median(A))) where c=-1/(sqrt(2)*erfcinv(3/2))
                                    Default value of threshold_factor is 3. If no axis is specified, 
                                    then operates over flattened array. Alternatively operates along axis specified 
                                    as dimension_label.
                - movmean           the same as mean but uses rolling mean with a specified window,
                                    default window value is 5
                - movmedian         the same as mean but uses rolling median with a specified window,
                                    default window value is 5
        
        Output:
                numpy boolean array with 1 where condition was satisfied and 0 where not
                
        '''

        kwargs = {'method': method,
                  'threshold_value': threshold_value,
                  'threshold_factor': threshold_factor,
                  'quantiles': quantiles,
                  'window': window,
                  'axis': axis}

        super(MaskGenerator, self).__init__(**kwargs)
    
    def check_input(self, data):
        
        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise Exception('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        elif (data.geometry == None):
            raise Exception('Geometry is not defined.')
        else:
            return True 

    def process(self):

        data = self.get_input()
        
        mask = numpy.zeros_like(data.as_array())
        
        if self.method == 'special_values':
            
            mask[numpy.logical_or(numpy.isnan(data.as_array()), numpy.isinf(data.as_array()))] = 1
        
        elif self.method == 'nan':
            
            mask[numpy.isnan(data.as_array())] = 1
            
        elif self.method == 'inf':
            
            mask[numpy.isinf(data.as_array())] = 1
            
        elif self.method == 'threshold':
            
            if not(isinstance(self.threshold_value, tuple)):
                raise Exception("Threshold value must be given as a tuple containing two values,\n use None if no threshold value is given")
                
            threshold_value = []
            if self.threshold_value[0] is None:
                threshold_value.append(numpy.amin(data.as_array()))
            else:
                threshold_value.append(self.threshold_value[0])
                tmp_min = numpy.amin(data.as_array())
                if self.threshold_value[0] < tmp_min:
                    warnings.warn("Given threshold_value {} is smaller than min value of data {}".format(self.threshold_value[0], tmp_min))
            
            if self.threshold_value[1] is None:
                threshold_value.append(numpy.amax(data.as_array()))
            else:
                threshold_value.append(self.threshold_value[1])
                tmp_max = numpy.amax(data.as_array())
                if self.threshold_value[1] > tmp_max:
                    warnings.warn("Given threshold_value {} is larger than max value of data {}".format(self.threshold_value[1], tmp_max))
            
            if threshold_value[1] < threshold_value[0]:
                raise Exception("Upper threshold value must be larger than lower treshold value or min of data")
            
            mask[numpy.logical_or(data.as_array() < threshold_value[0], data.as_array() > threshold_value[1])] = 1
            
        elif self.method == 'quantile':
            
            if not(isinstance(self.quantiles, tuple)):
                raise Exception("Quantiles must be given as a tuple containing two values,\n use None if no quantile value is given")
            
            quantile_values = []
            if self.quantiles[0] is None:
                quantile_values.append(numpy.amin(data.as_array()))
            else:
                if self.quantiles[0] < 0 or self.quantiles[0] > 1:
                    raise Exception("quantile_values must be within 0 and 1")
                quantile_values.append(numpy.quantile(data.as_array(), self.quantiles[0]))
            if self.quantiles[1] is None:
                quantile_values.append(numpy.amax(data.as_array()))
            else:
                quantile_values.append(numpy.quantile(data.as_array(), self.quantiles[1]))
                if self.quantiles[1] < 0 or self.quantiles[1] > 1:
                    raise Exception("quantile_values must be within 0 and 1")
            
            if quantile_values[1] <  quantile_values[0]:
                raise Exception("Upper quantile must be larger than lower quantile.")
                
            mask[numpy.logical_or(data.as_array() < quantile_values[0], data.as_array() > quantile_values[1])] = 1
        
        elif self.method == 'mean':
            
            if self.axis is not None:
                
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                
                axis = data.get_dimension_axis(self.axis)
                ndim = len(data.dimension_labels)
                
                tile_par = []
                slice_obj = []
                for i in range(ndim):
                    if i == axis:
                        tile_par.append(data.get_dimension_size(self.axis))
                        slice_obj.append(numpy.newaxis)
                    else:
                        tile_par.append(1)
                        slice_obj.append(slice(None, None, None))
                tile_par = tuple(tile_par)
                slice_obj = tuple(slice_obj)
                
                mask[numpy.abs(data.as_array() - numpy.tile((numpy.mean(data.as_array(), axis=axis))[slice_obj], tile_par)) > 
                     self.threshold_factor * numpy.tile((numpy.std(data.as_array(), axis=axis))[slice_obj], tile_par)] = 1
                     
            else:
                
                 mask[numpy.abs(data.as_array() - numpy.mean(data.as_array())) > 
                 self.threshold_factor * numpy.std(data.as_array())] = 1
        
        elif self.method == 'median':
            
            c = -1 / (numpy.sqrt(2) * scipy.special.erfcinv(3 / 2))
            
            if self.axis is not None:
                
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                    
                axis = data.get_dimension_axis(self.axis)
                ndim = len(data.dimension_labels)
                
                tile_par = []
                slice_obj = []
                for i in range(ndim):
                    if i == axis:
                        tile_par.append(data.get_dimension_size(self.axis))
                        slice_obj.append(numpy.newaxis)
                    else:
                        tile_par.append(1)
                        slice_obj.append(slice(None, None, None))
                tile_par = tuple(tile_par)
                slice_obj = tuple(slice_obj)
            
                tmp = abs(data.as_array() - numpy.tile((numpy.median(data.as_array(), axis=axis))[slice_obj], tile_par))
                mask[tmp > self.threshold_factor * c * numpy.tile((numpy.median(tmp, axis=axis))[slice_obj], tile_par)] = 1
            
            else:
                
                tmp = abs(data.as_array() - numpy.median(data.as_array()))
                mask[tmp > self.threshold_factor * c * numpy.median(tmp)] = 1
            
        elif self.method == 'movmean':
            
            if self.window is None:
                window = 5
            else:
                window = self.window
            
            if self.axis is not None:
                
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                
                axis = data.get_dimension_axis(self.axis)
                
                # we compute rolling mean and rolling std through convolution                
                kernel = numpy.ones(window, dtype=numpy.float32)
                mean_array = scipy.ndimage.convolve1d(data.as_array(), weights=kernel, axis=axis, mode='reflect')
                q = mean_array ** 2
                q = scipy.ndimage.convolve1d(q, weights=kernel, axis=axis, mode='reflect')
                std_array = numpy.sqrt((q - (mean_array ** 2) / window) / (window-1))
                mask[numpy.abs(data.as_array()-mean_array) > self.threshold_factor * std_array] = 1
            
            else:
                
                ndim = len(data.dimension_labels)
                
                # we compute rolling mean and rolling std through convolution 
                kernel = numpy.ones((window,)*ndim, dtype=numpy.float32)
                mean_array = scipy.ndimage.convolve(data.as_array(), weights=kernel, mode='reflect')
                q = mean_array ** 2
                q = scipy.ndimage.convolve(q, weights=kernel, mode='reflect')
                std_array = numpy.sqrt((q - (mean_array ** 2) / window) / (window-1))
                
                mask[numpy.abs(data.as_array()-mean_array) > self.threshold_factor * std_array] = 1
                
            
        elif self.method == 'movmedian':
            
            if self.window is None:
                window = 5
            else:
                window = self.window
                    
            if self.axis is not None:
                
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                
                axis = data.get_dimension_axis(self.axis)
                ndim = len(data.dimension_labels)
                
                kernel_shape = []
                for i in range(ndim):
                    if i == axis:
                        kernel_shape.append(window)
                    else:
                        kernel_shape.append(1)
                
                kernel = numpy.ones(tuple(kernel_shape), dtype=numpy.float32)
                
                median_array = scipy.ndimage.median_filter(data.as_array(), footprint=kernel, mode='reflect')
                
                c = -1 / (numpy.sqrt(2) * scipy.special.erfcinv(3 / 2))
                tmp = abs(data.as_array() - median_array)
                mask[tmp > self.threshold_factor * c * scipy.ndimage.median_filter(tmp, footprint=kernel, mode='reflect')] = 1
                
            else:
                
                median_array = scipy.ndimage.median_filter(data.as_array(), size=window, mode='reflect')
                
                c = -1 / (numpy.sqrt(2) * scipy.special.erfcinv(3 / 2))
                tmp = abs(data.as_array() - median_array)
                mask[tmp > self.threshold_factor * c * scipy.ndimage.median_filter(tmp, size=window, mode='reflect')] = 1
                
        return mask