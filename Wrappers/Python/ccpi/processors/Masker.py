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


class Masker(DataProcessor):

    def __init__(self,
                 mask = None,
                 mode = 'value',
                 value = 0,
                 axis = None
                 ):
        
        '''
        Constructor
        
        Input:
            
            mode              
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

        kwargs = {'mask': mask,
                  'mode': mode,
                  'value': value,
                  'axis': axis}

        super(Masker, self).__init__(**kwargs)
    
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

        data_raw = self.get_input()
        data = data_raw.clone()
        
        if self.mode == 'value':
            
            data.as_array()[self.mask] = self.value
        
        elif self.mode == 'mean':
            
            if self.axis is not None:
                
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                
                axis = data.get_dimension_axis(self.axis)
                ndim = len(data.dimension_labels)
                    
                slice_obj = []
                for i in range(ndim):
                        slice_obj.append(slice(None, None, 1))
                            
                for i in range(data_raw.get_dimension_size(self.axis)):
                    slice_tmp = slice_obj[:]
                    slice_tmp[axis] = i
                    slice_tmp = tuple(slice_tmp)
                    tmp = data.as_array()[slice_tmp]
                    tmp[self.mask[slice_tmp]] = numpy.mean(tmp[~self.mask[slice_tmp]])
                    data.as_array()[slice_tmp] = tmp
                
            else:
                
                data.as_array()[self.mask] = numpy.mean(data.as_array()[~self.mask]) 
        
        elif self.mode == 'median':
            
            if self.axis is not None:
                
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                
                axis = data.get_dimension_axis(self.axis)
                ndim = len(data.dimension_labels)
                    
                slice_obj = []
                for i in range(ndim):
                        slice_obj.append(slice(None, None, 1))
                            
                for i in range(data_raw.get_dimension_size(self.axis)):
                    slice_tmp = slice_obj[:]
                    slice_tmp[axis] = i
                    slice_tmp = tuple(slice_tmp)
                    tmp = data.as_array()[slice_tmp]
                    tmp[self.mask[slice_tmp]] = numpy.median(tmp[~self.mask[slice_tmp]])
                    data.as_array()[slice_tmp] = tmp
                
            else:
                
                data.as_array()[self.mask] = numpy.median(data.as_array()[~self.mask])
        
        elif self.mode == 'lookup':
            
            ndim = len(data.dimension_labels)
            shape = numpy.array(data.shape)
            
            if self.axis is not None:
                if self.axis not in data.dimension_labels.values():
                    raise Exception("Wrong label is specified for axis")
                axis = data.get_dimension_axis(self.axis)
            else:
                axis = 0
            
            res_dim = 1
            for i in range(ndim):
                if i != axis:
                    res_dim *= shape[i]
            
            for i in range(res_dim):
                idx = []
                rest_shape = numpy.unravel_index(i, tuple(shape[1:]))
                k = 0
                for j in range(ndim):
                    if j == axis:
                        idx.append(slice(None,None,1))
                    else:
                        idx.append(rest_shape[k])
                        k += 1
                idx = tuple(idx)
                
                if numpy.any(self.mask[idx]):
                    tmp = data.as_array()[idx]
                    tmp[self.mask[idx]] = numpy.interp(numpy.where(self.mask[idx] == True)[0], 
                                       numpy.arange(shape[0])[~self.mask[idx]],
                                       tmp[~self.mask[idx]])
                    data.as_array()[idx] = tmp
                
                
        
        
        return data
        