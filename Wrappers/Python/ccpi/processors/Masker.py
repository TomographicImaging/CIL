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
import numpy
from scipy import interpolate


class Masker(DataProcessor):

    def __init__(self,
                 mask = None,
                 mode = 'value',
                 value = 0,
                 axis = None,
                 interp_kind = 'linear'
                 ):
        
        '''
        Constructor
        
        Input:
            
            mode              
                - value
                - mean
                - median
                - interpolation
        Output:
                numpy boolean array with 1 where condition was satisfied and 0 where not
                
        '''

        kwargs = {'mask': mask,
                  'mode': mode,
                  'value': value,
                  'axis': axis,
                  'interp_kind': interp_kind}

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
        
        if self.mode not in ['value', 'mean', 'median', 'interpolate']:
            raise Exception("Wrong mode. One of the following is expected:\n value, mean, median, interpolate")
        

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
        
        elif self.mode == 'interpolate':
            
            if self.interp_kind not in ['linear', 'nearest', 'zeros', 'linear', \
                                        'quadratic', 'cubic', 'previous', 'next']:
                raise Exception("Wrong interpolation kind, one of the follwoing is expected:\n linear, nearest, zeros, linear, quadratic, cubic, previous, next")
            
            ndim = len(data.dimension_labels)
            shape = data.shape
            
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
                    
            interp_axis = numpy.arange(shape[axis])
            
            for i in range(res_dim):
                
                rest_shape = []
                for j in range(ndim):
                    if j != axis:
                        rest_shape.append(shape[j])
                rest_shape = tuple(rest_shape)
                
                rest_idx = numpy.unravel_index(i, rest_shape)
                
                k = 0
                idx = []
                for j in range(ndim):
                    if j == axis:
                        idx.append(slice(None,None,1))
                    else:
                        idx.append(rest_idx[k])
                        k += 1
                idx = tuple(idx)
                
                if numpy.any(self.mask[idx]):
                    tmp = data.as_array()[idx]
                    f = interpolate.interp1d(interp_axis[~self.mask[idx]], tmp[~self.mask[idx]], 
                                             fill_value='extrapolate',
                                             assume_sorted=True,
                                             kind=self.interp_kind)
                    tmp[self.mask[idx]] = f(numpy.where(self.mask[idx] == True)[0])
                    data.as_array()[idx] = tmp
        
        return data
        