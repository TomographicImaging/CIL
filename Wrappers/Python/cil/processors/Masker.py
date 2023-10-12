# -*- coding: utf-8 -*-
#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

from cil.framework import DataProcessor, AcquisitionData, ImageData, ImageGeometry, DataContainer
import warnings
import numpy
from scipy import interpolate

class Masker(DataProcessor):
    r'''
    Processor to fill missing values provided by mask. Please use the desiried method to configure a processor for your needs.
    '''

    @staticmethod
    def value(mask=None, value=0):
        r'''This sets the masked values of the input data to the requested value.

        :param mask: A boolean array with the same dimensions as input, where 'False' represents masked values. Mask can be generated using 'MaskGenerator' processor to identify outliers. 
        :type mask: DataContainer, ImageData, AcquisitionData, numpy.ndarray
        :param value: values to be assigned to missing elements
        :type value: float, default=0
        '''

        processor = Masker(mode='value', mask=mask, value=value)

        return processor
    
    @staticmethod
    def mean(mask=None, axis=None):
        r'''This sets the masked values of the input data to the mean of the unmasked values across the array or axis.

        :param mask: A boolean array with the same dimensions as input, where 'False' represents masked values. Mask can be generated using 'MaskGenerator' processor to identify outliers.  
        :type mask: DataContainer, ImageData, AcquisitionData, numpy.ndarray
        :param axis: specify axis as int or from 'dimension_labels' to calculate mean. 
        :type axis: str, int
        '''

        processor = Masker(mode='mean', mask=mask, axis=axis)

        return processor
    
    @staticmethod
    def median(mask=None, axis=None):
        r'''This sets the masked values of the input data to the median of the unmasked values across the array or axis.

        :param mask: A boolean array with the same dimensions as input, where 'False' represents masked values. Mask can be generated using 'MaskGenerator' processor to identify outliers.  
        :type mask: DataContainer, ImageData, AcquisitionData, numpy.ndarray
        :param axis: specify axis as int or from 'dimension_labels' to calculate median. 
        :type axis: str, int
        '''

        processor = Masker(mode='median', mask=mask, axis=axis)

        return processor
    
    @staticmethod
    def interpolate(mask=None, axis=None, method='linear'):
        r'''This operates over the specified axis and uses 1D interpolation over remaining flattened array to fill in missing vaues.

        :param mask: A boolean array with the same dimensions as input, where 'False' represents masked values. Mask can be generated using 'MaskGenerator' processor to identify outliers.  
        :type mask: DataContainer, ImageData, AcquisitionData, numpy.ndarray
        :param axis: specify axis as int or from 'dimension_labels' to loop over and perform 1D interpolation. 
        :type axis: str, int
        :param method: One of the following interpoaltion methods: linear, nearest, zeros, linear, quadratic, cubic, previous, next
        :param method: str, default='linear'
        '''

        processor = Masker(mode='interpolate', mask=mask, axis=axis, method=method)

        return processor

    def __init__(self,
                 mask = None,
                 mode = 'value',
                 value = 0,
                 axis = None,
                 method = 'linear'):
        
        r'''Processor to fill missing values provided by mask.

        :param mask: A boolean array with the same dimensions as input, where 'False' represents masked values. Mask can be generated using 'MaskGenerator' processor to identify outliers. 
        :type mask: DataContainer, ImageData, AcquisitionData, numpy.ndarray
        :param mode: a method to fill in missing values (value, mean, median, interpolate)
        :type mode: str, default=value
        :param value: substitute all outliers with a specific value
        :type value: float, default=0
        :param axis: specify axis as int or from 'dimension_labels' to calculate mean or median in respective modes 
        :type axis: str or int
        :param method: One of the following interpoaltion methods: linear, nearest, zeros, linear, quadratic, cubic, previous, next
        :param method: str, default='linear'
        :return: DataContainer or it's subclass with masked outliers
        :rtype: DataContainer or it's subclass   
        '''

        kwargs = {'mask': mask,
                  'mode': mode,
                  'value': value,
                  'axis': axis,
                  'method': method}

        super(Masker, self).__init__(**kwargs)
    
    def check_input(self, data):

        if self.mask is None:
            raise ValueError('Please, provide a mask.')

        if not (data.shape == self.mask.shape):
            raise Exception("Mask and Data must have the same shape." + 
                            "{} != {}".format(self.mask.mask, data.shape))
        
        if hasattr(self.mask, 'dimension_labels') and data.dimension_labels != self.mask.dimension_labels:
            raise Exception("Mask and Data must have the same dimension labels." + 
                            "{} != {}".format(self.mask.dimension_labels, data.dimension_labels))

        if self.mode not in ['value', 'mean', 'median', 'interpolate']:
            raise Exception("Wrong mode. One of the following is expected:\n" + 
                            "value, mean, median, interpolate")
    
        return True 

    def process(self, out=None):
        
        data = self.get_input()
        
        return_arr = False
        if out is None:
            out = data.copy()
            arr = out.as_array()
            return_arr = True
        else:
            arr = out.as_array()
        
        #assumes mask has 'as_array' method, i.e. is a DataContainer or is a numpy array
        try:
            mask_arr = self.mask.as_array()
        except:
            mask_arr = self.mask

        if mask_arr.dtype != bool:
            raise TypeError("Mask expected to be a boolean array got {}".format(mask_arr.dtype))
        
        mask_invert = ~mask_arr
            
        try:
            axis_index = data.dimension_labels.index(self.axis)             
        except:
            if type(self.axis) == int:
                axis_index = self.axis
            else:
                axis_index = None
        
        if self.mode == 'value':
            
            arr[mask_invert] = self.value
        
        elif self.mode == 'mean' or self.mode == 'median':
            
            if axis_index is not None:
                
                ndim = data.number_of_dimensions
                    
                slice_obj = [slice(None, None, 1)] * ndim
                            
                for i in range(arr.shape[axis_index]):
                    current_slice_obj = slice_obj[:]
                    current_slice_obj[axis_index] = i
                    current_slice_obj = tuple(current_slice_obj)
                    slice_data = arr[current_slice_obj]
                    if self.mode == 'mean':
                        slice_data[mask_invert[current_slice_obj]] = numpy.mean(slice_data[mask_arr[current_slice_obj]])
                    else:
                        slice_data[mask_invert[current_slice_obj]] = numpy.median(slice_data[mask_arr[current_slice_obj]])
                    arr[current_slice_obj] = slice_data
                
            else:

                if self.mode == 'mean':
                    arr[mask_invert] = numpy.mean(arr[mask_arr]) 
                else:
                    arr[mask_invert] = numpy.median(arr[mask_arr]) 
        
        elif self.mode == 'interpolate':
            if self.method not in ['linear', 'nearest', 'zeros', 'linear', \
                                        'quadratic', 'cubic', 'previous', 'next']:
                raise TypeError("Wrong interpolation method, one of the follwoing is expected:\n" + 
                                "linear, nearest, zeros, linear, quadratic, cubic, previous, next")
            
            ndim = data.number_of_dimensions
            shape = arr.shape
            
            if axis_index is None:
                raise NotImplementedError ('Currently Only 1D interpolation is available. Please specify an axis to interpolate over.')
            
            res_dim = 1
            for i in range(ndim):
                if i != axis_index:
                    res_dim *= shape[i]
            
            # get axis for 1D interpolation
            interp_axis = numpy.arange(shape[axis_index])
            
            # loop over slice
            for i in range(res_dim):
                
                rest_shape = []
                for j in range(ndim):
                    if j != axis_index:
                        rest_shape.append(shape[j])
                rest_shape = tuple(rest_shape)
                
                rest_idx = numpy.unravel_index(i, rest_shape)
                
                k = 0
                idx = []
                for j in range(ndim):
                    if j == axis_index:
                        idx.append(slice(None,None,1))
                    else:
                        idx.append(rest_idx[k])
                        k += 1
                idx = tuple(idx)
                
                if numpy.any(mask_invert[idx]):
                    tmp = arr[idx]
                    f = interpolate.interp1d(interp_axis[mask_arr[idx]], tmp[mask_arr[idx]], 
                                            fill_value='extrapolate',
                                            assume_sorted=True,
                                            kind=self.method)
                    tmp[mask_invert[idx]] = f(numpy.where(mask_arr[idx] == False)[0])
                    arr[idx] = tmp
        
        else:
            raise ValueError('Mode is not recognised. One of the following is expected: ' + 
                              'value, mean, median, interpolate')
        
        out.fill(arr)

        if return_arr is True:
            return out
