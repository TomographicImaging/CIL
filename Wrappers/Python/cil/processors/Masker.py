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

from cil.framework import DataProcessor, AcquisitionData, ImageData, ImageGeometry, DataContainer
import warnings
import numpy

class Masker(DataProcessor):
    r'''
    Processor to fill missing values provided by mask. Please use the desiried method to configure a processor for your needs.
    '''

    @staticmethod
    def value(mask=None, value=0):
        r'''This imputes value where mask == 0.
        :param mask: mask 
        :type mask: DataContainer
        :param value: values to be assigned to missing elements
        :type value: float, default=0
        '''

        processor = Masker(mode='value', mask=mask, value=value)

        return processor
    
    @staticmethod
    def mean(mask=None, axis=None):
        r'''This imputes mean where mask == 0. If no axis is specified then operates over flattened array.
        :param mask: mask 
        :type mask: DataContainer
        :param axis: specify axis as int or from 'dimension_labels' to calculate mean. 
        :type axis: str, int
        '''

        processor = Masker(mode='mean', mask=mask, axis=axis)

        return processor
    
    @staticmethod
    def median(mask=None, axis=None):
        r'''This imputes median where mask == 0. If no axis is specified then operates over flattened array.
        :param mask: mask 
        :type mask: DataContainer
        :param axis: specify axis as int or from 'dimension_labels' to calculate median. 
        :type axis: str, int
        '''

        processor = Masker(mode='median', mask=mask, axis=axis)

        return processor

    def __init__(self,
                 mask = None,
                 mode = 'value',
                 value = 0,
                 axis = None ):
        
        r'''Processor to fill missing values provided by mask.
        :param mask: DataContainer containing a boolean array with zeros where outliers are detected
        :type mask: DataContainer
        :param mode: a method to fill in missing values (value, mean, median)
        :type mode: str, default=value
        :param value: substitute all outliers with a specific value
        :type value: float, default=0
        :param axis: specify axis as int or from 'dimension_labels' to calculate mean or median in respective modes 
        :type axis: str or int
        :return: DataContainer or it's subclass with masked outliers
        :rtype: DataContainer or it's subclass   
        '''

        kwargs = {'mask': mask,
                  'mode': mode,
                  'value': value,
                  'axis': axis}

        super(Masker, self).__init__(**kwargs)
    
    def check_input(self, data):

        if self.mask is None:
            raise ValueError('Please, provide a mask.')

        if not (isinstance(self.mask, DataContainer)):
            raise TypeError('Mask must be a DataContainer')
        
        if not (issubclass(type(data), DataContainer)):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n' +
                            ' - DataContainer')

        if not (data.shape == self.mask.shape and data.dimension_labels == self.mask.dimension_labels):
            raise Exception("Mask and Data has to have the same shape and dimension labels." + 
                            "{} != {} or {} != {}".format(data.shape, self.mask.shape, data.dimension_labels, self.mask.dimension_labels))

        if self.mode not in ['value', 'mean', 'median']:
            raise Exception("Wrong mode. One of the following is expected:\n" + \
                            "value, mean, median, interpolation")
    
        return True 

    def process(self, out=None):
        
        data = self.get_input()
        arr = data.copy().as_array()
        mask_arr = self.mask.as_array()

        try:
            axis_index = data.dimension_labels.index(self.axis)             
        except:
            if type(self.axis) == int:
                axis_index = self.axis
            else:
                axis_index = None
        
        if self.mode == 'value':
            
            arr[~mask_arr] = self.value
        
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
                        slice_data[~mask_arr[current_slice_obj]] = numpy.mean(slice_data[mask_arr[current_slice_obj]])
                    else:
                        slice_data[~mask_arr[current_slice_obj]] = numpy.median(slice_data[mask_arr[current_slice_obj]])
                    arr[current_slice_obj] = slice_data
                
            else:

                if self.mode == 'mean':
                    arr[~mask_arr] = numpy.mean(arr[mask_arr]) 
                else:
                    arr[~mask_arr] = numpy.median(arr[mask_arr]) 
        
        else:
            raise ValueError('Mode is not recognised. One of the following is expected: ' + \
                              'value, mean, median')
        
        if out is None:
            out = data.copy()
            out.fill(arr)
            return out
        else:
            out.fill(arr)
