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

from cil.framework import DataProcessor, AcquisitionData, ImageData, DataContainer, ImageGeometry
import warnings
import numpy
from scipy import special, ndimage

class MaskGenerator(object):
    """
    This class contains factory methods to create a MaskGenerator object using the desired algorithm.
    """
    @staticmethod
    def special_values(nan=True, inf=True):
        r'''This creates a MaskGenerator processor which generates a mask for inf and/or nan values.

        :param nan: mask NaN values
        :type nan: bool, default=True
        :param inf: mask INF values
        :type inf: bool, default=True

        '''
        if nan is True:
            if inf is True:
                proccessor = MaskGenerator_backend(mode='special_values')
            else:
                proccessor = MaskGenerator_backend(mode='nan')
        else:
            if inf is True:
                proccessor = MaskGenerator_backend(mode='inf')
            else:
                raise ValueError("Please specify at least one type of value to threshold on")

        return proccessor

    @staticmethod
    def threshold(min_val=None, max_val=None):
        r'''This creates a MaskGenerator processor which generates a mask for values outside boundaries

        :param min_val: lower boundary
        :type min_val: float, default=None
        :param max_val: upper boundary
        :type max_val: float, default=None
        '''
        proccessor = MaskGenerator_backend(mode='threshold', threshold_value=(min_val,max_val))
        return proccessor

    @staticmethod
    def quantile(min_quantile=None, max_quantile=None):
        r'''This creates a MaskGenerator processor which generates a mask for values outside boundaries

        :param min_quantile: lower quantile, 0-1
        :type min_quantile: float, default=None
        :param max_quantile: upper quantile, 0-1
        :type max_quantile: float, default=None
        '''
        proccessor = MaskGenerator_backend(mode='quantile', quantiles=(min_quantile,max_quantile))
        return proccessor

    @staticmethod
    def mean(axis=None, threshold_factor=3, window=None):
        r'''This creates a MaskGenerator processor which generates a mask for values outside a multiple of standard-devaiations from the mean.

        abs(A - mean(A)) < threshold_factor * std(A).
        
        :param threshold_factor: scale factor of standard-deviations to use as threshold
        :type threshold_factor: float, default=3
        :param axis: specify axis from 'dimension_labels' to calculate mean. If no axis is specified then operates over flattened array.
        :type axis: string
        :param window: specify number of pixels to use in calculation of a rolling mean
        :type window: int, default=None
        '''
        if window == None:
            proccessor = MaskGenerator_backend(mode='mean', threshold_factor=threshold_factor, axis=axis)
        else:
            proccessor = MaskGenerator_backend(mode='movmean', threshold_factor=threshold_factor, axis=axis, window=window)

        return proccessor

    @staticmethod
    def median(axis=None, threshold_factor=3, window=None):
        r'''This creates a MaskGenerator processor which generates a mask for values outside a multiple of median absolute deviation (MAD) from the mean.

        abs(A - median(A)) < threshold_factor * MAD(A),
        MAD = c*median(abs(A-median(A))) where c=-1/(sqrt(2)*erfcinv(3/2))

        :param threshold_factor: scale factor of MAD to use as threshold
        :type threshold_factor: float, default=3        
        :param axis: specify axis from 'dimension_labels' to calculate median. If no axis is specified then operates over flattened array.
        :type axis: string
        :param window: specify number of pixels to use in calculation of a rolling median
        :type window: int, default=None
        '''

        if window == None:
            proccessor = MaskGenerator_backend(mode='median', threshold_factor=threshold_factor, axis=axis)
        else:
            proccessor = MaskGenerator_backend(mode='movmedian', threshold_factor=threshold_factor, axis=axis, window=window)

        return proccessor

class MaskGenerator_backend(DataProcessor):

    r'''Processor to detect outliers and return mask with 0 where outliers were detected.
        
    :param mode: a method for detecting outliers (special_values, nan, inf, threshold, quantile, mean, median, movmean, movmedian)
    :type mode: string, default=special_values
    :param threshold_value: specify lower and upper boundaries if 'threshold' mode is selected
    :type threshold_value: tuple
    :param quantiles: specify lower and upper quantiles if 'quantile' mode is selected
    :type quantiles: tuple
    :param threshold_factor: scales detction threshold (standard deviation in case of 'mean', 'movmean' and median absolute deviation in case of 'median', movmedian')
    :type threshold_factor: float, default=3
    :param window: specify running window if 'movmean' or 'movmedian' mode is selected
    :type window: int, default=5
    :param axis: specify axis to alculate statistics for 'mean', 'median', 'movmean', 'movmean' modes
    :type axis: string
    :return: returns a DataContainer with boolean mask with 0 where outliers were detected
    :rtype: DataContainer
    '''
    
    '''             
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
                          given by quantiles = (q1,q2), 0<=q1,q2<=1. 
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
                          scaled median absolute deviation (MAD) is defined as 
                          c*median(abs(A-median(A))) where c=-1/(sqrt(2)*erfcinv(3/2))
                          Default value of threshold_factor is 3. If no axis is specified, 
                          then operates over flattened array. Alternatively operates along axis specified 
                          as dimension_label.
      - movmean           the same as mean but uses rolling mean with a specified window,
                          default window value is 5
      - movmedian         the same as mean but uses rolling median with a specified window,
                          default window value is 5

      '''

    def __init__(self,
                 mode = 'special_values',
                 threshold_value = (None, None),
                 quantiles = (None, None),
                 threshold_factor = 3,
                 window = 5,
                 axis = None):

        kwargs = {'mode': mode,
                  'threshold_value': threshold_value,
                  'threshold_factor': threshold_factor,
                  'quantiles': quantiles,
                  'window': window,
                  'axis': axis}

        super(MaskGenerator_backend, self).__init__(**kwargs)
    
    def check_input(self, data):
        
        if not (issubclass(type(data), DataContainer)):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n' +
                            ' - DataContainer')
        if self.mode not in ['special_values', 'nan', 'inf', 'threshold', 'quantile',\
                             'mean', 'median', 'movmean', 'movmedian']:
            raise Exception("Wrong mode. One of the following is expected:\n" + \
                            "special_values, nan, inf, threshold, \n quantile, mean, median, movmean, movmedian")
        
        if self.axis is not None:
                
            if self.axis not in data.dimension_labels:
                raise Exception("Wrong label is specified for axis. " + \
                    "Expected {}, got {}.".format(data.dimension_labels, self.axis))

        return True 

    def process(self):

        # get input DataContainer
        data = self.get_input()
        arr = data.as_array()
        
        # intialise mask with all ones
        mask = numpy.ones(arr.shape, dtype=numpy.bool)
        
        # if NaN or +/-Inf
        if self.mode == 'special_values':
            
            mask[numpy.logical_or(numpy.isnan(arr), numpy.isinf(arr))] = 0
        
        elif self.mode == 'nan':
            
            mask[numpy.isnan(arr)] = 0
            
        elif self.mode == 'inf':
            
            mask[numpy.isinf(arr)] = 0
            
        elif self.mode == 'threshold':
            
            if not(isinstance(self.threshold_value, tuple)):
                raise Exception("Threshold value must be given as a tuple containing two values,\n" +\
                    "use None if no threshold value is given")
                
            threshold = self._parse_threshold_value(arr)
            
            mask[numpy.logical_or(arr < threshold[0], arr > threshold[1])] = 0
            
        elif self.mode == 'quantile':
            
            if not(isinstance(self.quantiles, tuple)):
                raise Exception("Quantiles must be given as a tuple containing two values,\n " + \
                    "use None if no quantile value is given")
            
            quantile = self._parse_quantile_value(arr)
                
            mask[numpy.logical_or(arr < quantile[0], arr > quantile[1])] = 0
        
        elif self.mode == 'mean':
            
            # if mean along specific axis
            if self.axis is not None:
                
                axis = data.dimension_labels.index(self.axis)
                ndim = data.number_of_dimensions
                
                tile_par = []
                slice_obj = []
                for i in range(ndim):
                    if i == axis:
                        tile_par.append(data.dimension_labels.index(self.axis))
                        slice_obj.append(numpy.newaxis)
                    else:
                        tile_par.append(1)
                        slice_obj.append(slice(None, None, 1))
                tile_par = tuple(tile_par)
                slice_obj = tuple(slice_obj)
                
                tmp_mean = numpy.tile((numpy.mean(arr, axis=axis))[slice_obj], tile_par)
                tmp_std = numpy.tile((numpy.std(arr, axis=axis))[slice_obj], tile_par)
                mask[numpy.abs(arr - tmp_mean) > self.threshold_factor * tmp_std] = 0

            # if global mean    
            else:
                
                 mask[numpy.abs(arr - numpy.mean(arr)) > self.threshold_factor * numpy.std(arr)] = 0
        
        elif self.mode == 'median':
            
            c = -1 / (numpy.sqrt(2) * special.erfcinv(3 / 2))
            
            # if median along specific axis
            if self.axis is not None:

                axis = data.dimension_labels.index(self.axis)
                ndim = data.number_of_dimensions
                
                tile_par = []
                slice_obj = []
                for i in range(ndim):
                    if i == axis:
                        tile_par.append(data.dimension_labels.index(self.axis))
                        slice_obj.append(numpy.newaxis)
                    else:
                        tile_par.append(1)
                        slice_obj.append(slice(None, None, 1))
                tile_par = tuple(tile_par)
                slice_obj = tuple(slice_obj)

                tmp = numpy.abs(arr - numpy.tile((numpy.median(arr, axis=axis))[slice_obj], tile_par))
                median_absolute_dev = numpy.tile((numpy.median(tmp, axis=axis))[slice_obj], tile_par)
                mask[tmp > self.threshold_factor * c * median_absolute_dev] = 0
            
            # if global median
            else:
                
                tmp = numpy.abs(arr - numpy.median(arr))
                mask[tmp > self.threshold_factor * c * numpy.median(tmp)] = 0
            
        elif self.mode == 'movmean':

            # if movmean along specific axis     
            if self.axis is not None:
                
                axis = data.dimension_labels.index(self.axis)
                ndim = data.number_of_dimensions
                
                kernel = [1] * ndim
                kernel[axis] = self.window
                kernel = tuple(kernel)

                mean_array = ndimage.generic_filter(arr, numpy.mean, size=kernel, mode='reflect')
                std_array = ndimage.generic_filter(arr, numpy.std, size=kernel, mode='reflect')

                mask[numpy.abs(arr - mean_array) > self.threshold_factor * std_array] = 0
            
            # if global movmean
            else:
                
                ndim = data.number_of_dimensions

                mean_array = ndimage.generic_filter(arr, numpy.mean, size=(self.window,)*ndim, mode='reflect')
                std_array = ndimage.generic_filter(arr, numpy.std, size=(self.window,)*ndim, mode='reflect')
                
                mask[numpy.abs(arr - mean_array) > self.threshold_factor * std_array] = 0          
            
        elif self.mode == 'movmedian':

            c = -1 / (numpy.sqrt(2) * special.erfcinv(3 / 2))
            
            # if movmedian along specific axis
            if self.axis is not None:
                
                axis = data.dimension_labels.index(self.axis)
                ndim = data.number_of_dimensions
                
                # construct filter kernel
                kernel_shape = []
                for i in range(ndim):
                    if i == axis:
                        kernel_shape.append(self.window)
                    else:
                        kernel_shape.append(1)
                
                kernel_shape = tuple(kernel_shape)
                
                median_array = ndimage.median_filter(arr, footprint=kernel_shape, mode='reflect')
                
                tmp = abs(arr - median_array)
                mask[tmp > self.threshold_factor * c * ndimage.median_filter(tmp, footprint=kernel_shape, mode='reflect')] = 0
            
            # if global movmedian
            else:
                
                # construct filter kernel
                ndim = data.number_of_dimensions
                kernel_shape = tuple([self.window]*ndim)
                median_array = ndimage.median_filter(arr, size=kernel_shape, mode='reflect')
                
                tmp = abs(arr - median_array)
                mask[tmp > self.threshold_factor * c * ndimage.median_filter(tmp, size=kernel_shape, mode='reflect')] = 0
        
        else:
            raise ValueError('Mode not recognised. One of the following is expected: ' + \
                              'special_values, nan, inf, threshold, quantile, mean, median, movmean, movmedian')
        
        data.fill(arr)     
        return DataContainer(mask, deepcopy=True, dimension_labels=data.dimension_labels)
        

    def _parse_threshold_value(self, arr):

        threshold = []
        if self.threshold_value[0] is None:
            threshold.append(numpy.amin(arr))
        else:
            threshold.append(self.threshold_value[0])
            tmp_min = numpy.amin(arr)
            if self.threshold_value[0] < tmp_min:
                warnings.warn("Given threshold_value {} is smaller than min" + \
                    "value of data {}".format(input_threshold[0], tmp_min))
        
        if self.threshold_value[1] is None:
            threshold.append(numpy.amax(arr))
        else:
            threshold.append(self.threshold_value[1])
            tmp_max = numpy.amax(arr)
            if self.threshold_value[1] > tmp_max:
                warnings.warn("Given threshold_value {} is larger than max " + \
                    "value of data {}".format(self.threshold_value[1], tmp_max))
        
        if threshold[1] < threshold[0]:
            raise Exception("Upper threshold value must be larger than " + \
                "lower treshold value or min of data")
        
        return threshold
    

    def _parse_quantile_value(self, arr):

        quantile_values = []
        if self.quantiles[0] is None:
            quantile_values.append(numpy.amin(arr))
        else:
            quantile_values.append(numpy.quantile(arr, self.quantiles[0]))
        if self.quantiles[1] is None:
            quantile_values.append(numpy.amax(arr))
        else:
            quantile_values.append(numpy.quantile(arr, self.quantiles[1]))
        if self.quantiles[1] < 0 or self.quantiles[1] > 1:
            raise Exception("Quantile_values must be within 0 and 1. Got {}".format(self.quantiles))
        
        if quantile_values[1] <  quantile_values[0]:
            raise Exception("Upper quantile must be larger than lower quantile. Got ({}, {})".format(self.quantiles))

        return quantile_values
  