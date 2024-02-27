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

from cil.framework import DataProcessor, AcquisitionData, ImageData, DataContainer
import warnings
import numpy


class TransmissionAbsorptionConverter(DataProcessor):

    r'''Processor to convert from transmission measurements to absorption
    based on the Beer-Lambert law
    
    :param white_level: A float defining incidence intensity in the Beer-Lambert law.
    :type white_level: float, optional
    :param min_intensity: A float defining some threshold to avoid 0 in log, is applied after normalisation by white_level
    :type min_intensity: float, optional
    :return: returns AcquisitionData, ImageData or DataContainer depending on input data type, return is suppressed if 'out' is passed
    :rtype: AcquisitionData, ImageData or DataContainer
    
    Processor first divides by white_level (default=1) and then take negative logarithm. 
    Elements below threshold (after division by white_level) are set to threshold.
    '''

    def __init__(self,
                 min_intensity = 0.0,
                 white_level = 1.0
                 ):

        kwargs = {'min_intensity': min_intensity,
                  'white_level': white_level}

        super(TransmissionAbsorptionConverter, self).__init__(**kwargs)
    
    def check_input(self, data):
        
        if not (issubclass(type(data), DataContainer)):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n' +
                            ' - DataContainer')

        if data.min() <= 0 and self.min_intensity <= 0:
            raise ValueError('Zero or negative values found in the dataset. Please use `min_intensity` to provide a clipping value.')

        return True 

    def process(self, out=None):

        data = self.get_input()

        return_val = False
        if out is None:
            out = data.geometry.allocate(None)
            return_val = True

        arr_in = data.as_array()
        arr_out = out.as_array()

        #whitelevel
        if self.white_level != 1:
            numpy.divide(arr_in, self.white_level, out=arr_out)
            arr_in = arr_out

        #threshold
        if self.min_intensity > 0:
            numpy.clip(arr_in, self.min_intensity, None, out=arr_out)
            arr_in = arr_out

        #beer-lambert
        numpy.log(arr_in,out=arr_out)
        numpy.negative(arr_out,out=arr_out)
        
        out.fill(arr_out)

        if return_val:
            return out
