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
    :return: returns AcquisitionData, ImageData or DataContainer depending on input data type
    :rtype: AcquisitionData, ImageData or DataContainer
    
    Processor first divides by white_level (default=1) and then take negative logarithm. 
    Elements below threshold (after division by white_level) are set to threshold.
    '''

    def __init__(self,
                 min_intensity = 0,
                 white_level = 1
                 ):

        kwargs = {'min_intensity': min_intensity,
                  'white_level': white_level}

        super(TransmissionAbsorptionConverter, self).__init__(**kwargs)
    
    def check_input(self, data):
        
        if not (issubclass(type(data), DataContainer)):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n' +
                            ' - DataContainer')
        return True 

    def process(self, out=None):

        data = self.get_input()

        white_level = numpy.float32(self.white_level)

        if out is None:
            out = data.divide(white_level)
        else:
            data.divide(white_level, out=out)

        arr = out.as_array()
        threshold = numpy.float32(self.min_intensity)
        threshold_indices = arr < threshold
        arr[threshold_indices] = threshold
        out.fill(arr)

        try:
            out.log(out=out)
        except RuntimeWarning:
            raise ValueError('Zero encountered in log. Please set threshold to some value to avoid this.')
                
        out.multiply(-1.0,out=out)
        return out
