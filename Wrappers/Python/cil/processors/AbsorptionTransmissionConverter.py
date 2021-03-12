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
import warnings
import numpy


class AbsorptionTransmissionConverter(DataProcessor):

    '''
    Processor to convert from absorption measurements to transmission
    Processor first multiplies data by -1, then calculates exponent 
    and scales result by white_level (default=1)
    '''

    def __init__(self,
                 white_level=1):

        kwargs = {'white_level': white_level}

        super(AbsorptionTransmissionConverter, self).__init__(**kwargs)

    def check_input(self, data):
        
        if not (issubclass(type(data), DataContainer)):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n' +
                            ' - DataContainer')
        return True 

    def process(self, out=None):

        if out is None:
        
            data = self.get_input().clone()
            data.__imul__(-1)
            data.exp(out=data)
            data.__imul__(self.white_level)
        
            return data

        else:

            out.__imul__(-1)
            out.exp(out=out)
            out.__imul__(self.white_level)
