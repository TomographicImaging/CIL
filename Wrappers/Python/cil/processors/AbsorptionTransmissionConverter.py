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


from cil.framework import DataProcessor, AcquisitionData, ImageData, DataContainer, AcquisitionGeometry, ImageGeometry
import warnings
import numpy


class AbsorptionTransmissionConverter(DataProcessor):

    '''Processor to convert from absorption measurements to transmission

    :param white_level: A float defining incidence intensity in the Beer-Lambert law.
    :type white_level: float, optional
    :return: returns AcquisitionData, ImageData or DataContainer depending on input data type
    :rtype: AcquisitionData, ImageData or DataContainer

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

        data = self.get_input()
        if out is None:
            out = data.multiply(-1.0)
        else:
            data.multiply(-1.0, out=out)

        out.exp(out=out)
        out.multiply(numpy.float32(self.white_level), out=out)
        return out
