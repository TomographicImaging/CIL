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
import logging
import numpy
import numba
from cil.utilities import multiprocessing as cil_mp

log = logging.getLogger(__name__)

class TransmissionAbsorptionConverter(DataProcessor):

    r'''Processor to convert from transmission measurements to absorption
    based on the Beer-Lambert law, by:
    - dividing by white_level, 
    - clipping data to min_intensity (any values below min intensity are set to min_intensity) 
    - taking the logarithm and multiplying by minus 1. 
    If the data contains zero or negative values, inf or NaN values will be present in the output.

    Parameters:
    -----------
    min_intensity: float, default=0.0
        Clips data below this value to ensure log is taken of positive numbers only. 
    
    white_level: float, default=1.0
        A float defining incidence intensity in the Beer-Lambert law.

    accelerated: bool, default=True
        Specify whether to use multi-threading using numba. 
    
    Returns:
    --------
    AcquisitionData, ImageData or DataContainer depending on input data type, return is suppressed if 'out' is passed

    Notes:
    ------
    If it's unknown whether the data contains zero or negative values, it's recommended to set min_intensity to a 
    small positive value.
    '''

    def __init__(self,
                 min_intensity = 0.0,
                 white_level = 1.0,
                 accelerated = True,
                 ):

        kwargs = {'min_intensity': min_intensity,
                  'white_level': white_level,
                  '_accelerated': accelerated}

        super(TransmissionAbsorptionConverter, self).__init__(**kwargs)

    def check_input(self, data):

        if not (issubclass(type(data), DataContainer)):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n' +
                            ' - DataContainer')

        if self.min_intensity <= 0:
            log.info(f"\n Current min_intensity = {self.min_intensity}: output may contain NaN or inf. Ensure your data only contains positive values or set min_intensity to a small positive value.")
        
        return True

    def process(self, out=None):

        data = self.get_input()

        if out is None:
            out = data.geometry.allocate(None)

        arr_in = data.as_array()
        arr_out = out.as_array()

        # we choose an arbitrary chunk size of 6400, which is a multiple of 32, to allow for efficient threading
        chunk_size = 6400
        num_chunks = data.size // chunk_size
        if (self._accelerated):
            remainder = data.size % chunk_size
            num_threads_original = numba.get_num_threads()
            numba.set_num_threads(cil_mp.NUM_THREADS)
            numba_loop(arr_in, num_chunks, chunk_size, remainder, 1/self.white_level, self.min_intensity,  arr_out)
            numba.set_num_threads(num_threads_original)

        else:
            if self.white_level != 1:
                numpy.multiply(arr_in, 1/self.white_level, out=arr_out)
                arr_in = arr_out

            if self.min_intensity > 0:
                numpy.clip(arr_in, self.min_intensity, None, out=arr_out)
                arr_in = arr_out
                
            numpy.log(arr_in, out=arr_out)
            numpy.negative(arr_out,out=arr_out)

        out.fill(arr_out)

        return out
    
@numba.njit(parallel=True)
def numba_loop(arr_in, num_chunks, chunk_size, remainder, multiplier, min_intensity, arr_out):
    in_flat = arr_in.ravel()
    out_flat = arr_out.ravel()
    for i in numba.prange(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        out_flat[start:end] = numpy.multiply(in_flat[start:end], multiplier)
        out_flat[start:end] = numpy.clip(out_flat[start:end], min_intensity, None)
        out_flat[start:end] = -numpy.log(out_flat[start:end])


    if remainder > 0:
        start = num_chunks * chunk_size
        end = start + remainder
        out_flat[start:end] = numpy.multiply(in_flat[start:end], multiplier)
        out_flat[start:end] = numpy.clip(out_flat[start:end], min_intensity, None)
        out_flat[start:end] = -numpy.log(out_flat[start:end])



