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
import numba
from cil.utilities import multiprocessing as cil_mp

class TransmissionAbsorptionConverter(DataProcessor):

    r'''Processor to convert from transmission measurements to absorption
    based on the Beer-Lambert law

    Parameters:
    -----------
    min_intensity: float, default=0
        Clips data below this value to ensure log is taken of positive numbers only. 
        If it's unknown whether the data contains non-positive values, set this parameter to a small positive value.
    
    white_level: float, default=1.0
        A float defining incidence intensity in the Beer-Lambert law.

    accelerated: bool, default=True
        Specify whether to use multi-threading using numba. 
    
    Returns:
    --------
    AcquisitionData, ImageData or DataContainer depending on input data type, return is suppressed if 'out' is passed

    Notes:
    ------
    Processor first divides by white_level, then clips data to min_intensity (elements below min intensity are set to min_intensity) 
    and then takes negative logarithm. If non-positive values are present in the data after clipping, NaN and inf values will be present in the output,
    it's therefore recommended to set min_intensity to a small positive value if it's unknown whether the data contains non-positive values.
    '''

    def __init__(self,
                 min_intensity = 0,
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
            warning = f"\n Current min_intensity = {self.min_intensity}: ensure your data only contains positive values or set min_intensity to a small positive value, otherwise output may contain NaN or inf."
            warnings.warn(warning)
        
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
        # Use numba if _accelerated is True and if the number of chunks is greater than 5, to avoid the overhead of threading when the data is small
        if (self._accelerated) & (num_chunks > 5):
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
        # if numpy.any(out_flat[start:end]<=0):
        #     warning_flag = True
        out_flat[start:end] = -numpy.log(out_flat[start:end])


    if remainder > 0:
        start = num_chunks * chunk_size
        end = start + remainder
        out_flat[start:end] = numpy.multiply(in_flat[start:end], multiplier)
        out_flat[start:end] = numpy.clip(out_flat[start:end], min_intensity, None)
        # if numpy.any(out_flat[start:end]<=0):
        #     warning_flag = True
        out_flat[start:end] = -numpy.log(out_flat[start:end])



