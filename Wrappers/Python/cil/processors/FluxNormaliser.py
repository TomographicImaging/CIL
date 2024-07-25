#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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

from cil.framework import Processor, DataContainer, AcquisitionData,\
 AcquisitionGeometry, ImageGeometry, ImageData
import numpy

class FluxNormaliser(Processor):
    '''Flux normalisation based on float or region of interest

    This processor reads in a AcquisitionData and normalises it based on
    a float or array of float values, or a region of interest.

    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSet
    '''

    def __init__(self, flux=None, roi=None, tolerance=1e-5):
            kwargs = {
                    'flux'  : flux,
                    'roi'  : roi,
                    # very small number. Used when there is a division by zero
                    'tolerance'   : tolerance
                    }
            super(FluxNormaliser, self).__init__(**kwargs)
            
    def check_input(self, dataset):
        flux_size = (numpy.shape(self.flux))
        if len(flux_size) > 0:
            data_size = numpy.shape(dataset.geometry.angles)
            if data_size != flux_size:
                raise ValueError("Flux must be a scalar or array with length \
                                    \n = data.geometry.angles, found {} and {}"
                                    .format(flux_size, data_size))
            
        return True

    def process(self, out=None):

        data = self.get_input()

        if out is None:
            out = data.copy()

        flux_size = (numpy.shape(self.flux))
        
        proj_axis = data.get_dimension_axis('angle')
        slice_proj = [slice(None)]*len(data.shape)
        slice_proj[proj_axis] = 0
        
        f = self.flux
        for i in range(numpy.shape(data)[proj_axis]):
            if len(flux_size) > 0:
                f = self.flux[i]

            slice_proj[proj_axis] = i
            with numpy.errstate(divide='ignore', invalid='ignore'):
                out.array[tuple(slice_proj)] = data.array[tuple(slice_proj)]/f
                        
        out.array[ ~ numpy.isfinite( out.array )] = self.tolerance

        return out
