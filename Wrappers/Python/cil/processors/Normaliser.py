#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
import warnings

class Normaliser(Processor):
    '''Normalisation based on flat and dark

    This processor read in a AcquisitionData and normalises it based on
    the instrument reading with and without incident photons or neutrons.

    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSet
    '''

    def __init__(self, flat_field = None, dark_field = None, tolerance = 1e-5):
        kwargs = {
                  'flat_field'  : flat_field,
                  'dark_field'  : dark_field,
                  # very small number. Used when there is a division by zero
                  'tolerance'   : tolerance
                  }

        #DataProcessor.__init__(self, **kwargs)
        super(Normaliser, self).__init__(**kwargs)
        if not flat_field is None:
            self.set_flat_field(flat_field)
        if not dark_field is None:
            self.set_dark_field(dark_field)

    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or\
           dataset.number_of_dimensions == 2:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

    def set_dark_field(self, df):
        if type(df) is numpy.ndarray:
            if len(numpy.shape(df)) == 3:
                raise ValueError('Dark Field should be 2D')
            elif len(numpy.shape(df)) == 2:
                self.dark_field = df
        elif issubclass(type(df), DataContainer):
            self.dark_field = self.set_dark_field(df.as_array())

    def set_flat_field(self, df):
        if type(df) is numpy.ndarray:
            if len(numpy.shape(df)) == 3:
                raise ValueError('Flat Field should be 2D')
            elif len(numpy.shape(df)) == 2:
                self.flat_field = df
        elif issubclass(type(df), DataContainer):
            self.flat_field = self.set_flat_field(df.as_array())


    @staticmethod
    def estimate_normalised_error(projection, flat, dark, delta_flat, delta_dark):
        '''returns the estimated relative error of the normalised projection

        n = (projection - dark) / (flat - dark)
        Dn/n = (flat-dark + projection-dark)/((flat-dark)*(projection-dark))*(Df/f + Dd/d)
        '''
        a = (projection - dark)
        b = (flat-dark)
        df = delta_flat / flat
        dd = delta_dark / dark
        rel_norm_error = (b + a) / (b * a) * (df + dd)
        return rel_norm_error

    def process(self, out=None):

        input_arr = self.get_input().array

        if out is None:
            out = self.get_input().geometry.allocate(None)
        
        out_array = out.array

        if self.flat_field is None and self.dark_field is None:
            raise ValueError('No flat or dark field provided. Please provide at least one.')
        
        if self.dark_field is None:
            offset = 0
        else:
            if input_arr.shape[-2::] != self.dark_field.shape:
                raise ValueError('Dark field and projections size do not match.')
            offset = self.dark_field

        if self.flat_field is None:
            scale = 1
        else:
            if input_arr.shape[-2::] != self.flat_field.shape:
                raise ValueError('Flat field and projections size do not match.')
            scale = self.flat_field - offset

        zero_err = False
        numpy.seterr(divide='warn')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            denom_inv = 1.0 / scale
            # Check if any warnings were raised
            if len(w) > 0 and issubclass(w[-1].category, RuntimeWarning):
                zero_err = True
        
        if self.dark_field is not None:
            numpy.subtract(input_arr, offset, out=out_array)
            input_arr = out_array

        if self.flat_field is not None:
            numpy.multiply(input_arr, denom_inv, out=out_array)

        if zero_err:
            warnings.warn("Divide by zero detected. These values will be substitued by the set tolerance. Please check your data for bad pixels.", UserWarning, stacklevel=2)
            # create map of zeros in scale array
            zero_map = numpy.where(scale == 0)
            # set these to the tolerance value in all projections
            for i in range(out_array.shape[0]):
                out_array[i][zero_map] = self.tolerance

        return out
