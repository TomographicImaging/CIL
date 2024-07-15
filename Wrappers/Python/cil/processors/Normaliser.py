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

class Normaliser(Processor):

    @staticmethod
    def flat_and_dark(flat_field=None, dark_field=None, tolerance=1e-5):
        return Normaliser_flat_and_dark(flat_field=flat_field, 
                                        dark_field=dark_field, 
                                        tolerance=tolerance)
    
    @staticmethod
    def flux(flux, tolerance=1e-5):
        return Normaliser_flux(flux=flux, 
                                        tolerance=tolerance)
    
    @staticmethod
    def region_of_interest(roi, tolerance=1e-5):
        return Normaliser_region_of_interest(roi=roi, tolerance=tolerance)
    
class Normaliser_flat_and_dark(Processor):

    '''Normalisation based on flat and dark

    This processor read in a AcquisitionData and normalises it based on
    the instrument reading with and without incident photons or neutrons.
    Parameters
    ----------
    flat_field: AcquisitionData
        2D projection with flat field (or stack)

    dark_field: AcquisitionData
        2D projection with dark field (or stack)

    tolerance: float
        Tolerance of the calculation, used when there is a division by zero

    Returns
    -------
    AcquisitionData
        Normalised data
    '''

    def __init__(self, flat_field, dark_field, tolerance):
        kwargs = {
                  'flat_field'  : flat_field,
                  'dark_field'  : dark_field,
                  # very small number. Used when there is a division by zero
                  'tolerance'   : tolerance
                  }

        #DataProcessor.__init__(self, **kwargs)
        super(Normaliser_flat_and_dark, self).__init__(**kwargs)
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
    def Normalise_projection(projection, flat, dark, tolerance):
        a = (projection - dark)
        b = (flat-dark)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            c = numpy.true_divide( a, b )
            c[ ~ numpy.isfinite( c )] = tolerance # set to not zero if 0/0
        return c

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

        projections = self.get_input()
        dark = self.dark_field
        flat = self.flat_field

        if projections.number_of_dimensions == 3:
            if not (projections.shape[1:] == dark.shape and \
               projections.shape[1:] == flat.shape):
                raise ValueError('Flats/Dark and projections size do not match.')


            a = numpy.asarray(
                    [ Normaliser.Normalise_projection(
                            projection, flat, dark, self.tolerance) \
                     for projection in projections.as_array() ]
                    )
        elif projections.number_of_dimensions == 2:
            a = Normaliser.Normalise_projection(projections.as_array(),
                                                flat, dark, self.tolerance)
        y = type(projections)( a , True,
                    dimension_labels=projections.dimension_labels,
                    geometry=projections.geometry)
        return y

class Normaliser_flux(Processor):
    '''Normalisation based on beam flux

    This processor read in a AcquisitionData and normalises it based on
    the beam flux measured during the experiment.
    Parameters
    ----------
    flux: float or array of floats
        The beam flux measured during the experiment, either a single float or
        an array of floats with size equal to the number of projections
    tolerance: float
        Tolerance of the calculation, used when there is a division by zero

    Output: AcquisitionDataSet
    '''

    def __init__(self, flux, tolerance):
        
        kwargs = {
                  'flux'  : flux,
                  # very small number. Used when there is a division by zero
                  'tolerance'   : tolerance
                  }
        super(Normaliser_flux, self).__init__(**kwargs)
        
        
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
        for i in range(len(data.geometry.angles)):
            if len(flux_size) > 0:
                f = self.flux[i]

            slice_proj[proj_axis] = i
            with numpy.errstate(divide='ignore', invalid='ignore'):
                out.array[tuple(slice_proj)] = data.array[tuple(slice_proj)]/f
                        
        out.array[ ~ numpy.isfinite( out.array )] = self.tolerance

        return out

            
class Normaliser_region_of_interest(Processor):
    def __init__(self, roi, tolerance):
        kwargs = {
                  'flux'  : roi,
                  # very small number. Used when there is a division by zero
                  'tolerance'   : tolerance
                  }
        

