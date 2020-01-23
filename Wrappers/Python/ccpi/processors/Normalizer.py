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

from ccpi.framework import DataProcessor, DataContainer, AcquisitionData,\
 AcquisitionGeometry, ImageGeometry, ImageData
import numpy

class Normalizer(DataProcessor):
    '''Normalization based on flat and dark
    
    This processor read in a AcquisitionData and normalises it based on 
    the instrument reading with and without incident photons or neutrons.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self, flat_field = None, dark_field = None, tolerance = 1e-5):
        kwargs = {
                  'flat_field'  : flat_field, 
                  'dark_field'  : dark_field,
                  # very small number. Used when there is a division by zero
                  'tolerance'   : tolerance
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(Normalizer, self).__init__(**kwargs)
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
    def normalize_projection(projection, flat, dark, tolerance):
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
                    [ Normalizer.normalize_projection(
                            projection, flat, dark, self.tolerance) \
                     for projection in projections.as_array() ]
                    )
        elif projections.number_of_dimensions == 2:
            a = Normalizer.normalize_projection(projections.as_array(), 
                                                flat, dark, self.tolerance)
        y = type(projections)( a , True, 
                    dimension_labels=projections.dimension_labels,
                    geometry=projections.geometry)
        return y
    