# -*- coding: utf-8 -*-
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
#from cil.framework import Processor

from cil.framework import Processor
from cil.processors import PhaseRetriever

import numpy as np
from tqdm import tqdm
from scipy.fft import fft2
from scipy.fft import ifft2

class Filter(Processor): 

    @staticmethod
    def low_pass_Paganin(delta_beta = 1e2, energy = 40000, propagation_distance=None, magnification=None, pixel_size=None, geometry_unit_multiplier=1):
        '''
        Method to filter a set of projections using a low pass filter based on Paganin phase retrieval described in https://doi.org/10.1046/j.1365-2818.2002.01010.x 
        In this implmemntation, the strength of the filter can be set using the delta_beta ratio alone. To retrieve thickness information from phase contrast 
        images use the `PhaseRetriever.Paganin() method` instead

        Parameters
        ----------           
        delta_beta: float
            Filter strength, can be given by the ratio of the real and complex part of the material refractive index, where refractive index n = (1 - delta) + i beta 
            (energy-dependent refractive index information for x-ray wavelengths can be found at https://henke.lbl.gov/optical_constants/getdb2.html ) default is 1e2

        energy: float (optional)
            Energy of the incident photon in eV, default is 40000

        propagation_distance: float (optional)
            The sample to detector distance in meters. If not specified, either the value in data.geometry.dist_center_detector will be used or a defeault of 1

        magnification: float (optional)
            The optical magnification at the detector. If not specified, either the value in data.geometry.magnification will be used or a default of 1

        pixel_size: float (optional)
            The detector pixel size. If not specified, either values in data.geometry.pixel_size_h and pixel_size_v will be used or a default of 10e-6

        geometry_unit_multiplier: float (optional)
            Multiplier to convert units stored in geometry to metres, conversion applies to pixel size or propagation distance (only if the geometry value is used not 
            if user specified), default is 1

        Returns
        -------
        Processor
            Paganin phase filter processor
                    
        Example
        -------
        >>> processor = Filter.Paganin()
        >>> processor.set_input(self.data)
        >>> processor.get_output()

        '''
        return PaganinFilter(delta_beta=delta_beta, energy=energy, propagation_distance=propagation_distance, magnification=magnification, pixel_size=pixel_size, geometry_unit_multiplier=geometry_unit_multiplier)
    
class PaganinFilter(Filter):
    def __init__(self, delta_beta, energy, propagation_distance, magnification, pixel_size, geometry_unit_multiplier):
        
        kwargs = {
        'delta_beta': delta_beta,
        'filter' : None,
        'energy' : energy,
        'propagation_distance_user' : propagation_distance,
        'magnification_user' : magnification,
        'pixel_size' : None,
        'propagation_distance' : None,
        'magnification' : None,
        'pixel_size_user' : pixel_size,
        'geometry_unit_multiplier' : geometry_unit_multiplier}

        super(PaganinFilter, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(self.delta_beta, (int, float)):
            raise TypeError('delta_beta must be a real number, got type '.format(type(self.delta_beta)))
        
        # if magnification is not specified by the user, use the value in geometry, or default = 1
        if self.magnification_user is None:
            if data.geometry.magnification == None:
                self.magnification = 1
            else:
                self.magnification = data.geometry.magnification
        else:
            self.magnification = self.magnification_user

        # if propagation_distance is not specified by the user, use the value in geometry, or default = 1
        if self.propagation_distance_user is None: 
            if data.geometry.dist_center_detector is None:
                self.propagation_distance = 1
            elif data.geometry.dist_center_detector == 0:
                self.propagation_distance = 1
            else:
                self.propagation_distance = data.geometry.dist_center_detector*self.geometry_unit_multiplier
        else:
            if not isinstance(self.propagation_distance_user, (int, float)):
                raise TypeError('propagation_distance must be a real number, got type '.format(type(self.propagation_distance_user)))
            self.propagation_distance = self.propagation_distance_user        

        # if pixel_size is not specified by the user, use the value in geometry, or default = 10e-6
        if self.pixel_size_user is None:
            if (data.geometry.pixel_size_h == None) | (data.geometry.pixel_size_v == None):
                self.pixel_size = 10e-6
            else:
                if (data.geometry.pixel_size_h - data.geometry.pixel_size_v ) / \
                    (data.geometry.pixel_size_h + data.geometry.pixel_size_v ) < 1e-5:
                    self.pixel_size = (data.geometry.pixel_size_h*self.geometry_unit_multiplier)
                else:
                    raise ValueError('Panel pixel size is not homogeneous up to 1e-5: got {} {}'\
                            .format( data.geometry.pixel_size_h, data.geometry.pixel_size_v )
                        )
        else:
            if not isinstance(self.pixel_size_user, (int, float)):
                raise TypeError('pixel_size must be a real number, got type '.format(type(self.pixel_size_user)))
            self.pixel_size = self.pixel_size_user

        return True

    def create_filter(self, Nx, Ny):
        processor = PhaseRetriever.Paganin(delta = 1, beta = 1/self.delta_beta, energy = self.energy, propagation_distance = self.propagation_distance, magnification = self.magnification, 
                                           pixel_size=self.pixel_size, geometry_unit_multiplier = self.geometry_unit_multiplier, filter_type='paganin_method')
        processor.set_input(self.get_input())
        processor.create_filter(Nx, Ny)
        self.filter = processor.filter
    
    def process(self, out=None):

        data = self.get_input()

        must_return = False        
        if out is None:
            out = data.geometry.allocate(None)
            must_return = True

        filter_shape = np.shape(data.get_slice(angle=0).as_array())
        if data.geometry.dimension == '2D':
            data = np.expand_dims(data.as_array(),2)
        else:
            data = data.as_array()

        filter_shape = np.shape(data.take(indices = 0, axis = out.get_dimension_axis('angle')))
        self.create_filter(filter_shape[0], filter_shape[1])

        for i in tqdm(range(len(out.geometry.angles))):
            projection = data.take(indices = i, axis = out.get_dimension_axis('angle'))
            iffI = ifft2(fft2(projection)*self.filter)
            out.fill(np.squeeze(iffI), angle = i)
 
        if must_return:
            return out
        

    