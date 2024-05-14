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

from cil.framework import Processor
import numpy as np

from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import ifftshift
from scipy import constants

from tqdm import tqdm
import logging
from multiprocessing.pool import ThreadPool

import warnings

class PaganinProcessor(Processor):

    """
    Processor to retrieve quantitative information from phase contrast images using the Paganin phase retrieval algorithm described in 
    https://doi.org/10.1046/j.1365-2818.2002.01010.x 
    The phase retrieval is valid under the following assumptions
        - it's used with paraxial propagation induced phase contrast images on single-material samples
        - using intensity data which has been flat field corrected
        - and under the assumption that the Fresnel number = pixel size^2/(wavelength*propagation_distance) >> 1
    
    To just apply a filter to images using the Paganin method, call get_output(full_retrieval=False) in this case the pre-scaling and conversion to absorption is not applied so 
    the requirement to supply flat field corrected intensity data is relaxed
    
    Parameters
    ----------
    delta: float (optional)
        Real part of the deviation of the material refractive index from 1, where refractive index n = (1 - delta) + i beta 
        energy-dependent refractive index information for x-ray wavelengths can be found at https://henke.lbl.gov/optical_constants/getdb2.html , default is 1
    
    beta: float (optional)
        Complex part of the material refractive index, where refractive index n = (1 - delta) + i beta
        energy-dependent refractive index information for x-ray wavelengths can be found at https://henke.lbl.gov/optical_constants/getdb2.html , default is 1e-2
    
    energy: float (optional)
        Energy of the incident photon in eV, default is 40000

    geometry_unit_multiplier: float (optional)
        Multiplier to convert units stored in geometry to metres, conversion applies to pixel size or propagation distance, default is 1

    filter_type: string (optional)
        The form of the Paganin filter to use, either 'paganin_method' (default) or 'generalised_paganin_method' as described in https://iopscience.iop.org/article/10.1088/2040-8986/abbab9 
    
    Returns
    -------
    Processor
        Paganin phase retrieval processor
                
    Example
    -------
    >>> processor = PaganinProcessor()(data)
    >>> processor.get_output()

    Example
    -------
    >>> processor = PaganinProcessor(delta=1,beta=10e2)
    >>> processor.set_input(data)
    >>> processor.get_output()

    """
   
    def __init__(self, delta = 1, beta = 1e-2, energy = 40000, geometry_unit_multiplier = 1,  filter_type='paganin_method'):
        kwargs = {
            'energy' : energy,
            'wavelength' : self.energy_to_wavelength(energy),
            'delta': delta,
            'beta': beta,
            'geometry_unit_multiplier' : geometry_unit_multiplier,
            'filter_type' : filter_type,
            'mu' : None,
            'alpha' : None,
            'pixel_size' : None,
            'propagation_distance' : None,
            'magnification' : None,
            'filter' : None,
            'full_retrieval' : True,
            }
        
        super(PaganinProcessor, self).__init__(**kwargs)

    def check_input(self, data):
        geometry = data.geometry

        if geometry.magnification is None:
            warnings.warn('Magnification not found, please update data.geometry.dist_center_detector or over-ride with processor.update_magnification()')
            self.magnification = 1.0
            print('Magnification = ' + str(self.magnification) + ' (default value)')
        elif geometry.magnification == 0:
            warnings.warn('Found magnification = 0, please update data.geometry.dist_center_detector with processor.update_magnification()')
            self.magnification = 1.0
            print('Magnification = ' + str(self.magnification)+ ' (default value)')
        else:
            self.magnification = geometry.magnification
        print('Magnification = ' + str(self.magnification))

        if geometry.dist_center_detector is None:
            warnings.warn('Propagation distance not found, please update data.geometry.dist_center_detectoror over-ride with processor.update_propagation_distance()')
            self.propagation_distance = 0.1
            print('Propagation distance = ' + str(self.propagation_distance) + 'm (default value)')
        elif geometry.dist_center_detector == 0:
            warnings.warn('Found propagation distance = 0, please update the data.geometry.dist_center_detector or over-ride with processor.update_propagation_distance()')
            self.propagation_distance = 0.1
            print('Propagation distance = ' + str(self.propagation_distance) + 'm (default value)')
        else:
            self.propagation_distance = geometry.dist_center_detector*self.geometry_unit_multiplier
            print('Propagation distance = ' + str(self.propagation_distance) + ' m')

        if (geometry.pixel_size_h is None) | (geometry.pixel_size_v is None):
            warnings.warn('Pixel size not found, please update data.geometry.pixel_size_h or data.geometry.pixel_size_v or over-ride with processor.update_pixel_size()')
            
        elif (geometry.pixel_size_h == 0) | (geometry.pixel_size_v == 0):
            warnings.warn('Found pixel size = 0, please update data.geometry.pixel_size_h or data.geometry.pixel_size_v or over-ride the pixel size with processor.update_pixel_size()')
            self.pixel_size = 10e-6
            print('Pixel size = ' + str(self.propagation_distance) + ' m (default value)')
        elif (geometry.pixel_size_h - geometry.pixel_size_v ) / \
            (geometry.pixel_size_h + geometry.pixel_size_v ) >= 1e-5:
            warnings.warn('Panel pixel size is not homogeneous up to 1e-5: got {} {}, please update geometry using data.geometry.pixel_size_h or data.geometry.pixel_size_v or over-ride with processor.update_pixel_size()'\
                    .format( geometry.pixel_size_h, geometry.pixel_size_v ))
            self.pixel_size = 10e-6
            print('Pixel size = ' + str(self.propagation_distance) + ' m (default value)')
        else:
            self.pixel_size = geometry.pixel_size_h*self.geometry_unit_multiplier
            print('Pixel size = ' + str(self.propagation_distance) + ' m')
        
        return True

    def update_parameters(self, parameters):
        parameter_list = ['propagation_distance','pixel_size','magnification', 'delta', 'beta']
        for key in parameters.keys():
            if key not in parameter_list:
                raise ValueError('Parameter {} not recognised, expected one of {}.'.format(key, parameter_list))
            elif parameters[key] == None:
                raise ValueError('Parameter {} cannot be None.'.format(key))
            elif parameters[key] == 0:
                raise ValueError('Parameter {} cannot be 0.'.format(key))
            else:
                setattr(self, key, parameters[key])
        
    def create_filter(self, Nx, Ny):
        '''
        Function to create the Paganin filter, either using the paganin or generalised paganin method
        The filter is created on a mesh in Fourier space kx, ky
        '''
        
        self.__calculate_mu()
        self.__calculate_alpha()

        kx,ky = np.meshgrid( 
            np.arange(-Nx/2, Nx/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            np.arange(-Ny/2, Ny/2, 1, dtype=np.float64) * (2*np.pi)/(Ny*self.pixel_size),
            sparse=False, 
            indexing='ij'
            )
        
        if self.filter_type == 'paganin_method':
            self.filter =  ifftshift(1/(1. + self.alpha*(kx**2 + ky**2)))
        elif self.filter_type == 'generalised_paganin_method':       
            self.filter =  ifftshift(1/(1. - (2*self.alpha/self.pixel_size**2)*(np.cos(self.pixel_size*kx) + np.cos(self.pixel_size*ky) -2)))
        else:
            raise ValueError("filter_type not recognised: got {0} expected one of 'paganin_method' or 'generalised_paganin_method'"\
                            .format(self.filter_type))
        
    def process(self, out=None):

        data  = self.get_input()

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
        scaling_factor = -(1/self.mu)
        for i in tqdm(range(len(out.geometry.angles))):
            projection = data.take(indices = i, axis = out.get_dimension_axis('angle'))
            
            if self.full_retrieval==True:
                fI = fft2(self.magnification**2*projection)
                iffI = ifft2(fI*self.filter)
                processed_projection = scaling_factor*np.log(iffI)
            else:
                fI = fft2(projection)
                processed_projection = ifft2(fI*self.filter)
                
            out.fill(np.squeeze(processed_projection), angle = i)

        if must_return:
            return out
        
    def get_output(self, out=None, full_retrieval=True):
        self.full_retrieval = full_retrieval
        return super().get_output(out)

        
    def __calculate_mu(self):
        """
        Function to calculate the linear attenutation coefficient mu
        """
        self.mu = 4.0*np.pi*self.beta/self.wavelength   

    def __calculate_alpha(self):
        '''
        Function to calculate alpha, a constant defining the Paganin filter strength
        '''
        self.alpha = self.propagation_distance*self.delta/self.mu
    
    def energy_to_wavelength(self, energy):
        """Converts photon energy in eV to wavelength in m
        
        Parameters
        ----------
        energy: float
            Photon energy in eV
        
        Returns
        -------
        float
            Photon wavelength in m

        """
        return (constants.h*constants.speed_of_light)/(energy*constants.electron_volt)
    