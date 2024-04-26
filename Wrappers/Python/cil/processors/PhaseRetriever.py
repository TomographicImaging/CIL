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

class PhaseRetriever(Processor):

    @staticmethod
    def Paganin(delta = 1, beta = 1e-2, energy = 40000, propagation_distance=None, magnification=None, pixel_size=None, geometry_unit_multiplier = 1,  filter_type='paganin_method'):
        """
        Method to create a Paganin processor to retrieve quantitative information from phase contrast images using the Paganin phase retrieval algorithm described in 
        https://doi.org/10.1046/j.1365-2818.2002.01010.x The phase retrieval is valid under the following assumptions
         - it is used with paraxial propagation induced phase contrast images on single-material samples
         - using intensity data which has been flat field corrected
         - and under the assumption that the Fresnel number = pixel size^2/(wavelength*propagation_distance) >> 1
        
        To filter images using the Paganin method use the `Filter.low_pass_Paganin()` method in this case the conversion to absorption is not applied so 
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

        propagation_distance: float (optional)
            The sample to detector distance in meters. If not specified, the value in data.geometry.dist_center_detector will be used

        magnification: float (optional)
            The optical magnification at the detector. If not specified, the value in data.geometry.magnification will be used

        pixel_size: float (optional)
            The detector pixel size. If not specified, values from in data.geometry.pixel_size_h and pixel_size_v will be used

        geometry_unit_multiplier: float (optional)
            Multiplier to convert units stored in geometry to metres, conversion applies to pixel size or propagation distance (only if the geometry value is used not if 
            user specified), default is 1

        filter_type: string (optional)
            The form of the Paganin filter to use, either 'paganin_method' (default) or 'generalised_paganin_method' as described in https://iopscience.iop.org/article/10.1088/2040-8986/abbab9 
        
        Returns
        -------
        Processor
            Paganin phase retrieval processor
                    

        Example
        -------
        >>> processor = PaganinPhaseProcessor.retrieve(energy, delta, beta, geometry_unit_multiplier)
        >>> processor.set_input(self.data)
        >>> processor.get_output()
        
        """
        processor = PaganinPhaseRetriever(delta=delta, beta=beta, energy=energy, propagation_distance=propagation_distance, magnification=magnification, pixel_size=pixel_size, geometry_unit_multiplier=geometry_unit_multiplier, filter_type=filter_type)
        return processor

    
class PaganinPhaseRetriever(PhaseRetriever):

    def __init__(self, energy, delta, beta, geometry_unit_multiplier, propagation_distance, magnification, pixel_size, filter_type):
        kwargs = {
            'energy' : energy,
            'wavelength' : self.energy_to_wavelength(energy),
            'delta': delta,
            'beta': beta,
            'geometry_unit_multiplier' : geometry_unit_multiplier,
            'pixel_size_user' : pixel_size,
            'propagation_distance_user' : propagation_distance,
            'magnification_user' : magnification,
            'filter_type' : filter_type,
            'mu' : None,
            'alpha' : None,
            'pixel_size' : None,
            'propagation_distance' : None,
            'magnification' : None,
            'filter' : None
            }
        
        super(PhaseRetriever, self).__init__(**kwargs)

    def check_input(self, data):
        geometry = data.geometry

        # if propagation_distance is not specified by the user, use the value in geometry
        if self.propagation_distance_user is None: 
            if data.geometry.dist_center_detector is None:
                raise ValueError('Propagation distance not found, please provide propagation_distance as an argument or update geometry.dist_center_detector')
            elif data.geometry.dist_center_detector == 0:
                raise ValueError('Found geometry.dist_center_detector = 0, phase retrieval is not compatible with virtual magnification\
                                 please provide a real propagation_distance as an argument or update geometry.dist_center_detector')
            else:
                propagation_distance = geometry.dist_center_detector
                self.propagation_distance = (propagation_distance)*self.geometry_unit_multiplier
        else:
            self.propagation_distance = self.propagation_distance_user
        
        # if magnification is not specified by the user, use the value in geometry, or default = 1
        if self.magnification_user is None:
            if geometry.magnification == None:
                self.magnification = 1
            else:
                self.magnification = geometry.magnification
        else:
            self.magnification = self.magnification_user

        if self.pixel_size_user is None:
            if (geometry.pixel_size_h - geometry.pixel_size_v ) / \
                (geometry.pixel_size_h + geometry.pixel_size_v ) < 1e-5:
                self.pixel_size = (data.geometry.pixel_size_h*self.geometry_unit_multiplier)
            else:
                raise ValueError('Panel pixel size is not homogeneous up to 1e-5: got {} {}'\
                        .format( geometry.pixel_size_h, geometry.pixel_size_v )
                    )
        else:
            self.pixel_size = self.pixel_size_user
        
        return True
        
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
            self.filter =  ifftshift(1/(1. + self.alpha*(kx**2 + ky**2)/self.magnification))
        
        elif self.filter_type == 'generalised_paganin_method':       
            self.filter =  ifftshift(1/(1. - (2*self.alpha/self.pixel_size**2)*(np.cos(self.pixel_size*kx) + np.cos(self.pixel_size*ky) -2)/self.magnification))
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
            fI = fft2(self.magnification**2*projection)
            iffI = ifft2(fI*self.filter)
            processed_projection = scaling_factor*np.log(iffI)
            out.fill(np.squeeze(processed_projection), angle = i)
 
        if must_return:
            return out
        
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
        