# -*- coding: utf-8 -*-
#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
from scipy.fft import fftshift
from scipy import constants

from tqdm import tqdm
import logging
from multiprocessing.pool import ThreadPool

class PaganinPhaseProcessor(Processor):

    @staticmethod
    def retrieve(energy_eV = 40000, delta = 1, beta = 1e-2, unit_multiplier = 1, propagation_distance = None, filter_type='paganin_method', verbose=True):
        """
        Method to create a Paganin processor to retrieve quantitative information from phase contrast images using
        the Paganin phase retrieval algorithm described in https://doi.org/10.1046/j.1365-2818.2002.01010.x 
        The phase retrieval is valid under the following assumptions
         - it is used with paraxial propagation induced phase contrast images on single-material samples
         - using intensity data which has been flat field corrected
         - and under the assumption that the Fresnel number = pixel size^2/(wavelength*propagation_distance) >> 1
        
        To use Paganin phase filtering without transmission to absorption conversion use the `PaganinPhaseProcessor.filter()` method
        in this case the conversion to absorption is not applied so the requirement to supply intensity data is relaxed
        
        Parameters
        ----------
        energy_eV: float (optional)
            Energy of the incident photon in eV, default is 40000
            
        delta: float (optional)
            Real part of the deviation of the material refractive index from 1, where refractive index n = (1 - delta) + i beta 
            energy-dependent refractive index information can be found at https://refractiveindex.info/ , default is 1
        
        beta: float (optional)
            Complex part of the material refractive index, where refractive index n = (1 - delta) + i beta
            energy-dependent refractive index information can be found at https://refractiveindex.info/ , default is 1e-2

        unit_multiplier: float (optional)
            Multiplier to convert units stored in geometry to metres, conversion applies to pixel size (and propagation distance if data.geometry.dist_center_detector is used), default is 1

        propagation_distance: float (optional)
            The sample to detector distance in meters. If not specified, the value in data.geometry.dist_center_detector will be used

        filter_type: string (optional)
            The form of the Paganin filter to use, either 'paganin_method' (default) or 'generalised_paganin_method' as described in https://iopscience.iop.org/article/10.1088/2040-8986/abbab9 
        
        verbose: boolean (optional)
            If True print progress bar (default is True)

        Returns
        -------
        Processor
            Paganin phase retrieval processor
                    

        Example
        -------
        >>> processor = PaganinPhaseProcessor.retrieve(energy_eV, delta, beta, unit_multiplier)
        >>> processor.set_input(self.data)
        >>> processor.get_output()

        or to retrieve the projected thickness of the object
        >>> processor = PaganinPhaseProcessor.retrieve(energy_eV, delta, beta, unit_multiplier)
        >>> processor.set_input(self.data)
        >>> processor.get_output(output_type='thickness')
        
        """
        processor = PaganinPhaseRetrieval(energy_eV=energy_eV, delta=delta, beta=beta, unit_multiplier=unit_multiplier, propagation_distance=propagation_distance, filter_type=filter_type, verbose=verbose)
        return processor


    @staticmethod
    def filter(energy_eV = 40000, delta = 1, beta = 1e-2, unit_multiplier = 1, propagation_distance = None, filter_type='paganin_method', verbose=True):
        '''
        Method to create a Paganin processor to filter images using the Paganin phase retrieval algorithm
        described in https://doi.org/10.1046/j.1365-2818.2002.01010.x 

        To retrieve quantitative information from phase contrast images use the `PaganinPhaseProcessor.retrieve() method` instead

        Parameters
        ----------
        energy_eV: float (optional)
            Energy of the incident photon in eV, default is 40000
            
        delta: float (optional)
            Real part of the deviation of the material refractive index from 1, where refractive index n = (1 - delta) + i beta 
            energy-dependent refractive index information can be found at https://refractiveindex.info/ , default is 1
        
        beta: float (optional)
            Complex part of the material refractive index, where refractive index n = (1 - delta) + i beta
            energy-dependent refractive index information can be found at https://refractiveindex.info/ , default is 1e-2

        unit_multiplier: float (optional)
            Multiplier to convert units stored in geometry to metres, conversion applies to pixel size (and propagation distance if data.geometry.dist_center_detector is used), default is 1

        propagation_distance: float (optional)
            The sample to detector distance in meters. If not specified, the value in data.geometry.dist_center_detector will be used. If neither are supplied, default is 10

        filter_type: string (optional)
            The form of the Paganin filter to use, either 'paganin_method' (default) or 'generalised_paganin_method' as described in https://iopscience.iop.org/article/10.1088/2040-8986/abbab9  (equation 17)
        
        verbose: boolean (optional)
            If True print progress bar (default is True)

        Returns
        -------
        Processor
            Paganin phase filter processor
                    
        Example
        -------
        >>> processor = PaganinPhaseProcessor.filter()
        >>> processor.set_input(self.data)
        >>> processor.get_output()

        '''
        
        processor = PaganinPhaseFilter(energy_eV=energy_eV, delta=delta, beta=beta, unit_multiplier=unit_multiplier, propagation_distance=propagation_distance, filter_type=filter_type, verbose=verbose)
        return processor
    
class PaganinProcessor(Processor):
    '''
    Parent class setting up Paganin processing
    '''
    def __init__(self, energy_eV = 40000, delta = 1, beta = 1e-2, unit_multiplier = 1, propagation_distance=None, filter_type='paganin_method', verbose=True):
        kwargs = {
            'energy' : energy_eV,
            'wavelength' : self.energy_to_wavelength(energy_eV),
            'delta': delta,
            'beta': beta,
            'unit_multiplier' : unit_multiplier,
            'propagation_distance_user' : propagation_distance,
            'filter_type' : filter_type,
            'verbose' : verbose,
            'output_type' : 'phase',
            'mu' : None,
            'alpha' : None,
            'pixel_size' : None,
            'propagation_distance' : None,
            'magnification' : None,
            'filter' : None
            }
        
        super(PaganinProcessor, self).__init__(**kwargs)
        
        self.__calculate_mu()
        
    def check_input(self, data):
        geometry = data.geometry

        if geometry.magnification == None:
            self.magnification = 1
        else:
            self.magnification = geometry.magnification

        if (geometry.pixel_size_h - geometry.pixel_size_v ) / \
            (geometry.pixel_size_h + geometry.pixel_size_v ) < 1e-5:
            self.pixel_size = data.geometry.pixel_size_h*self.unit_multiplier
        else:
            raise ValueError('Panel pixel size is not homogeneous up to 1e-5: got {} {}'\
                    .format( geometry.pixel_size_h, geometry.pixel_size_v )
                )

        self.__calculate_alpha()

        return True

    def get_output(self, out=None, output_type = 'phase'):
        '''
        Runs the configured processor and returns the processed data

        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return
        
        output_type: string, optional
            if 'attenuation', returns the attenuation of the sample corrected for phase effects, attenuation = µT 
            if 'thickness', returns the projected thickness T of the sample projected onto the image plane 
            if 'phase' (default), returns the phase of the beam at the material exit, phase ϕ(r⊥) = −δ T(r⊥) · 2π/λ
        
        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed

        '''
        self.output_type = output_type
        return super().get_output(out)
    
    def process(self, out=None):
        '''
        Process the Paganin for all projections using parallel processing
        '''
        data  = self.get_input()

        self.create_filter(data.get_slice(angle=0).as_array())

        must_return = False        
        if out is None:
            out = data.geometry.allocate(None)
            must_return = True
        
        original_data_order = data.dimension_labels
        original_out_order = out.dimension_labels
        data.reorder(('angle', 'vertical', 'horizontal'))
        out.reorder(('angle', 'vertical', 'horizontal'))

        with ThreadPool(4) as pool:
            if self.verbose == True:
                results = list(tqdm(pool.imap(self.process_projection, list(data.array)), total = len(list(data.array)) ))
                out.fill(np.array(results, dtype = out.dtype))
            else:
                out.fill(np.array(pool.map(self.process_projection, list(data.array)), dtype = out.dtype))
            
            pool.close()
            pool.join()
        
        data.reorder(original_data_order)
        out.reorder(original_out_order)

        if must_return:
            return out
        
    def create_filter(self, image):
        '''
        Function to create the Paganin filter, either using the paganin or generalised paganin method
        The filter is created on a mesh in Fourier space kx, ky
        '''
        Nx, Ny = image.shape

        kx,ky = np.meshgrid( 
            np.arange(-Nx/2, Nx/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            np.arange(-Ny/2, Ny/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            sparse=False, 
            indexing='ij'
            )
        
        if self.filter_type == 'paganin_method':
            kW = np.abs(kx.max()*self.pixel_size)       
            if (kW >= 1): 
                logging.warning("This algorithm is valid for k*W << 1, found np.abs(kx.max()*self.pixel_size) = {}, results may not be accurate, \
                                \nconsider using filter_type = 'generalised_paganin_method'".format(kW))
            self.filter =  (1. + self.alpha*(kx**2 + ky**2)*self.magnification)
        
        elif self.filter_type == 'generalised_paganin_method':       
            kW = np.abs(kx.max()*self.pixel_size)       
            if (kW > np.pi): 
                logging.warning("This algorithm is valid for |k*W| <= pi, found np.abs(kx.max()*self.pixel_size) = {}, results may not be accurate".format(kW))
            self.filter =  (1. - (2*self.alpha/self.pixel_size**2)*(np.cos(self.pixel_size*kx) + np.cos(self.pixel_size*ky) -2)/self.magnification)
        else:
            raise ValueError("filter_type not recognised: got {0} expected one of 'paganin_method' or 'generalised_paganin_method' or 'phase'"\
                            .format(self.output_type))
        
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
    
    def energy_to_wavelength(self, energy_eV):
        """Converts photon energy in eV to wavelength in m
        
        Parameters
        ----------
        energy_eV: float
            Photon energy in eV
        
        Returns
        -------
        float
            Photon wavelength in m

        """
        return (constants.h*constants.speed_of_light)/(energy_eV*constants.electron_volt)

class PaganinPhaseRetrieval(PaganinProcessor):
    '''
    Class for retrieving phase information
    '''
    def process_projection(self, image):

        fI = fftshift(
            fft2(self.magnification**2*image)
            )
        
        iffI = ifft2(fftshift(fI/self.filter))
        
        if self.output_type == 'attenuation':
            return -np.log(iffI)
        elif self.output_type == 'thickness':
            return -(1/self.mu)*np.log(iffI)
        elif self.output_type == 'phase':
            return (-self.delta*2*np.pi/self.wavelength)*((-1/self.mu)*np.log(iffI))
        else:
            raise ValueError("output_type not recognised: got {0} expected one of 'attenuation', 'thickness' or 'phase'"\
                            .format(self.output_type))
        
    def check_input(self, data):
        
        if self.propagation_distance_user is None: 
            if data.geometry.dist_center_detector is None:
                raise ValueError('Propagation distance not found, please provide propagation_distance as an argument or update geometry.dist_center_detector')
            elif data.geometry.dist_center_detector == 0:
                raise ValueError('Found geometry.dist_center_detector = 0, phase retrieval is not compatible with virtual magnification\
                                 please provide a real propagation_distance as an argument or update geometry.dist_center_detector')
            else:
                propagation_distance = data.geometry.dist_center_detector
                self.propagation_distance = (propagation_distance)*self.unit_multiplier
        else:
            self.propagation_distance = self.propagation_distance_user
        
        return super().check_input(data)
    
class PaganinPhaseFilter(PaganinProcessor):
    '''
    Class for Paganin filter
    '''
    def process_projection(self, image):

        fI = fftshift(
            fft2(self.magnification**2*image)
            )
        
        iffI = ifft2(fftshift(fI/self.filter))

        if self.output_type == 'attenuation':
            return iffI
        elif self.output_type == 'thickness':
            return (1/self.mu)*iffI
        elif self.output_type == 'phase':
            return (-self.delta*2*np.pi/self.wavelength)*((1/self.mu)*iffI)
        else:
            raise ValueError("output_type not recognised: got {0} expected one of 'attenuation', 'thickness' or 'phase'"\
                            .format(self.output_type))
        
    def check_input(self, data):
        
        if self.propagation_distance_user is None: 
            if data.geometry.dist_center_detector is None:
                self.propagation_distance = 10
            elif data.geometry.dist_center_detector == 0:
                raise ValueError('Found geometry.dist_center_detector = 0, phase retrieval is not compatible with virtual magnification\
                                 please provide a real propagation_distance as an argument or update geometry.dist_center_detector')
            else:
                propagation_distance = data.geometry.dist_center_detector
                self.propagation_distance = (propagation_distance)*self.unit_multiplier
        else:
            self.propagation_distance = self.propagation_distance_user
        
        return super().check_input(data)
        
        