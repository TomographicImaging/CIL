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
from cil.processors import Padder, Slicer
import numpy as np

from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import ifftshift
from scipy import constants

from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

class PaganinProcessor(Processor):

    """
    Processor to retrieve quantitative information from phase contrast images using the Paganin phase retrieval algorithm described in [1]
     
    The phase retrieval is valid under the following assumptions
        - it's used with paraxial propagation-induced phase contrast images on single-material samples
        - using intensity data which has been flat field corrected
        - and under the assumption that the Fresnel number = pixel size^2/(wavelength*propagation_distance) >> 1
    
    To just apply a filter to images using the Paganin method, call get_output(full_retrieval=False) in this case the pre-scaling and conversion to absorption is not applied so 
    the requirement to supply flat field corrected intensity data is relaxed
    
    Parameters
    ----------
    delta: float (optional)
        Real part of the deviation of the material refractive index from 1, where refractive index n = (1 - delta) + i beta\
        energy-dependent refractive index information for x-ray wavelengths can be found at [2], default is 1
    
    beta: float (optional)
        Complex part of the material refractive index, where refractive index n = (1 - delta) + i beta\
        energy-dependent refractive index information for x-ray wavelengths can be found at [2], default is 1e-2
    
    energy: float (optional)
        Energy of the incident photon in eV, default is 40000

    geometry_unit: float (optional)
        The units of distance describing parameters stored in geometry, must be one of 'm', 'cm', 'mm' or 'um' default is mm

    filter_type: string (optional)
        The form of the Paganin filter to use, either 'paganin_method' (default) or 'generalised_paganin_method' as described in [3] 

    pad: int (optional)
        Number of pixels to pad the image in Fourier space to reduce aliasing, default is 0 
    
    [1] https://doi.org/10.1046/j.1365-2818.2002.01010.x 
    [2] https://henke.lbl.gov/optical_constants/getdb2.html 
    [3] https://iopscience.iop.org/article/10.1088/2040-8986/abbab9
    
    Returns
    -------
    Processor
        AcquisitionData corrected for phase effects, retrieved sample thickness in m or (if get_output(full_retrieval=False)) filtered data 
                
    Example
    -------
    >>> processor = PaganinProcessor(delta=5, beta=0.05, energy=18000)
    >>> processor.set_input(data)
    >>> thickness = processor.get_output()

    Example
    -------
    >>> processor = PaganinProcessor(delta=1,beta=10e2)
    >>> processor.set_input(data)
    >>> filtered_image = processor.get_output(full_retrieval=False)

    Example
    -------
    >>> processor = PaganinProcessor()
    >>> processor.set_input(data)
    >>> thickness = processor.get_output(override_parameters={'alpha':10})
    >>> phase_retrieved_image = thickness*processor.mu

    """
   
    def __init__(self, delta = 1, beta = 1e-2, energy = 40000,  filter_type='paganin_method', pad = False):
        
        kwargs = {
            'energy' : energy,
            'wavelength' : self.energy_to_wavelength(energy),
            'delta': delta,
            'beta': beta,
            'delta_user' : delta,
            'beta_user' : beta,
            'filter_type' : filter_type,
            'mu' : None,
            'alpha' : None,
            'pixel_size' : None,
            'propagation_distance' : None,
            'magnification' : None,
            'filter' : None,
            'full_retrieval' : True,
            'pad' : pad,
            'override_geometry' : None,
            'override_parameters' : None
            }
        
        super(PaganinProcessor, self).__init__(**kwargs)

    def check_input(self, data):

        
        return True

    def check_geometry(self, geometry, geometry_override):

        unit_list = ['m','cm','mm','um']
        unit_multipliers = [1, 1e-2, 1e-3, 1e-6]
        if geometry.config.units in unit_list:
            unit_multiplier = unit_multipliers[unit_list.index(geometry.config.units)]
        else:
            unit_multiplier = None
        
        parameters = ['magnification', 'propagation_distance', 'pixel_size']
        
        if geometry_override is None:
            geometry = {}

        # get and check parameters from over-ride geometry dictionary
        for key in geometry_override.keys():
            if key not in parameters:
                raise ValueError('Parameter {} not recognised, expected one of {}.'.format(key, parameters))
            elif geometry_override[parameter] is None | geometry_override[parameter] == 0:
                raise ValueError("Parameter {} cannot be {}, please update data.geometry.{} or over-ride with processor.get_output(override_geometry={'{}':value})")\
                    .format(parameter, str(getattr(self, parameter)), geometry_parameters[i], parameter)    
            else:
                self.__setattr__(key, self.override_geometry[key])

        # specify parameter names as defined in geometry
        geometry_parameters = ['magnification', 'dist_center_detector', ('pixel_size_h', 'pixel_size_v')]
        # specify if parameter requires unit conversion
        convert_units = [False, True, True]

        # get and check parameters from geometry if they are not in the over-ride geometry dictionary
        for i, parameter in enumerate(parameters):
            if parameter not in geometry_override:
                if type(geometry_parameters[i])==tuple:
                    param1 = getattr(geometry, geometry_parameters[i][0])
                    param2 = getattr(geometry, geometry_parameters[i][1])
                    if (param1 - param2) / (param1 + param2) >= 1e-5:
                        raise ValueError("Parameter {} is not homogeneous up to 1e-5: got {} and {}, please update geometry using data.geometry.{} and data.geometry.{}\
                                            or over-ride with processor.get_output(override_geometry={'{}':value})"\
                                            .format(parameter, str(param1), str(param2), geometry_parameters[i][0], geometry_parameters[i][1], parameter))
                else:
                    param1 = getattr(geometry, geometry_parameters[i])
                
                if param1 is None | param1 == 0:
                    raise ValueError("Parameter {} cannot be {}, please update data.geometry.{} or over-ride with processor.get_output(override_geometry={'{}':value})")\
                    .format(parameter, str(getattr(self, parameter)), geometry_parameters[i], parameter)
                else:
                    if unit_multipliers[i]:
                        if convert_units:
                            if unit_multiplier == None:
                                raise ValueError("Geometry units {} not recognised, expected one of {}").format(str(geometry.config.units),str(unit_list))
                            else:
                                self.__setattr__(parameter, param1)*unit_multiplier
                        else:
                            self.__setattr__(parameter, param1)
        
    def create_filter(self, Nx, Ny):
        '''
        Function to create the Paganin filter, either using the paganin [1] or generalised paganin [2] method
        The filter is created on a mesh in Fourier space kx, ky
        [1] https://doi.org/10.1046/j.1365-2818.2002.01010.x
        [2] https://iopscience.iop.org/article/10.1088/2040-8986/abbab9 
        '''

        if ('alpha' in self.override_parameters) & ('delta' in self.override_parameters):
            raise log.warning('Because you specified alpha, it will not be calculated and therefore delta will be ignored')
        else:
            if ('delta' in self.override_parameters):
                self.delta = self.override_parameters['delta']
            else:
                self.delta = self.delta_user
        
        if ('beta' in self.override_parameters):
            self.beta = self.override_parameters['beta']
        else:
            self.beta = self.beta_user

        self.__calculate_mu()

        if ('alpha' in self.override_parameters):
            self.alpha = self.override_parameters['alpha']
        else:
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
        self.check_geometry(data.geometry, self.override_geometry)

        if out is None:
            out = data.geometry.allocate(None)

        if self.pad>0:
            try:
                Padder.edge(pad_width={'horizontal': self.pad, 'vertical':self.pad})(data)
                Padder.edge(pad_width={'horizontal': self.pad, 'vertical':self.pad})(out)
            except:
                Padder.edge(pad_width={'horizontal': self.pad})(data)
                Padder.edge(pad_width={'horizontal': self.pad})(out)

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
        
        if self.pad>0:
            try:
                Slicer(roi={'horizontal': (self.pad,out.get_dimension_size('horizontal')-self.pad), 'vertical':(self.pad,out.get_dimension_size('vertical')-self.pad)})(out)
            except:
                Slicer(roi={'horizontal': (self.pad,out.get_dimension_size('horizontal')-self.pad)})(out)

        return out
        
    def get_output(self, out=None, full_retrieval=True, override_geometry=None, override_parameters=None):
        '''
        Function to get output from the PaganinProcessor
        
        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data

        full_retrieval : bool, optional
            If True, perform the full phase retrieval and return the thickness. If False, return a filtered image, default is True

        override_geometry: dict, optional
            Geometry parameters to use in the phase retrieval if you want to over-ride values found in data.geometry. Specify parameters as {'parameter':value}\
            where parameter is 'magnification', 'propagation_distance' or 'pixel_size' and value is the new value to use. Specify distance parameters in units of m.

        override_parameters: dict, optional
            Over-ride the parameters to use in the phase retrieval. Specify parameters as {'parameter':value} where parameter is 'delta', 'beta' or 'alpha' and value is the new value to use.\
            If 'alpha' is specified the new value will be used, delta will be ignored but beta will still be used to calculate mu = 4.0*np.pi*beta/wavelength which is used for scaling the thickness,\
            therefore it is only recommended to specify alpha when also using get_output(full_retrieval=False), or re-scaling the result by mu e.g. thickness*processor.mu\
            If alpha is not specified, it will be calculated = (propagation_distance*delta*wavelength)/(4.0*np.pi*beta)

        Returns
        -------
        Processor
            AcquisitionData corrected for phase effects, retrieved sample thickness in m or (if get_output(full_retrieval=False)) filtered data 
                    
        Example
        -------
        >>> processor = PaganinProcessor(delta=5, beta=0.05, energy=18000)
        >>> processor.set_input(data)
        >>> thickness = processor.get_output()

        Example
        -------
        >>> processor = PaganinProcessor(delta=1,beta=10e2)
        >>> processor.set_input(data)
        >>> filtered_image = processor.get_output(full_retrieval=False)

        Example
        -------
        >>> processor = PaganinProcessor()
        >>> processor.set_input(data)
        >>> thickness = processor.get_output(override_parameters={'alpha':10})
        >>> phase_retrieved_image = thickness*processor.mu

        '''
        self.override_geometry = override_geometry
        self.override_parameters = override_parameters
        self.full_retrieval = full_retrieval
        
        return super().get_output(out)
    
    def __call__(self, x, out=None, full_retrieval=True, override_geometry=None, override_parameters=None):
        self.override_geometry = override_geometry
        self.override_parameters = override_parameters
        self.full_retrieval = full_retrieval
        return super().__call__(x, out)

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