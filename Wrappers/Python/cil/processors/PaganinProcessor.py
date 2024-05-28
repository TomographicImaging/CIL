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

from cil.framework import Processor, AcquisitionData
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

    r"""
    Processor to retrieve quantitative information from phase contrast images using the Paganin phase retrieval algorithm described in [1]
    
    .. math:: T(x,y) = - \frac{1}{\mu}\ln\left (\mathcal{F}^{-1}\left (\frac{\mathcal{F}\left ( M^2I_{norm}(x, y,z = \Delta) \right )}{1 + \alpha\left ( k_x^2 + k_y^2 \right )/M}  \right )\right ),
    
    where

        - :math:`T`, is the sample thickness,
        - :math:`\mu = \frac{4\pi\beta}{\lambda}` is the material linear attenuation coefficient where :math:`\beta` is the complex part of the material refractive index and :math:`\lambda=\frac{hc}{E}` is the probe wavelength,
        - :math:`M` is the magnification at the detector,
        - :math:`I_{norm}` is the input image which is expected to be the normalised transmission data, 
        - :math:`\Delta` is the propagation distance,
        - :math:`\alpha = \frac{\Delta\delta}{\mu}` is a parameter determining the strength of the filter to be applied in Fourier space where :math:`\delta` is the real part of the deviation of the material refractive index from 1 
        - :math:`k_x, k_y = \left ( \frac{2\pi p}{N_xW}, \frac{2\pi q}{N_yW} \right )` where :math:`p` and :math:`q` are co-ordinates in a Fourier mesh in the range :math:`-N_x/2` to :math:`N_x/2` for an image with size :math:`N_x, N_y` and pixel size :math:`W`.
    
    A generalised form of the Paganin phase retrieval method can be called using :code:`filter_type='generalised_paganin_method'`, which uses the form of the algorithm described in [2]
    
    .. math:: T(x,y) = -\frac{1}{\mu}\ln\left (\mathcal{F}^{-1}\left (\frac{\mathcal{F}\left ( M^2I_{norm}(x, y,z = \Delta) \right )}{1 - \frac{2\alpha}{W^2}\left ( \cos(Wk_x) + \cos(Wk_y) -2 \right )/M}  \right )\right )
    
    The phase retrieval is valid under the following assumptions

        - it's used with paraxial propagation-induced phase contrast images on single-material samples
        - using intensity data which has been flat field corrected
        - and under the assumption that the Fresnel number :math:`F_N = W^2/(\lambda\Delta) >> 1`
    
    To apply a filter to images using the Paganin method, call :code:`get_output(full_retrieval=False)`. In this case the pre-scaling and conversion to absorption is not applied so 
    the requirement to supply flat field corrected intensity data is relaxed,
    
    .. math:: I_{filt} = \mathcal{F}^{-1}\left (\frac{\mathcal{F}\left ( I(x, y,z = \Delta) \right )}{1 - \alpha\left ( k_x^2 + k_y^2 \right )}  \right )
    
    Parameters
    ----------
    delta: float (optional)
        Real part of the deviation of the material refractive index from 1, where refractive index :math:`n = (1 - \delta) + i \beta` \
        energy-dependent refractive index information for x-ray wavelengths can be found at [3], default is 1
    
    beta: float (optional)
        Complex part of the material refractive index, where refractive index :math:`n = (1 - \delta) + i \beta` \
        energy-dependent refractive index information for x-ray wavelengths can be found at [3], default is 1e-2
    
    energy: float (optional)
        Energy of the incident photon in eV, default is 40000

    filter_type: string (optional)
        The form of the Paganin filter to use, either 'paganin_method' (default) or 'generalised_paganin_method' as described in [2] 

    pad: int (optional)
        Number of pixels to pad the image in Fourier space to reduce aliasing, default is 0 

    Returns
    -------
    AcquisitionData
        AcquisitionData corrected for phase effects, retrieved sample thickness in m or (if :code:`get_output(full_retrieval=False)`) filtered data 
                
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
    >>> thickness = processor.get_output(override_filter={'alpha':10})
    >>> phase_retrieved_image = thickness*processor.mu

    References
    ---------
    - [1] https://doi.org/10.1046/j.1365-2818.2002.01010.x 
    - [2] https://iopscience.iop.org/article/10.1088/2040-8986/abbab9
    - [3] https://henke.lbl.gov/optical_constants/getdb2.html
    With thanks to Rajmund Mokso for help with the initial implementation of the phase retrieval algorithm

    """
   
    def __init__(self, delta = 1, beta = 1e-2, energy = 40000,  filter_type='paganin_method', pad = 0):
        
        kwargs = {
            'energy' : energy,
            'wavelength' : self._energy_to_wavelength(energy),
            'delta': delta,
            'beta': beta,
            '_delta_user' : delta,
            '_beta_user' : beta,
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
            'override_filter' : None
            }
        
        super(PaganinProcessor, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(data, (AcquisitionData)):
            raise TypeError('Processor only supports AcquisitionData')
      
        return True
        
    def process(self, out=None):

        data  = self.get_input()
        self._set_geometry(data.geometry, self.override_geometry)

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
        self._create_filter(filter_shape[0], filter_shape[1], self.override_filter)
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
    
    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        dataset : AcquisitionData
            The input AcquisitionData
        """
        return super().set_input(dataset)
        
    def get_output(self, out=None, full_retrieval=True, override_geometry=None, override_filter=None):
        r'''
        Function to get output from the PaganinProcessor
        
        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data

        full_retrieval : bool, optional
            If True, perform the full phase retrieval and return the thickness. If False, return a filtered image, default is True

        override_geometry: dict, optional
            Geometry parameters to use in the phase retrieval if you want to over-ride values found in `data.geometry`. Specify parameters as :code:`{'parameter':value}`\
            where parameter is :code:`'magnification', 'propagation_distance'` or :code:`'pixel_size'` and value is the new value to use. Specify distance parameters in units of m.

        override_filter: dict, optional
            Over-ride the filter parameters to use in the phase retrieval. Specify parameters as :code:`{'parameter':value}` where parameter is :code:`'delta', 'beta'` or :code:`'alpha'` and value is the new value to use. \
            If :code:`'alpha'` is specified the new value will be used, delta will be ignored but beta will still be used to calculate :math:`\mu = \frac{4\pi\beta}{\lambda}` which is used for scaling the thickness, \
            therefore it is only recommended to specify alpha when also using `get_output(full_retrieval=False)`, or re-scaling the result by :math:`\mu` e.g. :code:`thickness*processor.mu` \
            If :code:`alpha` is not specified, it will be calculated = :math:`\frac{\Delta\delta\lambda}{4\pi\beta}`

        Returns
        -------
        AcquisitionData
            AcquisitionData corrected for phase effects, retrieved sample thickness in m or (if :code:`get_output(full_retrieval=False)`) filtered data 
                    
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
        >>> thickness = processor.get_output(override_filter={'alpha':10})
        >>> phase_retrieved_image = thickness*processor.mu

        '''
        self.override_geometry = override_geometry
        self.override_filter = override_filter
        self.full_retrieval = full_retrieval
        
        return super().get_output(out)
    
    def __call__(self, x, out=None, full_retrieval=True, override_geometry=None, override_filter=None):
        self.set_input(x)

        if out is None:
            out = self.get_output(full_retrieval=full_retrieval, override_geometry=override_geometry, override_filter=override_filter)
        else:
            self.get_output(out=out, full_retrieval=full_retrieval, override_geometry=override_geometry, override_filter=override_filter)

        return out

    def _set_geometry(self, geometry, override_geometry=None):
        '''
        Function to set the geometry parameters for the processor, from the data geometry unless the geometry is overridden with an override_geometry dictionary.
        '''

        unit_list = ['m','cm','mm','um']
        unit_multipliers = [1, 1e-2, 1e-3, 1e-6]
        if geometry.config.units in unit_list:
            unit_multiplier = unit_multipliers[unit_list.index(geometry.config.units)]
        else:
            unit_multiplier = None
        
        parameters = ['magnification', 'propagation_distance', 'pixel_size']
        
        if override_geometry is None:
            override_geometry = {}

        # get and check parameters from over-ride geometry dictionary
        for parameter in override_geometry.keys():
            if parameter not in parameters:
                raise ValueError('Parameter {} not recognised, expected one of {}.'.format(parameter, parameters))
            elif (override_geometry[parameter] is None) | (override_geometry[parameter] == 0):
                raise ValueError("Parameter {} cannot be {}, please update data.geometry.{} or over-ride with processor.get_output(override_geometry= {{ '{}' : value }} )"\
                    .format(parameter, str(getattr(self, parameter)), geometry_parameters[i], parameter))
            else:
                self.__setattr__(parameter, override_geometry[parameter])

        # specify parameter names as defined in geometry
        geometry_parameters = ['magnification', 'dist_center_detector', ('pixel_size_h', 'pixel_size_v')]
        # specify if parameter requires unit conversion
        convert_units = [False, True, True]

        # get and check parameters from geometry if they are not in the over-ride geometry dictionary
        for i, parameter in enumerate(parameters):
            if parameter not in override_geometry:
                if type(geometry_parameters[i])==tuple:
                    param1 = getattr(geometry, geometry_parameters[i][0])
                    param2 = getattr(geometry, geometry_parameters[i][1])
                    if (param1 - param2) / (param1 + param2) >= 1e-5:
                        raise ValueError("Parameter {} is not homogeneous up to 1e-5: got {} and {}, please update geometry using data.geometry.{} and data.geometry.{}\
                                            or over-ride with processor.get_output(override_geometry={{ '{}' : value }} )"\
                                            .format(parameter, str(param1), str(param2), geometry_parameters[i][0], geometry_parameters[i][1], parameter))
                else:
                    param1 = getattr(geometry, geometry_parameters[i])
                
                if (param1 is None) | (param1 == 0):
                    raise ValueError("Parameter {} cannot be {}, please update data.geometry.{} or over-ride with processor.get_output(override_geometry={{ '{}' : value }} )"\
                    .format(parameter, str(param1) ,str(geometry_parameters[i]), parameter))
                else:
                    if convert_units[i]:
                        if unit_multiplier is None:
                            raise ValueError("Geometry units {} not recognised, expected one of {}, please update data.geometry.config.units".format(str(geometry.config.units),str(unit_list)))
                        else:
                            self.__setattr__(parameter, param1*unit_multiplier)
                    else:
                        self.__setattr__(parameter, param1)
        
    def _create_filter(self, Nx, Ny, override_filter=None):
        '''
        Function to create the Paganin filter, either using the paganin [1] or generalised paganin [2] method
        The filter is created on a mesh in Fourier space kx, ky
        [1] https://doi.org/10.1046/j.1365-2818.2002.01010.x
        [2] https://iopscience.iop.org/article/10.1088/2040-8986/abbab9 
        '''
        if override_filter is None:
            override_filter = {}

        if ('alpha' in override_filter) & ('delta' in override_filter):
            log.warning(msg="Because you specified alpha, it will not be calculated and therefore delta will be ignored")

        if ('delta' in override_filter):
            self.delta = override_filter['delta']
        else:
            self.delta = self._delta_user
        
        if ('beta' in override_filter):
            self.beta = override_filter['beta']
        else:
            self.beta = self._beta_user

        self._calculate_mu()

        if ('alpha' in override_filter):
            self.alpha = override_filter['alpha']
        else:
            self._calculate_alpha()
            
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
        
    def _calculate_mu(self):
        '''
        Function to calculate the linear attenutation coefficient mu
        '''
        self.mu = 4.0*np.pi*self.beta/self.wavelength

    def _calculate_alpha(self):
        '''
        Function to calculate alpha, a constant defining the Paganin filter strength
        '''
        self.alpha = self.propagation_distance*self.delta/self.mu
    
    def _energy_to_wavelength(self, energy):
        '''
        Function to convert photon energy in eV to wavelength in m
        
        Parameters
        ----------
        energy: float
            Photon energy in eV
        
        Returns
        -------
        float
            Photon wavelength in m
        '''
        return (constants.h*constants.speed_of_light)/(energy*constants.electron_volt)