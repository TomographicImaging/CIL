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
# CIL Developers, listed at: 
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.framework import Processor, AcquisitionData, DataOrder

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
    Processor to retrieve quantitative information from phase contrast images 
    using the Paganin phase retrieval algorithm described in [1]
    
    Parameters
    ----------
    delta: float (optional)
        Real part of the deviation of the material refractive index from 1, 
        where refractive index :math:`n = (1 - \delta) + i \beta` energy-
        dependent refractive index information for x-ray wavelengths can be 
        found at [2], default is 1
    
    beta: float (optional)
        Complex part of the material refractive index, where refractive index 
        :math:`n = (1 - \delta) + i \beta` energy-dependent refractive index 
        information for x-ray wavelengths can be found at [2], default is 1e-2
    
    energy: float (optional)
        Energy of the incident photon, default is 40000

    energy_units: string (optional)
        Energy units, default is 'eV'
    
    full_retrieval : bool, optional
        If True, perform the full phase retrieval and return the thickness. If 
        False, return a filtered image, default is True

    filter_type: string (optional)
        The form of the Paganin filter to use, either 'paganin_method' 
        (default) or 'generalised_paganin_method' as described in [3] 

    pad: int (optional)
        Number of pixels to pad the image in Fourier space to reduce aliasing, 
        default is 0 

    return_units: string (optional)
        The distance units to return the sample thickness in, must be one of 
        'm', 'cm', 'mm' or 'um'. Only applies if full_retrieval=True (default 
        is'cm')

    Returns
    -------
    AcquisitionData
        AcquisitionData corrected for phase effects, retrieved sample thickness 
        or (if :code:`full_retrieval=False`) filtered data 
                
    Example
    -------
    >>> processor = PaganinProcessor(delta=5, beta=0.05, energy=18000)
    >>> processor.set_input(data)
    >>> thickness = processor.get_output()

    Example
    -------
    >>> processor = PaganinProcessor(delta=1,beta=10e2, full_retrieval=False)
    >>> processor.set_input(data)
    >>> filtered_image = processor.get_output()

    Example
    -------
    >>> processor = PaganinProcessor()
    >>> processor.set_input(data)
    >>> thickness = processor.get_output(override_filter={'alpha':10})
    >>> phase_retrieved_image = thickness*processor.mu

    Notes
    -----
    This processor will work most efficiently using the cil data order with
    `data.reorder('cil')`
    
    Notes
    -----
    This processor uses the phase retrieval algorithm described by Paganin et 
    al. [1] to retrieve the sample thickness
    
    .. math:: T(x,y) = - \frac{1}{\mu}\ln\left (\mathcal{F}^{-1}\left 
        (\frac{\mathcal{F}\left ( M^2I_{norm}(x, y,z = \Delta) \right )}{1 + 
          \alpha\left ( k_x^2 + k_y^2 \right )}  \right )\right ),
    
    where

        - :math:`T`, is the sample thickness,
        - :math:`\mu = \frac{4\pi\beta}{\lambda}` is the material linear 
        attenuation coefficient where :math:`\beta` is the complex part of the 
        material refractive index and :math:`\lambda=\frac{hc}{E}` is the probe 
        wavelength,
        - :math:`M` is the magnification at the detector,
        - :math:`I_{norm}` is the input image which is expected to be the 
        normalised transmission data, 
        - :math:`\Delta` is the propagation distance,
        - :math:`\alpha = \frac{\Delta\delta}{\mu}` is a parameter determining 
        the strength of the filter to be applied in Fourier space where 
        :math:`\delta` is the real part of the deviation of the material 
        refractive index from 1 
        - :math:`k_x, k_y = \left ( \frac{2\pi p}{N_xW}, \frac{2\pi q}{N_yW} 
        \right )` where :math:`p` and :math:`q` are co-ordinates in a Fourier 
        mesh in the range :math:`-N_x/2` to :math:`N_x/2` for an image with 
        size :math:`N_x, N_y` and pixel size :math:`W`.
    
    A generalised form of the Paganin phase retrieval method can be called 
    using :code:`filter_type='generalised_paganin_method'`, which uses the 
    form of the algorithm described in [2]
    
    .. math:: T(x,y) = -\frac{1}{\mu}\ln\left (\mathcal{F}^{-1}\left (\frac{
        \mathcal{F}\left ( M^2I_{norm}(x, y,z = \Delta) \right )}{1 - \frac{2
        \alpha}{W^2}\left ( \cos(Wk_x) + \cos(Wk_y) -2 \right )}  \right )
        \right )
    
    The phase retrieval is valid under the following assumptions

        - used with paraxial propagation-induced phase contrast images which 
        can be assumed to be single-material locally
        - using intensity data which has been flat field corrected
        - and under the assumption that the Fresnel number 
        :math:`F_N = W^2/(\lambda\Delta) >> 1`
    
    To apply a filter to images using the Paganin method, call 
    :code:`full_retrieval=False`. In this case the pre-scaling and conversion 
    to absorption is not applied so the requirement to supply flat field 
    corrected intensity data is relaxed,
    
    .. math:: I_{filt} = \mathcal{F}^{-1}\left (\frac{\mathcal{F}\left ( 
        I(x, y,z = \Delta) \right )}
        {1 - \alpha\left ( k_x^2 + k_y^2 \right )}  \right )

    References
    ---------
    - [1] https://doi.org/10.1046/j.1365-2818.2002.01010.x 
    - [2] https://henke.lbl.gov/optical_constants/getdb2.html
    - [3] https://iopscience.iop.org/article/10.1088/2040-8986/abbab9
    With thanks to colleagues at DTU for help with the initial implementation 
    of the phase retrieval algorithm

    """

    def __init__(self, delta=1, beta=1e-2, energy=40000,
                 energy_units='eV',  full_retrieval=True, 
                 filter_type='paganin_method', pad=0, 
                 return_units='cm'):
        
        kwargs = {
            'energy' : energy,
            'wavelength' : self._energy_to_wavelength(energy, energy_units,
                                                      return_units),
            'delta': delta,
            'beta': beta,
            '_delta_user' : delta,
            '_beta_user' : beta,
            'filter_Nx' : None,
            'filter_Ny' : None,
            'filter_type' : filter_type,
            'mu' : None,
            'alpha' : None,
            'pixel_size' : None,
            'propagation_distance' : None,
            'magnification' : None,
            'filter' : None,
            'full_retrieval' : full_retrieval,
            'pad' : pad,
            'override_geometry' : None,
            'override_filter' : None,
            'return_units' : return_units
            }
        
        super(PaganinProcessor, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(data, (AcquisitionData)):
            raise TypeError('Processor only supports AcquisitionData')
    
        return True
        
    def process(self, out=None):

        data  = self.get_input()
        cil_order = tuple(DataOrder.get_order_for_engine('cil', data.geometry))
        if data.dimension_labels != cil_order:
            log.warning(msg="This processor will work most efficiently using\
                        \nCIL data order, consider using `data.reorder('cil')`")

        # set the geometry parameters to use from data.geometry unless the 
        # geometry is overridden with an override_geometry
        self._set_geometry(data.geometry, self.override_geometry)

        if out is None:
            out = data.geometry.allocate(None)

        # make slice indices to get the projection
        slice_proj = [slice(None)]*len(data.shape)
        angle_axis = data.get_dimension_axis('angle')
        slice_proj[angle_axis] = 0
        
        if data.geometry.channels>1:
            channel_axis = data.get_dimension_axis('channel')
            slice_proj[channel_axis] = 0
        else:
            channel_axis = None

        data_proj = data.as_array()[tuple(slice_proj)]

        # create an empty axis if the data is 2D
        if len(data.shape) == 2:
            data.array = np.expand_dims(data.array, len(data.shape))
            slice_proj.append(slice(None))
            data_proj = data.as_array()[tuple(slice_proj)]
            
        elif len(data_proj.shape) == 2:
            pass
        else:
            raise(ValueError('Data must be 2D or 3D per channel'))
        
        if len(out.shape) == 2:
            out.array = np.expand_dims(out.array, len(out.shape))
        
        # create a filter based on the shape of the data
        filter_shape = np.shape(data_proj)
        self.filter_Nx = filter_shape[0]+self.pad*2
        self.filter_Ny = filter_shape[1]+self.pad*2
        self._create_filter(self.override_filter)
        
        # pre-calculate the scaling factor
        scaling_factor = -(1/self.mu)

        # allocate padded buffer
        padded_buffer = np.zeros(tuple(x+self.pad*2 for x in data_proj.shape))
        
        # make slice indices to unpad the data
        if self.pad>0:
            slice_pad = tuple([slice(self.pad,-self.pad)]
                                *len(padded_buffer.shape))
        else:
            slice_pad = tuple([slice(None)]*len(padded_buffer.shape))
        # loop over the channels
        for j in range(data.geometry.channels):
            if channel_axis is not None:
                slice_proj[channel_axis] = j
            # loop over the projections
            for i in tqdm(range(len(data.geometry.angles))):
                
                slice_proj[angle_axis] = i
                padded_buffer[slice_pad] = data.array[(tuple(slice_proj))]
                
                if self.full_retrieval==True:
                    # apply the filter in fourier space, apply log and scale 
                    # by magnification
                    fI = fft2(self.magnification**2*padded_buffer)
                    iffI = ifft2(fI*self.filter)
                    # apply scaling factor
                    padded_buffer = scaling_factor*np.log(iffI)
                else:
                    # apply the filter in fourier space
                    fI = fft2(padded_buffer)
                    padded_buffer = ifft2(fI*self.filter)
                if data.geometry.channels>1:
                    out.fill(padded_buffer[slice_pad], angle = i, 
                             channel=j)
                else:
                    x = padded_buffer[slice_pad]
                    out.fill(x, angle = i)
                    
        data.array = np.squeeze(data.array)
        out.array = np.squeeze(out.array)
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
        
    def get_output(self, out=None, override_geometry=None, 
                   override_filter=None):
        r'''
        Function to get output from the PaganinProcessor
        
        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data

        override_geometry: dict, optional
            Geometry parameters to use in the phase retrieval if you want to 
            over-ride values found in `data.geometry`. Specify parameters as a 
            dictionary :code:`{'parameter':value}` where parameter is 
            :code:`'magnification', 'propagation_distance'` or 
            :code:`'pixel_size'` and value is the new value to use. Specify 
            distance parameters in the same units as :code:`return_units` 
            (default is cm).

        override_filter: dict, optional
            Over-ride the filter parameters to use in the phase retrieval. 
            Specify parameters as :code:`{'parameter':value}` where parameter 
            is :code:`'delta', 'beta'` or :code:`'alpha'` and value is the new 
            value to use.

        Returns
        -------
        AcquisitionData
            AcquisitionData corrected for phase effects, retrieved sample 
            thickness or (if :code:`full_retrieval=False`) filtered data 
                    
        Example
        -------
        >>> processor = PaganinProcessor(delta=5, beta=0.05, energy=18000)
        >>> processor.set_input(data)
        >>> thickness = processor.get_output()

        Example
        -------
        >>> processor = PaganinProcessor(delta=1,beta=10e2, 
        full_retrieval=False)
        >>> processor.set_input(data)
        >>> filtered_image = processor.get_output()

        Example
        -------
        >>> processor = PaganinProcessor()
        >>> processor.set_input(data)
        >>> thickness = processor.get_output(override_filter={'alpha':10})
        >>> phase_retrieved_image = thickness*processor.mu

        Notes
        -----
        If :code:`'alpha'` is specified in override_filter the new value will 
        be used and delta will be ignored but beta will still be used to 
        calculate :math:`\mu = \frac{4\pi\beta}{\lambda}` which is used for 
        scaling the thickness, therefore it is only recommended to specify 
        alpha when also using :code:`get_output(full_retrieval=False)`, or 
        re-scaling the result by :math:`\mu` e.g. 
        :code:`thickness*processor.mu` If :code:`alpha` is not specified, 
        it will be calculated :math:`\frac{\Delta\delta\lambda}{4\pi\beta}`

        '''
        self.override_geometry = override_geometry
        self.override_filter = override_filter
        
        return super().get_output(out)
    
    def __call__(self, x, out=None, override_geometry=None, 
                 override_filter=None):
        self.set_input(x)

        if out is None:
            out = self.get_output(override_geometry=override_geometry, 
                                  override_filter=override_filter)
        else:
            self.get_output(out=out, override_geometry=override_geometry, 
                            override_filter=override_filter)

        return out

    def _set_geometry(self, geometry, override_geometry=None):
        '''
        Function to set the geometry parameters for the processor. Values are 
        from the data geometry unless the geometry is overridden with an 
        override_geometry dictionary.
        '''
        
        parameters = ['magnification', 'propagation_distance', 'pixel_size']
        # specify parameter names as defined in geometry
        geometry_parameters = ['magnification', 'dist_center_detector', 
                               ('pixel_size_h', 'pixel_size_v')]
        # specify if parameter requires unit conversion
        convert_units = [False, True, True]
        
        if override_geometry is None:
            override_geometry = {}

        # get and check parameters from over-ride geometry dictionary
        for parameter in override_geometry.keys():
            if parameter not in parameters:
                raise ValueError('Parameter {} not recognised, expected one of\
                                 {}.'.format(parameter, parameters))
            elif (override_geometry[parameter] is None) \
                | (override_geometry[parameter] == 0):
                raise ValueError("Parameter {} cannot be {}, please update \
                                 data.geometry.{} or over-ride with \
                                 processor.get_output(override_geometry= \
                                 {{ '{}' : value }} )"\
                    .format(parameter, str(getattr(self, parameter)), 
                            geometry_parameters[i], parameter))
            else:
                self.__setattr__(parameter, override_geometry[parameter])


        # get and check parameters from geometry if they are not in the 
        # over-ride geometry dictionary
        for i, parameter in enumerate(parameters):
            if parameter not in override_geometry:
                if type(geometry_parameters[i])==tuple:
                    param1 = getattr(geometry, geometry_parameters[i][0])
                    param2 = getattr(geometry, geometry_parameters[i][1])
                    if (param1 - param2) / (param1 + param2) >= 1e-5:
                        raise ValueError("Parameter {} is not homogeneous up \
                                         to 1e-5: got {} and {}, please update\
                                          geometry using data.geometry.{} and \
                                         data.geometry.{} or over-ride with \
                                         processor.get_output(\
                                         override_geometry={{ '{}' : value }})"
                                         .format(parameter, str(param1), 
                                                 str(param2), 
                                                 geometry_parameters[i][0], 
                                                 geometry_parameters[i][1], 
                                                 parameter))
                else:
                    param1 = getattr(geometry, geometry_parameters[i])
                
                if (param1 is None) | (param1 == 0):
                    raise ValueError("Parameter {} cannot be {}, please update\
                                      data.geometry.{} or over-ride with \
                                     processor.get_output(override_geometry\
                                     ={{ '{}' : value }} )"
                                     .format(parameter, str(param1), 
                                             str(geometry_parameters[i]),
                                             parameter))
                else:
                    if convert_units[i]:
                        param1 = self._convert_units(param1, 'distance',
                                                       geometry.config.units, 
                                                       self.return_units)
                    self.__setattr__(parameter, param1)

        
    def _create_filter(self, override_filter=None):
        '''
        Function to create the Paganin filter, either using the paganin [1] or 
        generalised paganin [2] method
        The filter is created on a mesh in Fourier space kx, ky
        [1] https://doi.org/10.1046/j.1365-2818.2002.01010.x
        [2] https://iopscience.iop.org/article/10.1088/2040-8986/abbab9 
        '''
        if override_filter is None:
            override_filter = {}

        # update any parameter which has been over-ridden with override_filter
        if ('alpha' in override_filter) & ('delta' in override_filter):
            log.warning(msg="Because you specified alpha, it will not be \
                        calculated and therefore delta will be ignored")

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
            
        # create the Fourier mesh
        kx,ky = np.meshgrid( 
            np.arange(-self.filter_Nx/2, self.filter_Nx/2, 1, dtype=np.float64) 
            * (2*np.pi)/(self.filter_Nx*self.pixel_size),
            np.arange(-self.filter_Ny/2, self.filter_Ny/2, 1, dtype=np.float64) 
            * (2*np.pi)/(self.filter_Ny*self.pixel_size),
            sparse=False, 
            indexing='ij'
            )
        
        # create the filter using either paganin or generalised paganin method
        if self.filter_type == 'paganin_method':
            self.filter =  ifftshift(1/(1. + self.alpha*(kx**2 + ky**2)))
        elif self.filter_type == 'generalised_paganin_method':       
            self.filter =  ifftshift(1/(1. - (2*self.alpha/self.pixel_size**2)
                                        *(np.cos(self.pixel_size*kx) 
                                          + np.cos(self.pixel_size*ky) -2)))
        else:
            raise ValueError("filter_type not recognised: got {0} expected one\
                              of 'paganin_method' or \
                             'generalised_paganin_method'"
                             .format(self.filter_type))
        
    def _calculate_mu(self):
        '''
        Function to calculate the linear attenutation coefficient mu
        '''
        self.mu = 4.0*np.pi*self.beta/self.wavelength

    def _calculate_alpha(self):
        '''
        Function to calculate alpha, a constant defining the Paganin filter 
        strength
        '''
        self.alpha = self.propagation_distance*self.delta/self.mu
    
    def _energy_to_wavelength(self, energy, energy_units, return_units):
        '''
        Function to convert photon energy in eV to wavelength in return_units
        
        Parameters
        ----------
        energy: float
            Photon energy
        
        energy_units
            Energy units

        return_units
            Distance units in which to return the wavelength
        
        Returns
        -------
        float
            Photon wavelength in return_units
        '''
        top = self._convert_units(constants.h*constants.speed_of_light, 
                                    'distance', 'm', return_units)
        bottom = self._convert_units(energy, 'energy', energy_units, 'J')

        return top/bottom
    
    def _convert_units(self, value, unit_type, input_unit, output_unit):
        unit_types = ['distance','energy','angle']

        if unit_type == unit_types[0]:
            unit_list = ['m','cm','mm','um']
            unit_multipliers = [1.0, 1e-2, 1e-3, 1e-6]
        elif unit_type == unit_types[1]:
            unit_list = ['meV', 'eV', 'keV', 'MeV', 'J']
            unit_multipliers = [1e-3, 1, 1e3, 1e6, 1/constants.eV]
        elif unit_type == unit_types[2]:
            unit_list = ['deg', 'rad']
            unit_multipliers = [1, np.rad2deg(1)]
        else:
            raise ValueError("Unit type '{}' not recognised, must be one of {}"
                            .format(unit_type, unit_types))

        for x in [input_unit, output_unit]:
            if x not in unit_list:
                raise ValueError("Unit '{}' not recognised, must be one of {}.\
                                 \nGeometry units can be updated using geometry.config.units"
                                 .format(x, unit_list))
            
        return value*unit_multipliers[unit_list.index(input_unit)]\
            /unit_multipliers[unit_list.index(output_unit)]