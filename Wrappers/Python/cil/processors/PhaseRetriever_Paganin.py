from cil.framework import Processor, DataContainer
import numpy as np

from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftshift
from scipy import constants

from tqdm import tqdm

import logging
import dask
from dask import delayed

class PaganinPhaseRetrieval(Processor):

    """docstring for Paganin"""

    def __init__(self, energy = 40000, delta = 1, beta = 1e-3, unit_multiplier = 1, propagation_distance=None):

        """
        Method to retrieve phase information from  using the Paganin phase retrieval algorithm
        described in https://doi.org/10.1046/j.1365-2818.2002.01010.x 
    

        Parameters
        ----------
        energy_eV: float
            Energy of the incident photon in eV
            
        delta: float
            Real part of the deviation of the material refractive index from 1, where refractive index n = (1 - delta) + i beta 
            energy-dependent refractive index information can be found at https://refractiveindex.info/
        
        beta: float
            Complex part of the material refractive index, where refractive index n = (1 - delta) + i beta
            energy-dependent refractive index information can be found at https://refractiveindex.info/ 

        unit_multiplier: float
            Multiplier to convert units stored in geometry to metres, conversion applies to pixel size and propagation distance

        propagation_distance: float (optional)
            The sample to detector distance in meters. If not specified, the value in data.geometry.dist_center_detector will be used

        Returns
        -------
        DataContainer
            returns the data corrected for the effects of phase difference in the material. 
        
        """

        kwargs = {
                    'energy' : energy,
                    'wavelength' : self.energy_to_wavelength(energy),
                    'delta': delta,
                    'beta': beta,
                    'unit_multiplier' : unit_multiplier,
                    'propagation_distance_user' : propagation_distance,
                    'output_type' : None,
                    'mu' : None,
                    'alpha' : None,
                    'pixel_size' : None,
                    'propagation_distance' : None,
                    'magnification' : None,
                    'filter' : None
                 }
        
        super(PaganinPhaseRetrieval, self).__init__(**kwargs)
        
        self.__calculate_mu()

    def __calculate_mu(self):
        """
        Function to calculate the linear attenutation coefficient mu
        """
        self.mu = 4.0*np.pi*self.beta/self.wavelength   

    def __calculate_alpha(self):
        self.alpha = (self.propagation_distance)*self.delta/self.mu
        
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

        if self.propagation_distance_user is None: 
            if data.geometry.dist_center_detector is None:
                raise ValueError('Propagation distance not found, please provide propagation_distance as an argument to PaganinPhaseRetriever or update geometry.dist_center_detector')
            elif data.geometry.dist_center_detector == 0:
                raise ValueError('Found geometry.dist_center_detector = 0, phase retrieval is not compatible with virtual magnification\
                                 please provide a real propagation_distance as an argument to PaganinPhaseRetriever or update geometry.dist_center_detector')
            else:
                propagation_distance = data.geometry.dist_center_detector
                self.propagation_distance = (propagation_distance)*self.unit_multiplier
        else:
            self.propagation_distance = self.propagation_distance_user

        self.__calculate_alpha()

        return True

    def get_output(self, out=None, output_type = 'attenuation'):
        '''
        Runs the configured processor and returns the processed data

        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return
        
        output_type: string, optional
            if 'attenuation' (default), returns the projected attenuation of the sample corrected for phase effects, attenuation = µT 
            if 'thickness', returns the projected thickness T of the sample projected onto the image plane 
            if 'phase', returns the phase of the beam at the material exit, phase ϕ(r⊥) = −δ T(r⊥) · 2π/λ
        
        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed

        '''
        self.output_type = output_type
        return super().get_output(out)
    
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

    def process(self, out=None):
        
        data  = self.get_input()

        self.create_filter(data.get_slice(angle=0).as_array())

        must_return = False        
        if out is None:
            out = data.geometry.allocate(None)
            must_return = True

        # set up delayed computation and
        # compute in parallel processes
        i = 0
        max_proc = len(data.geometry.angles)
        num_parallel_processes = 6
        # tqdm progress bar on the while loop
        with tqdm(total=max_proc) as pbar:
            while (i < max_proc):
                j = 0
                while j < num_parallel_processes:
                    if j + i == max_proc:
                        break
                    j += 1
                # set up j delayed computation
                procs = []
                for idx in range(j):
                    projection = data.get_slice(angle=i+idx).as_array()
                    procs.append(delayed(self.process_projection)(projection))
                # call compute on j (< num_parallel_processes) processes concurrently
                # this limits the amount of memory required to store the output of the 
                # phase retrieval to j projections.
                res = dask.compute(*procs[:j])
                
                # copy the output in the output buffer
                for k, el in enumerate(res):
                    out.fill(el, angle=i+k)
                    pbar.update(1)
                i += j
            
        if must_return:
            return out
        
    def process_projection(self, image):
        
        iffI = self.paganin_filter(image)
        if self.output_type == 'filtered_image':
            return iffI
        if self.output_type == 'attenuation':
            return -np.log(iffI)
        elif self.output_type == 'thickness':
            return 
        elif self.output_type == 'phase':
            return (self.delta*2*np.pi/self.wavelength)*np.log(iffI)/self.mu
        else:
            raise ValueError("output_type not recognised: got {0} expected one of 'attenuation', 'thickness' or 'phase'"\
                            .format(self.output_type))
    
    def filter_image(self, out=None):
        self.output_type = 'filtered_image' 
        return self.process(out=out)

    def create_filter(self, image):
        
        Nx, Ny = image.shape

        kx,ky = np.meshgrid( 
            np.arange(-Nx/2, Nx/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            np.arange(-Ny/2, Ny/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            sparse=False, 
            indexing='ij'
            )
        
        kW = np.abs(kx.max()*self.pixel_size)       
        if (kW >= 1): 
            logging.warning("This algorithm is valid for k*W << 1, found np.abs(kx.max()*self.pixel_size) = {}, results may not be accurate".format(kW))

        self.filter =  (1. + self.alpha*(kx**2 + ky**2)/self.magnification)
    
    def paganin_filter(self, image):
        
        fI = fftshift(
            fft2(self.magnification**2*image)
            )
        
        iffI = ifft2(fftshift(fI/self.filter))

        return iffI
    

class GeneralisedPhaseRetrieval(PaganinPhaseRetrieval):
    
    def create_filter(self, image):
        
        Nx, Ny = image.shape

        kx,ky = np.meshgrid( 
            np.arange(-Nx/2, Nx/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            np.arange(-Ny/2, Ny/2, 1, dtype=np.float64) * (2*np.pi)/(Nx*self.pixel_size),
            sparse=False, 
            indexing='ij'
            )
        
        kW = np.abs(kx.max()*self.pixel_size)       
        if (kW > np.pi): 
            logging.warning("This algorithm is valid for |k*W| <= pi, found np.abs(kx.max()*self.pixel_size) = {}, results may not be accurate".format(kW))

        self.filter =  (1. - (2*self.alpha/self.pixel_size**2)*(np.cos(self.pixel_size*kx) + np.cos(self.pixel_size*ky) -2)/self.magnification)

class OriginalPhaseRetrieval(PaganinPhaseRetrieval):
    def __calculate_alpha(self):
        self.alpha = 4.0*(np.pi**2)*self.propagation_distance*self.delta/self.mu

    def check_input(self, dataset):
        geometry = dataset.geometry
        if (geometry.pixel_size_h - geometry.pixel_size_v ) / \
            (geometry.pixel_size_h + geometry.pixel_size_v ) < 1e-5:
            self.pixel_size = (geometry.pixel_size_h / geometry.magnification)* self.unit_multiplier
        else:
            raise ValueError('Panel pixel size is not homogeneous up to 1e-5: got {} {}'\
                    .format( geometry.pixel_size_h, geometry.pixel_size_v )
                )
        
        if geometry.magnification == None:
            self.magnification = 1
        else:
            self.magnification = geometry.magnification
        
        if self.propagation_distance_user is None:
            distance_origin_detector = geometry.dist_center_detector
            distance_source_origin   = geometry.dist_source_center
            z = distance_origin_detector * distance_source_origin/(distance_source_origin + distance_origin_detector)
            self.propagation_distance = z* self.unit_multiplier
        else:
            self.propagation_distance = self.propagation_distance_user
        
        self.__calculate_alpha()
        return True
    
    def create_filter(self, image):
        Nx, Ny = image.shape
        dfx = 1./self.pixel_size/Nx
        dfy = 1./self.pixel_size/Ny

        FX,FY = np.meshgrid( 
            np.arange(-Nx/2, Nx/2, 1, dtype=np.float64) * dfx,
            np.arange(-Ny/2, Ny/2, 1, dtype=np.float64) * dfy,
            sparse=False, 
            indexing='ij'
            )

        self.filter = 1. + self.alpha*(FX**2 + FY**2)/self.magnification 

    def retrieve_phase(self, Image):
        
        I = Image    
        fI = fftshift(
                        fft2(self.magnification**2*I)
                    )

        tmp = fI / self.filter 
        tmp = fftshift(tmp)
        Txy = ifft2(tmp)
        np.abs(Txy, out=Txy)
        np.log(Txy, out=Txy)
        Txy *= -1/self.mu

        if self.output_type == 'attenuation':
            return self.mu*Txy
        elif self.output_type == 'thickness':
            return Txy
        elif self.output_type == 'phase':
            return -self.delta*Txy*2*np.pi/self.wavelength
        else:
            raise ValueError("output_type not recognised: got {0} expected one of 'attenuation', 'thickness' or 'phase'"\
                            .format(self.output_type))