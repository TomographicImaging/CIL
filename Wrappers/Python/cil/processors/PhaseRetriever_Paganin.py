from cil.framework import Processor, DataContainer
import numpy as np

from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftshift
from scipy.fft import ifftshift

from scipy import constants

from tqdm import tqdm
import weakref

from multiprocessing import Pool, TimeoutError
from functools import partial
from cil.utilities.display import show2D

import dask
from dask import delayed

class PaganinPhaseRetrieval(Processor):

    """docstring for Paganin"""

    def __init__(self, energy = 40000, delta = 1, beta = 1e-3, unit_multiplier = 1, normalise = False, units_output=False):

        """
        Method to create a phase retrieval processor using the Paganin phase retrieval algorithm
        described in https://doi.org/10.1046/j.1365-2818.2002.01010.x 
    

        Parameters
        ----------
        energy_eV: float
            Energy of the incident photon in eV
            
        delta: float
            Real part of the material refractive index
        
        beta: float
            Complex part of the material refractive index

        unit_multiplier: float
            Multiplier to convert units stored in geometry to metres, conversion applies to pixel size and propagation distance

        normalise: boolean
            Flag to indicate whether the data should be normalised before applying the filter

        units_output: string
            if 'absorption' (default), returns the projected absorption of the sample corrected for phase effects absorption = µT 
            if 'thickness', returns the projected thickness T of the sample projected onto the image plane 
            if 'phase', returns the phase of the beam at the material exit ϕ(r⊥) = −δ T(r⊥) · 2π/λ

        Returns
        -------
        DataContainer
            returns the absorption corrected data, thickness of phase (dependent on units_output) 
        
        """

        kwargs = {
                    'energy' : energy,
                    'wavelength' : self.energy_to_wavelength(energy),
                    'delta': delta,
                    'beta': beta,
                    'unit_multiplier' : unit_multiplier,
                    'normalise' : normalise,
                    'units_output' : units_output,
                    'mu' : None,
                    'alpha' : None,
                    'pixel_size' : None,
                    'effective_distance' : None,
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
        self.alpha = 4.0*(np.pi**2)*self.effective_distance*self.delta/self.mu

    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        input : DataContainer
            The input DataContainer
        """

        if issubclass(type(dataset), DataContainer):
            if self.check_input(dataset):
                self.__dict__['input'] = weakref.ref(dataset)
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(dataset), DataContainer))
        
       
        self.pixel_size = (dataset.geometry.pixel_size_h / dataset.geometry.magnification)*self.unit_multiplier 

        self.magnification = dataset.geometry.magnification

        if (dataset.geometry.dist_source_center is not None) and (dataset.geometry.dist_center_detector is not None):
            effective_distance = dataset.geometry.dist_center_detector / self.magnification
            self.effective_distance = effective_distance*self.unit_multiplier
        else:
            self.effective_distance = 1
        self.__calculate_alpha()
        
    def check_input(self, data):
        geometry = data.geometry
        if (geometry.pixel_size_h - geometry.pixel_size_v ) / \
            (geometry.pixel_size_h + geometry.pixel_size_v ) < 1e-5:
            pass
        else:
            raise ValueError('Panel pixel size is not homogeneous up to 1e-5: got {} {}'\
                    .format( geometry.pixel_size_h, geometry.pixel_size_v )
                )
        return True
    
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

    def phase_retrieval(self, Image, normalise=False):
        
        if normalise:
            Image /= np.mean(Image)
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
        
        if self.units_output == 'absorption':
            return self.mu*Txy
        elif self.units_output == 'thickness':
            return(Txy)
        elif self.units_output == 'phase':
            return -self.delta*Txy*2*np.pi/self.wavelength
        else:
            raise ValueError("units_output not recognised: got {0} expected one of 'absorption', 'thickness' or 'phase'"\
                            .format(self.units_output))

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
                    procs.append(delayed(self.phase_retrieval)(projection, normalise=self.normalise))
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