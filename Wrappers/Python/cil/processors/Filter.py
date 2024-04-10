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

from tqdm import tqdm
from scipy.fft import fft2
from scipy.fft import ifft2

class Filter(Processor):

    def process(self, out=None):

        data = self.get_input()

        self.create_filter(data.get_dimension_size('vertical'), data.get_dimension_size('horizontal'))

        must_return = False        
        if out is None:
            out = data.geometry.allocate(None)
            must_return = True
        
        for i in tqdm(range(len(data.geometry.angles))):
            projection = data.get_slice(angle=i).as_array()
            iffI = ifft2(fft2(projection)*self.filter)
            out.fill(iffI, angle = i)
 
        if must_return:
            return out
    

    @staticmethod
    def Paganin(delta_beta):
        '''
        Method to filter a set of projections using the Paganin phase filter
        described in https://doi.org/10.1046/j.1365-2818.2002.01010.x 
        In this implmemntation, the strength of the filter is set using the delta_beta ratio alone. To retrieve quantitative information 
        from phase contrast images with other physical parameters, use the `PhaseRetriever.Paganin() method` instead

        Parameters
        ----------           
        delta_beta: float
            Filter strength, can be given by the ratio of the real and complex part of the material refractive index, where refractive 
            index n = (1 - delta) + i beta (energy-dependent refractive index information can be found at https://refractiveindex.info/ )
            default is 1e-2

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
        return PaganinFilter(delta_beta)
    
class PaganinFilter(Filter):
    def __init__(self, delta_beta=1e-2):
        
        kwargs = {
        'delta_beta': delta_beta,
        'filter' : None}

        super(PaganinFilter, self).__init__(**kwargs)

    def create_filter(self, Nx, Ny):
        processor = PhaseRetriever.Paganin(energy_eV = 1, delta = 1, beta = 1/self.delta_beta, unit_multiplier = 1, propagation_distance = 1, magnification = 1, filter_type='paganin_method')
        self.filter = processor.create_filter(Nx, Ny)
    
        

    