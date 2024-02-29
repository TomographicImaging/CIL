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

from cil.framework import DataProcessor

from .PhaseRetriever_Paganin import PaganinPhaseRetrieval, GeneralisedPhaseRetrieval, OriginalPhaseRetrieval
from cil.framework import Processor


class PaganinPhaseRetriever(Processor):
    """
    This class contains methods to create a phase retrieval processor using the desired algorithm.
    """

    @staticmethod
    def paganin(energy_eV = 40000, delta = 1, beta = 1e-3, unit_multiplier = 1, propagation_distance = None):
        """Method to create a phase retrieval processor using the Paganin phase retrieval algorithm
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
        Processor
            Paganin phase retrieval processor
                    

        Example
        -------
        >>> processor = PhaseRetriever.paganin(energy_eV, delta, beta, unit_multiplier)
        >>> processor.set_input(self.data)
        >>> processor.get_output()
        
        Or calling only the Paganin filter step
        >>> processor = PhaseRetriever.paganin(energy_eV, delta, beta, unit_multiplier)
        >>> processor.set_input(self.data)
        >>> processor.filter_image()
        """
        processor = PaganinPhaseRetrieval(energy_eV, delta, beta, unit_multiplier, propagation_distance)
        return processor
    
    @staticmethod
    def generalised_paganin(energy_eV = 40000, delta = 1, beta = 1e-3, unit_multiplier = 1, propagation_distance = None):
        return GeneralisedPhaseRetrieval(energy_eV, delta, beta, unit_multiplier, propagation_distance)
    
    @staticmethod
    def original_paganin(energy_eV = 40000, delta = 1, beta = 1e-3, unit_multiplier = 1, propagation_distance = None):
        return OriginalPhaseRetrieval(energy_eV, delta, beta, unit_multiplier, propagation_distance)