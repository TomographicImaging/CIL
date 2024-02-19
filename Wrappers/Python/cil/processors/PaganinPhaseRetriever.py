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

from .PhaseRetriever_Paganin import PaganinPhaseRetrieval
from cil.framework import Processor


class PaganinPhaseRetriever(Processor):
    """
    This class contains methods to create a phase retrieval processor using the desired algorithm.
    """

    @staticmethod
    def paganin(energy_eV = 40000, delta = 1, beta = 1e-3, unit_multiplier = 1, normalise = False, units_output='absorption'):
        """Method to create a phase retrieval processor using the Paganin phase retrieval algorithm
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
        Processor
            Paganin phase retrieval processor
                    

        Example
        -------
        >>> processor = PhaseRetriever.paganin(energy_eV, delta, beta, unit_multiplier)
        >>> processor.set_input(self.data)

        """
        processor = PaganinPhaseRetrieval(energy_eV, delta, beta, unit_multiplier, normalise, units_output)
        return processor