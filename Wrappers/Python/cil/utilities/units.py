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
# %%
from enum import Enum
from scipy import constants
import numpy as np

class Units(Enum):
    
    def __init__(self, aliases, multiplier):
        self.aliases = aliases
        self.multiplier = multiplier
    
    @classmethod
    def get_multiplier(cls, unit):
        if isinstance(unit, cls):
            return unit.multiplier
        elif isinstance(unit, str):
            # unit = unit.lower()
            for member in cls:
                if unit in member.aliases:
                    return member.multiplier
        raise ValueError(f"Unknown unit: {unit}, must be one of {cls.list()}")
    
    @classmethod
    def convert(cls, value, unit_from, unit_to):
        return value*cls.get_multiplier(unit_from)/cls.get_multiplier(unit_to)
    @classmethod
    def list(cls):
        return [e for e in cls]
    
class DistanceUnits(Units):
    m = (['m', 'meter', 'metre'], 1.0)
    cm = (['cm', 'centimeter', 'centimetre'], 1e-2)
    mm = (['mm', 'millimeter', 'millimetre'], 1e-3)
    um = (['um', 'micrometer', 'micrometre'], 1e-6)

class AngleUnits(Units):
    rad = (['rad', 'radian'], np.rad2deg(1))
    deg = (['deg', 'degree'], 1)

class EnergyUnits(Units):
    meV = (['meV', 'millielectronvolts'], 1e-3)
    eV = (['eV', 'electronvolts'], 1)
    keV = (['keV', 'kiloelectronvolts'], 1e3)
    MeV = (['MeV', 'megaelectronvolts'], 1e6)
    J = (['J', 'Joules', 'joules'], 1/constants.eV)
# %%
