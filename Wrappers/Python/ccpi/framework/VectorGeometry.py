# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import sys
from datetime import timedelta, datetime
import warnings
from functools import reduce
from numbers import Number
from ccpi.framework.VectorData import VectorData

class VectorGeometry(object):
    RANDOM = 'random'
    RANDOM_INT = 'random_int'
        
    def __init__(self, 
                 length):
        
        self.length = length
        self.shape = (length, )
        
        
    def clone(self):
        '''returns a copy of VectorGeometry'''
        return VectorGeometry(self.length)

    def allocate(self, value=0, **kwargs):
        '''allocates an VectorData according to the size expressed in the instance'''
        self.dtype = kwargs.get('dtype', numpy.float32)
        out = VectorData(geometry=self, dtype=self.dtype)
        if isinstance(value, Number):
            if value != 0:
                out += value
        else:
            if value == VectorGeometry.RANDOM:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed) 
                out.fill(numpy.random.random_sample(self.shape))
            elif value == VectorGeometry.RANDOM_INT:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                max_value = kwargs.get('max_value', 100)
                out.fill(numpy.random.randint(max_value,size=self.shape))
            else:
                raise ValueError('Value {} unknown'.format(value))
        return out