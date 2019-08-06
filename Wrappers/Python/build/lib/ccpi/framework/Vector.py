# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

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
from ccpi.framework import DataContainer

class VectorData(DataContainer):
    '''DataContainer to contain 1D array'''
    def __init__(self, array=None, **kwargs):
        self.geometry = kwargs.get('geometry', None)
        self.dtype = kwargs.get('dtype', numpy.float32)
        
        if self.geometry is None:
            if array is None:
                raise ValueError('Please specify either a geometry or an array')
            else:
                if len(array.shape) > 1:
                    raise ValueError('Incompatible size: expected 1D got {}'.format(array.shape))
                out = array
                self.geometry = VectorGeometry(array.shape[0])
                self.length = self.geometry.length
        else:
            self.length = self.geometry.length
                
            if array is None:
                out = numpy.zeros((self.length,), dtype=self.dtype)
            else:
                if self.length == array.shape[0]:
                    out = array
                else:
                    raise ValueError('Incompatible size: expecting {} got {}'.format((self.length,), array.shape))
        deep_copy = True
        super(VectorData, self).__init__(out, deep_copy, None)

class VectorGeometry(object):
    '''Geometry describing VectorData to contain 1D array'''
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
