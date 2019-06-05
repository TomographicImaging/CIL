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
from ccpi.framework import DataContainer, VectorGeometry

class VectorData(DataContainer):
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
                self.geometry = VectorGeometry.VectorGeometry(array.shape[0])
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
