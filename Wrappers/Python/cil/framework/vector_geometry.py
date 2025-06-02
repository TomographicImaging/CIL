#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
import copy
from numbers import Number
import warnings

import numpy

from .labels import FillType

class VectorGeometry:
    '''Geometry describing VectorData to contain 1D array'''
    @property
    def RANDOM(self):
        warnings.warn("use FillType.RANDOM instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM

    @property
    def RANDOM_INT(self):
        warnings.warn("use FillType.RANDOM_INT instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM_INT

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    def __init__(self,
                 length, **kwargs):

        self.length = int(length)
        self.shape = (length, )
        self.dtype = kwargs.get('dtype', numpy.float32)
        self.dimension_labels = kwargs.get('dimension_labels', None)

    def clone(self):
        '''returns a copy of VectorGeometry'''
        return copy.deepcopy(self)

    def copy(self):
        '''alias of clone'''
        return self.clone()

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        if self.length == other.length \
            and self.shape == other.shape \
            and self.dimension_labels == other.dimension_labels:
            return True
        return False

    def __str__ (self):
        repres = ""
        repres += "Length: {0}\n".format(self.length)
        repres += "Shape: {0}\n".format(self.shape)
        repres += "Dimension_labels: {0}\n".format(self.dimension_labels)

        return repres

    def allocate(self, value=0, **kwargs):
        '''Allocates a VectorData according to the geometry

        Parameters
        ----------
        value : number or string, default=0
            The value to allocate. Accepts a number to allocate a uniform array, 
            None to allocate an empty memory block, or a string to create a random 
            array: 'random' allocates floats between 0 and 1, 'random_int' allocates 
            ints between 0 and 100.

        **kwargs:
            dtype : numpy data type, optional
                The data type to allocate if different from the geometry data type. 
                Default None allocates an array with the geometry data type.

            seed : int, optional
                A random seed to fix reproducibility, only used if `value` is a random
                method. Default is `None`.

            min_value : number, optional
                The minimum value random integer to generate, only used if `value` 
                is 'random_int'. Default is 0.

            max_value : number, optional
                The maximum value random integer to generate, only used if `value` 
                is 'random_int'. Default is 100.

        Note
        ----
            The methods used by 'random' or 'random_int' use `numpy.random.default_rng` 
            which allocates memory only for the array of the specified dtype. This
            method does not use the global numpy.random.seed() so if a seed is 
            required it should be passed directly as an argument to allocate.
            To allocate random numbers using the deprecated `numpy.random.random_sample`
            and `numpy.random.randint` methods use `value='random_deprecated'` 
            or `value='random_int_deprecated'` 

        '''
        from .vector_data import VectorData

        dtype = kwargs.pop('dtype', self.dtype)

        out = VectorData(geometry=self.copy(), dtype=dtype)
        if value is not None:
            out.fill(value, **kwargs)

        return out
