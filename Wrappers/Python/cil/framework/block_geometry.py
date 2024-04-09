# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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

import functools
from .block_data_container import BlockDataContainer

class BlockGeometry(object):

    RANDOM = 'random'
    RANDOM_INT = 'random_int'

    @property
    def dtype(self):
        return tuple(i.dtype for i in self.geometries)

    '''Class to hold Geometry as column vector'''
    #__array_priority__ = 1
    def __init__(self, *args, **kwargs):
        ''''''
        self.geometries = args
        self.index = 0
        shape = (len(args),1)
        self.shape = shape

        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements, len(args)))

    def get_item(self, index):
        '''returns the Geometry in the BlockGeometry located at position index'''
        return self.geometries[index]

    def allocate(self, value=0, **kwargs):

        '''Allocates a BlockDataContainer according to geometries contained in the BlockGeometry'''

        symmetry = kwargs.get('symmetry',False)
        containers = [geom.allocate(value, **kwargs) for geom in self.geometries]

        if symmetry == True:

            # for 2x2
            # [ ig11, ig12\
            #   ig21, ig22]

            # Row-wise Order

            if len(containers)==4:
                containers[1]=containers[2]

            # for 3x3
            # [ ig11, ig12, ig13\
            #   ig21, ig22, ig23\
            #   ig31, ig32, ig33]

            elif len(containers)==9:
                containers[1]=containers[3]
                containers[2]=containers[6]
                containers[5]=containers[7]

            # for 4x4
            # [ ig11, ig12, ig13, ig14\
            #   ig21, ig22, ig23, ig24\ c
            #   ig31, ig32, ig33, ig34
            #   ig41, ig42, ig43, ig44]

            elif len(containers) == 16:
                containers[1]=containers[4]
                containers[2]=containers[8]
                containers[3]=containers[12]
                containers[6]=containers[9]
                containers[7]=containers[10]
                containers[11]=containers[15]

        return BlockDataContainer(*containers)

    def __iter__(self):
        '''BlockGeometry is an iterable'''
        return self

    def __next__(self):
        '''BlockGeometry is an iterable'''
        if self.index < len(self.geometries):
            result = self.geometries[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration
