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
import warnings
from numbers import Number

import numpy

from .image_data import ImageData
from .labels import ImageDimension, FillType


class BackwardCompat(type):
    @property
    def CHANNEL(cls):
        warnings.warn("use ImageDimension.CHANNEL instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.CHANNEL

    @property
    def HORIZONTAL_X(cls):
        warnings.warn("use ImageDimension.HORIZONTAL_X instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.HORIZONTAL_X

    @property
    def HORIZONTAL_Y(cls):
        warnings.warn("use ImageDimension.HORIZONTAL_Y instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.HORIZONTAL_Y

    @property
    def RANDOM(cls):
        warnings.warn("use FillType.RANDOM instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM

    @property
    def RANDOM_INT(cls):
        warnings.warn("use FillType.RANDOM_INT instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM_INT

    @property
    def VERTICAL(cls):
        warnings.warn("use ImageDimension.VERTICAL instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.VERTICAL


class ImageGeometry(metaclass=BackwardCompat):
    @property
    def shape(self):
        shape_dict = {ImageDimension.CHANNEL: self.channels,
                      ImageDimension.VERTICAL: self.voxel_num_z,
                      ImageDimension.HORIZONTAL_Y: self.voxel_num_y,
                      ImageDimension.HORIZONTAL_X: self.voxel_num_x}
        return tuple(shape_dict[label] for label in self.dimension_labels)

    @property
    def spacing(self):
        spacing_dict = {ImageDimension.CHANNEL: self.channel_spacing,
                        ImageDimension.VERTICAL: self.voxel_size_z,
                        ImageDimension.HORIZONTAL_Y: self.voxel_size_y,
                        ImageDimension.HORIZONTAL_X: self.voxel_size_x}
        return tuple(spacing_dict[label] for label in self.dimension_labels)

    @property
    def length(self):
        return len(self.dimension_labels)

    @property
    def ndim(self):
        return len(self.dimension_labels)

    @property
    def dimension_labels(self):

        labels_default = ImageDimension.get_order_for_engine("cil")

        shape_default = [   self.channels,
                            self.voxel_num_z,
                            self.voxel_num_y,
                            self.voxel_num_x]

        try:
            labels = self._dimension_labels
        except AttributeError:
            labels = labels_default
        labels = list(labels)

        for i, x in enumerate(shape_default):
            if x == 0 or x==1:
                try:
                    labels.remove(labels_default[i])
                except ValueError:
                    pass #if not in custom list carry on
        return tuple(labels)

    @dimension_labels.setter
    def dimension_labels(self, val):
        self.set_labels(val)

    def set_labels(self, labels):
        if labels is not None:
            self._dimension_labels = tuple(map(ImageDimension, labels))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.voxel_num_x == other.voxel_num_x \
            and self.voxel_num_y == other.voxel_num_y \
            and self.voxel_num_z == other.voxel_num_z \
            and self.voxel_size_x == other.voxel_size_x \
            and self.voxel_size_y == other.voxel_size_y \
            and self.voxel_size_z == other.voxel_size_z \
            and self.center_x == other.center_x \
            and self.center_y == other.center_y \
            and self.center_z == other.center_z \
            and self.channels == other.channels \
            and self.channel_spacing == other.channel_spacing \
            and self.dimension_labels == other.dimension_labels:

            return True

        return False

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    def __init__(self,
                 voxel_num_x=0,
                 voxel_num_y=0,
                 voxel_num_z=0,
                 voxel_size_x=1,
                 voxel_size_y=1,
                 voxel_size_z=1,
                 center_x=0,
                 center_y=0,
                 center_z=0,
                 channels=1,
                 **kwargs):

        self.voxel_num_x = int(voxel_num_x)
        self.voxel_num_y = int(voxel_num_y)
        self.voxel_num_z = int(voxel_num_z)
        self.voxel_size_x = float(voxel_size_x)
        self.voxel_size_y = float(voxel_size_y)
        self.voxel_size_z = float(voxel_size_z)
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.channels = channels
        self.channel_labels = None# check if None, single channel geom
        self.channel_spacing = 1.0
        self.dimension_labels = kwargs.get('dimension_labels', None)
        self.dtype = kwargs.get('dtype', numpy.float32)


    def get_slice(self,channel=None, vertical=None, horizontal_x=None, horizontal_y=None):
        '''
        Returns a new ImageGeometry of a single slice in the requested direction.

        Parameters
        ----------
        channel : int, optional
            The channel index to slice. Default is None (no slicing).
        vertical : int or 'centre', optional
            The vertical index to slice. Default is None (no slicing).
        horizontal_x : int or 'centre', optional
            The horizontal x index to slice. Default is None (no slicing).
        horizontal_y : int or 'centre', optional
            The horizontal y index to slice. Default is None (no slicing).
        Returns
        -------
        geometry_new : ImageGeometry
            A new ImageGeometry object representing the sliced geometry.

        Note
        ----
        Slicing on vertical, horizontal_x or horizontal_y with 'centre' will return the
        central slice in that dimension.
        Slicing on channels returns a geometry with a single channel, however the channel label is not
        typically stored in the geometry.
        '''

        geometry_new = self.copy()
        if channel is not None:
            geometry_new.channels = 1
            try:
                geometry_new.channel_labels = [self.channel_labels[channel]]
            except:
                geometry_new.channel_labels = None


        if vertical is not None:
            geometry_new.voxel_num_z = 1
            if vertical != 'centre':
                if vertical == 0:
                    warnings.warn("Slicing vertical at index 0 results in a geometry \
                                  offset along the vertical axis. If you do not require an offset ImageGeometry, set vertical='centre",
                                  UserWarning)
                voxel_offset = (self.voxel_num_z)/2 - (vertical+0.5)
                geometry_new.center_z -= voxel_offset * geometry_new.voxel_size_z

        if horizontal_y is not None:
            geometry_new.voxel_num_y = 1
            if horizontal_y != 'centre':
                voxel_offset = (self.voxel_num_y)/2 - (horizontal_y +0.5)
                geometry_new.center_y -= voxel_offset * geometry_new.voxel_size_y

        if horizontal_x is not None:
            geometry_new.voxel_num_x = 1
            if horizontal_x != 'centre':
                voxel_offset = (self.voxel_num_x)/2 - (horizontal_x+0.5)
                geometry_new.center_x -= voxel_offset * geometry_new.voxel_size_x

        return geometry_new
    
    def get_centre_slice(self):
        '''
        Returns a new ImageGeometry of the centre slice in the vertical direction.
        '''
        return self.get_slice(vertical='centre')

    def get_order_by_label(self, dimension_labels, default_dimension_labels):
        order = []
        for i, el in enumerate(default_dimension_labels):
            for j, ek in enumerate(dimension_labels):
                if el == ek:
                    order.append(j)
                    break
        return order

    def get_min_x(self):
        return self.center_x - 0.5*self.voxel_num_x*self.voxel_size_x

    def get_max_x(self):
        return self.center_x + 0.5*self.voxel_num_x*self.voxel_size_x

    def get_min_y(self):
        return self.center_y - 0.5*self.voxel_num_y*self.voxel_size_y

    def get_max_y(self):
        return self.center_y + 0.5*self.voxel_num_y*self.voxel_size_y

    def get_min_z(self):
        if not self.voxel_num_z == 0:
            return self.center_z - 0.5*self.voxel_num_z*self.voxel_size_z
        else:
            return 0

    def get_max_z(self):
        if not self.voxel_num_z == 0:
            return self.center_z + 0.5*self.voxel_num_z*self.voxel_size_z
        else:
            return 0

    def clone(self):
        '''returns a copy of the ImageGeometry'''
        return copy.deepcopy(self)

    def copy(self):
        '''alias of clone'''
        return self.clone()

    def __str__ (self):
        repres = ""
        repres += "Number of channels: {0}\n".format(self.channels)
        repres += "channel_spacing: {0}\n".format(self.channel_spacing)

        if self.voxel_num_z > 0:
            repres += "voxel_num : x{0},y{1},z{2}\n".format(self.voxel_num_x, self.voxel_num_y, self.voxel_num_z)
            repres += "voxel_size : x{0},y{1},z{2}\n".format(self.voxel_size_x, self.voxel_size_y, self.voxel_size_z)
            repres += "center : x{0},y{1},z{2}\n".format(self.center_x, self.center_y, self.center_z)
        else:
            repres += "voxel_num : x{0},y{1}\n".format(self.voxel_num_x, self.voxel_num_y)
            repres += "voxel_size : x{0},y{1}\n".format(self.voxel_size_x, self.voxel_size_y)
            repres += "center : x{0},y{1}\n".format(self.center_x, self.center_y)

        return repres
    def allocate(self, value=0, **kwargs):
        '''Allocates an ImageData according to the geometry

        Parameters
        ----------
        value : number or string, default=0
            The value to allocate. Accepts a number to allocate a uniform array, 
            None to allocate an empty memory block, or a string to create a random 
            array: 'random' allocates floats between 0 and 1, 'random_int' by default
            allocates integers between 0 and 100  or between provided `min_value` and 
            `max_value`
        
        **kwargs:
            dtype : numpy data type, optional
                The data type to allocate if different from the geometry data type. 
                Default None allocates an array with the geometry data type.

            seed : int, optional
                A random seed to fix reproducibility, only used if `value` is a random
                method. Default is `None`.

            min_value : int, optional
                The minimum value random integer to generate, only used if `value` 
                is 'random_int'. New since version 25.0.0. Default is 0.
            
            max_value : int, optional
                The maximum value random integer to generate, only used if `value` 
                is 'random_int'. Default is 100.

        Note
        ----
            Since v25.0.0 the methods used by 'random' or 'random_int' use `numpy.random.default_rng`. 
            This method does not use the global numpy.random.seed() so if a seed is 
            required it should be passed directly as a kwarg.
            To allocate random numbers using the earlier behaviour use `value='random_deprecated'` 
            or `value='random_int_deprecated'` 

        '''
        dtype = kwargs.pop('dtype', self.dtype)


        out = ImageData(geometry=self.copy(), dtype=dtype)
        if value is not None:
            out.fill(value, **kwargs)
        
        return out
