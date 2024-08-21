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


class ImageGeometry:
    @property
    def CHANNEL(self):
        warnings.warn("use ImageDimensionLabels.CHANNEL instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.CHANNEL

    @property
    def HORIZONTAL_X(self):
        warnings.warn("use ImageDimensionLabels.HORIZONTAL_X instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.HORIZONTAL_X

    @property
    def HORIZONTAL_Y(self):
        warnings.warn("use ImageDimensionLabels.HORIZONTAL_Y instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.HORIZONTAL_Y

    @property
    def RANDOM(self):
        warnings.warn("use FillTypes.RANDOM instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM
    @property
    def RANDOM_INT(self):
        warnings.warn("use FillTypes.RANDOM_INT instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM_INT

    @property
    def VERTICAL(self):
        warnings.warn("use ImageDimensionLabels.VERTICAL instead", DeprecationWarning, stacklevel=2)
        return ImageDimension.VERTICAL

    @property
    def shape(self):
        shape_dict = {ImageDimension.CHANNEL: self.channels,
                      ImageDimension.VERTICAL: self.voxel_num_z,
                      ImageDimension.HORIZONTAL_Y: self.voxel_num_y,
                      ImageDimension.HORIZONTAL_X: self.voxel_num_x}
        return tuple(shape_dict[label] for label in self.dimension_labels)

    @shape.setter
    def shape(self, val):
        print("Deprecated - shape will be set automatically")

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
        self.channel_labels = None
        self.channel_spacing = 1.0
        self.dimension_labels = kwargs.get('dimension_labels', None)
        self.dtype = kwargs.get('dtype', numpy.float32)


    def get_slice(self,channel=None, vertical=None, horizontal_x=None, horizontal_y=None):
        '''
        Returns a new ImageGeometry of a single slice of in the requested direction.
        '''
        geometry_new = self.copy()
        if channel is not None:
            geometry_new.channels = 1

            try:
                geometry_new.channel_labels = [self.channel_labels[channel]]
            except:
                geometry_new.channel_labels = None

        if vertical is not None:
            geometry_new.voxel_num_z = 0

        if horizontal_y is not None:
            geometry_new.voxel_num_y = 0

        if horizontal_x is not None:
            geometry_new.voxel_num_x = 0

        return geometry_new

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
        '''allocates an ImageData according to the size expressed in the instance

        :param value: accepts numbers to allocate an uniform array, or a string as 'random' or 'random_int' to create a random array or None.
        :type value: number or string, default None allocates empty memory block, default 0
        :param dtype: numerical type to allocate
        :type dtype: numpy type, default numpy.float32
        '''

        dtype = kwargs.get('dtype', self.dtype)

        if kwargs.get('dimension_labels', None) is not None:
            raise ValueError("Deprecated: 'dimension_labels' cannot be set with 'allocate()'. Use 'geometry.set_labels()' to modify the geometry before using allocate.")

        out = ImageData(geometry=self.copy(),
                            dtype=dtype,
                            suppress_warning=True)

        if isinstance(value, Number):
            # it's created empty, so we make it 0
            out.array.fill(value)
        elif value in FillType:
            if value == FillType.RANDOM:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                if numpy.iscomplexobj(out.array):
                    r = numpy.random.random_sample(self.shape) + 1j * numpy.random.random_sample(self.shape)
                    out.fill(r)
                else:
                    out.fill(numpy.random.random_sample(self.shape))

            elif value == FillType.RANDOM_INT:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                max_value = kwargs.get('max_value', 100)
                if numpy.iscomplexobj(out.array):
                    out.fill(numpy.random.randint(max_value,size=self.shape, dtype=numpy.int32) + 1.j*numpy.random.randint(max_value,size=self.shape, dtype=numpy.int32))
                else:
                    out.fill(numpy.random.randint(max_value,size=self.shape, dtype=numpy.int32))
        elif value is None:
            pass
        else:
            raise ValueError(f'Value {value} unknown')
        return out
