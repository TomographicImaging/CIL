#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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


from cil.framework import DataProcessor, AcquisitionData, ImageData, ImageGeometry, DataContainer, AcquisitionGeometry
import numpy as np
import weakref

class Padder(DataProcessor):
    """
    Processor to pad an array with a border, wrapping numpy.pad. See https://numpy.org/doc/stable/reference/generated/numpy.pad.html


    It is recommended to use the static methods to configure your Padder object rather than initialising this class directly. See examples for details.


    Parameters
    ----------
    mode: str
        The method used to populate the border data. Accepts: 'constant', 'edge', 'linear_ramp', 'reflect', 'symmetric', 'wrap'
    pad_width: int, tuple, dict
        The size of the border along each axis, see usage notes
    pad_values: float, tuple, dict, default=0.0
        The additional values needed by some of the modes

    Notes
    -----
    `pad_width` behaviour (number of pixels):
        - int: Each axis will be padded with a border of this size
        - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
        - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

    `pad_values` behaviour:
        - float: Each border will use this value
        - tuple(float, float): Each value will be used asymmetrically for each axis i.e. (before, after)
        - dict: Specified axes and values: e.g. {'horizontal':(8, 23), 'channel':5}

    If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
    and the last two angles in geometry.angles. The user should ensure the output is as expected.


    Example
    -------
    >>> processor = Padder.edge(pad_width=1)
    >>> processor.set_input(data)
    >>> data_padded = processor.get_output()
    >>> print(data.array)
    [[0. 1. 2.]
    [3. 4. 5.]
    [6. 7. 8.]]
    >>> print(data_padded.array)
    [[0. 0. 1. 2. 2.]
    [0. 0. 1. 2. 2.]
    [3. 3. 4. 5. 5.]
    [6. 6. 7. 8. 8.]
    [6. 6. 7. 8. 8.]]

    Example
    -------
    >>> processor = Padder.constant(pad_width={'horizontal_y':(1,1),'horizontal_x':(1,2)}, constant_values=(-1.0, 1.0))
    >>> processor.set_input(data)
    >>> data_padded = processor.get_output()
    >>> print(data.array)
    [[0. 1. 2.]
    [3. 4. 5.]
    [6. 7. 8.]]
    >>> print(data_padded.array)
    [[-1. -1. -1. -1.  1.  1.]
    [-1.  0.  1.  2.  1.  1.]
    [-1.  3.  4.  5.  1.  1.]
    [-1.  6.  7.  8.  1.  1.]
    [-1.  1.  1.  1.  1.  1.]

    """


    @staticmethod
    def constant(pad_width=None, constant_values=0.0):
        """
        Padder processor wrapping numpy.pad with mode `constant`.

        Pads the data with a constant value border. Pads in all *spatial*
        dimensions unless a dictionary is passed to either `pad_width` or `constant_values`

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis, see usage notes
        constant_values: float, tuple, dict, default=0.0
            The value of the border, see usage notes

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - int: Each axis will be padded with a border of this size
         - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
         - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

        `constant_values` behaviour (value of pixels):
         - float: Each border will be set to this value
         - tuple(float, float): Each border value will be used asymmetrically for each axis i.e. (before, after)
         - dict: Specified axes and values: e.g. {'horizontal':(8, 23), 'channel':5}

        If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
        and the last two angles in geometry.angles. The user should ensure the output is as expected.

        Example
        -------
        >>> processor = Padder.constant(pad_width=1, constant_values=0.0)
        >>> processor.set_input(data)
        >>> data_padded = processor.get_output()
        >>> print(data.array)
        [[0. 1. 2.]
        [3. 4. 5.]
        [6. 7. 8.]]
        >>> print(data_padded.array)
        [[0. 0. 0. 0. 0.]
        [0. 0. 1. 2. 0.]
        [0. 3. 4. 5. 0.]
        [0. 6. 7. 8. 0.]
        [0. 0. 0. 0. 0.]]

        """

        processor = Padder(pad_width=pad_width, mode='constant', pad_values=constant_values)
        return processor


    @staticmethod
    def edge(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `edge`.

        Pads the data by extending the edge values in to the border. Pads in all *spatial*
        dimensions unless a dictionary is passed to `pad_width`.

        pad_width: int, tuple, dict
            The size of the border along each axis, see usage notes

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - int: Each axis will be padded with a border of this size
         - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
         - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

        If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
        and the last two angles in geometry.angles. The user should ensure the output is as expected.

        Example
        -------
        >>> processor = Padder.edge(pad_width=1)
        >>> processor.set_input(data)
        >>> data_padded = processor.get_output()
        >>> print(data.array)
        [[0. 1. 2.]
        [3. 4. 5.]
        [6. 7. 8.]]
        >>> print(data_padded.array)
        [[0. 0. 1. 2. 2.]
        [0. 0. 1. 2. 2.]
        [3. 3. 4. 5. 5.]
        [6. 6. 7. 8. 8.]
        [6. 6. 7. 8. 8.]]

        """

        processor = Padder(pad_width=pad_width, mode='edge')
        return processor

    @staticmethod
    def linear_ramp(pad_width=None, end_values=0.0):
        """Padder processor wrapping numpy.pad with mode `linear_ramp`

        Pads the data with values calculated from a linear ramp between the array edge
        value and the set end_value. Pads in all *spatial* dimensions unless a dictionary
        is passed to either `pad_width` or `constant_values`

        pad_width: int, tuple, dict
            The size of the border along each axis, see usage notes
        end_values: float, tuple, dict, default=0.0
            The target value of the linear_ramp, see usage notes

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - int: Each axis will be padded with a border of this size
         - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
         - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

        `end_values` behaviour:
         - float: Each border will use this end value
         - tuple(float, float): Each border end value will be used asymmetrically for each axis i.e. (before, after)
         - dict: Specified axes and end values: e.g. {'horizontal':(8, 23), 'channel':5}

        If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
        and the last two angles in geometry.angles. The user should ensure the output is as expected.

        Example
        -------
        >>> processor = Padder.linear_ramp(pad_width=2, end_values=0.0)
        >>> processor.set_input(data)
        >>> data_padded = processor.get_output()
        >>> print(data.array)
        [[0. 1. 2.]
        [3. 4. 5.]
        [6. 7. 8.]]
        >>> print(data_padded.array)
        [[0.  0.  0.  0.  0.  0.  0. ]
        [0.  0.  0.  0.5 1.  0.5 0. ]
        [0.  0.  0.  1.  2.  1.  0. ]
        [0.  1.5 3.  4.  5.  2.5 0. ]
        [0.  3.  6.  7.  8.  4.  0. ]
        [0.  1.5 3.  3.5 4.  2.  0. ]
        [0.  0.  0.  0.  0.  0.  0. ]]

        """
        processor = Padder(pad_width=pad_width, mode='linear_ramp', pad_values=end_values)
        return processor


    @staticmethod
    def reflect(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `reflect`.

        Pads with the reflection of the data mirrored along first and last values each axis.
        Pads in all *spatial* dimensions unless a dictionary is passed to `pad_width`.

        pad_width: int, tuple, dict
            The size of the border along each axis, see usage notes

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - int: Each axis will be padded with a border of this size
         - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
         - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

        If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
        and the last two angles in geometry.angles. The user should ensure the output is as expected.

        Example
        -------
        >>> processor = Padder.reflect(pad_width=1)
        >>> processor.set_input(data)
        >>> data_padded = processor.get_output()
        >>> print(data.array)
        [[0. 1. 2.]
        [3. 4. 5.]
        [6. 7. 8.]]
        >>> print(data_padded.array)
        [[4. 3. 4. 5. 4.]
        [1. 0. 1. 2. 1.]
        [4. 3. 4. 5. 4.]
        [7. 6. 7. 8. 7.]
        [4. 3. 4. 5. 4.]]

        """
        processor = Padder(pad_width=pad_width, mode='reflect')
        return processor


    @staticmethod
    def symmetric(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `symmetric`.

        Pads with the reflection of the data mirrored along the edge of the array.
        Pads in all *spatial* dimensions unless a dictionary is passed to `pad_width`.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - int: Each axis will be padded with a border of this size
         - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
         - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

        If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
        and the last two angles in geometry.angles. The user should ensure the output is as expected.

        Example
        -------
        >>> processor = Padder.symmetric(pad_width=1)
        >>> processor.set_input(data)
        >>> data_padded = processor.get_output()
        >>> print(data.array)
        [[0. 1. 2.]
        [3. 4. 5.]
        [6. 7. 8.]]
        >>> print(data_padded.array)
        [[0. 0. 1. 2. 2.]
        [0. 0. 1. 2. 2.]
        [3. 3. 4. 5. 5.]
        [6. 6. 7. 8. 8.]
        [6. 6. 7. 8. 8.]]

        """
        processor = Padder(pad_width=pad_width, mode='symmetric')
        return processor


    @staticmethod
    def wrap(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `wrap`.

        Pads with the wrap of the vector along the axis. The first values are used to pad the
        end and the end values are used to pad the beginning. Pads in all *spatial* dimensions
        unless a dictionary is passed to `pad_width`.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - int: Each axis will be padded with a border of this size
         - tuple(int, int): Each axis will be padded with an asymmetric border i.e. (before, after)
         - dict: Specified axes will be padded: e.g. {'horizontal':(8, 23), 'vertical': 10}

        If padding angles the angular values assigned to the padded axis will be extrapolated from the first two,
        and the last two angles in geometry.angles. The user should ensure the output is as expected.

        Example
        -------
        >>> processor = Padder.wrap(pad_width=1)
        >>> processor.set_input(data)
        >>> data_padded = processor.get_output()
        >>> print(data.array)
        [[0. 1. 2.]
        [3. 4. 5.]
        [6. 7. 8.]]
        >>> print(data_padded.array)
        [[8. 6. 7. 8. 6.]
        [2. 0. 1. 2. 0.]
        [5. 3. 4. 5. 3.]
        [8. 6. 7. 8. 6.]
        [2. 0. 1. 2. 0.]]

        """
        processor = Padder(pad_width=pad_width, mode='wrap')
        return processor


    def __init__(self,
                 mode='constant',
                 pad_width=None,
                 pad_values=0):

        kwargs = {'mode': mode,
                'pad_width': pad_width,
                'pad_values': pad_values,
                '_data_array': False,
                '_geometry': None,
                '_shape_in':None,
                '_shape_out':None,
                '_shape_out_full':None,
                '_labels_in':None,
                '_processed_dims':None,
                '_pad_width_param':None,
                '_pad_values_param':None,
            }

        super(Padder, self).__init__(**kwargs)


    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        dataset : DataContainer, Geometry
            The input DataContainer
        """

        if issubclass(type(dataset), DataContainer) or isinstance(dataset,(AcquisitionGeometry,ImageGeometry)):
            if self.check_input(dataset):
                self.__dict__['input'] = weakref.ref(dataset)
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(dataset), DataContainer))

        self._set_up()

    def check_input(self, data):

        if isinstance(data, (ImageData,AcquisitionData)):
            self._data_array = True
            self._geometry = data.geometry

        elif isinstance(data, DataContainer):
            self._data_array = True
            self._geometry = None

        elif isinstance(data, (ImageGeometry, AcquisitionGeometry)):
            self._data_array = False
            self._geometry = data

        else:
            raise TypeError('Processor supports following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n - DataContainer\n - ImageGeometry\n - AcquisitionGeometry')

        if self._data_array:
            if data.dtype != np.float32:
                raise TypeError("Expected float32")


        if self.mode not in ['constant', 'edge', 'linear_ramp', 'reflect', 'symmetric', 'wrap']:
            raise Exception("Wrong mode. One of the following is expected:\n" +
                            "constant, edge, linear_ramp, reflect, symmetric, wrap")

        if self.pad_width is None:
            raise ValueError('Please, specify pad_width')

        return True

    def _create_tuple(self, value, dtype):
        try:
            out = (dtype(value),dtype(value))
        except TypeError:
            try:
                out = (dtype(value[0]),dtype(value[1]))
            except:
                raise TypeError()

        return out

    def _get_dimensions_from_dict(self, dict):

        dimensions = []
        for k in dict.keys():
            if k not in self._labels_in:
                raise ValueError('Dimension label not found in data. Expected labels from {0}. Got {1}'.format(self._geometry.dimension_labels, k))

            dimensions.append(k)
        return dimensions


    def _set_up(self):
        
        data = self.get_input()
        offset = 4-data.ndim

        #set defaults
        self._labels_in = [None]*4
        self._labels_in[offset::] = data.dimension_labels

        self._shape_in = [1]*4
        self._shape_in[offset::] = data.shape

        self._shape_out_full = self._shape_in.copy()

        self._processed_dims = [0,0,0,0]

        self._pad_width_param = [(0,0)]*4
        self._pad_values_param = [(0,0)]*4


        # if pad_width or set_values is passed a dictionary these keys specify the axes to run over
        if isinstance(self.pad_width, dict) and isinstance(self.pad_values, dict):

            if self.pad_width.keys() != self.pad_values.keys():
                raise ValueError('Dictionaries must contain the same axes')

            dimensions = self._get_dimensions_from_dict(self.pad_width)

        elif isinstance(self.pad_width, dict):
            dimensions = self._get_dimensions_from_dict(self.pad_width)

        elif isinstance(self.pad_values, dict):
            dimensions = self._get_dimensions_from_dict(self.pad_values)

        else:
            spatial_dimensions =[
                        'vertical',
                        'horizontal',
                        'horizontal_y',
                        'horizontal_x'
                    ]

            dimensions = list(set(spatial_dimensions) & set(self._labels_in))


        # get pad_widths for these dimensions
        for dim in dimensions:
            try:
                values = self.pad_width[dim]
            except TypeError:
                values = self.pad_width

            try:
                i = self._labels_in.index(dim)
                self._pad_width_param[i] = self._create_tuple(values, int)
            except TypeError:
                raise TypeError("`pad_width` should be a integer or a tuple of integers. Got {0} for axis {1}".format(values, dim))

        # get pad_values for these dimensions
        for dim in dimensions:
            try:
                values = self.pad_values[dim]
            except TypeError:
                values = self.pad_values

            try:
                i = self._labels_in.index(dim)
                self._pad_values_param[i] = self._create_tuple(values, float)
            except TypeError:
                raise TypeError("`pad_values` should be a float or a tuple of floats. Got {0} for axis {1}".format(values, dim))

        #create list of processed axes and new_shape
        for i in range(4):
           if self._pad_width_param[i] != (0,0):
                self._processed_dims[i] = 1
                self._shape_out_full[i] += self._pad_width_param[i][0] + self._pad_width_param[i][1]

        self._shape_out = tuple([i for i in self._shape_out_full if i > 1])


    def _process_acquisition_geometry(self):
        """
        Creates the new acquisition geometry
        """

        geometry = self._geometry.copy()
        for i, dim in enumerate(self._labels_in):

            if not self._processed_dims[i]:
                continue

            offset = (self._pad_width_param[i][0] -self._pad_width_param[i][1])*0.5

            if dim == 'channel':
                geometry.set_channels(num_channels= geometry.config.channels.num_channels + \
                self._pad_width_param[i][0] + self._pad_width_param[i][1])
            elif dim == 'angle':
                # extrapolate pre-values from a[1]-a[0]
                # extrapolate post-values from a[-1]-a[-2]
                a = self._geometry.angles
                end_values = (
                    a[0]-(a[1]-a[0] )* self._pad_width_param[i][0],
                    a[-1]+(a[-1]-a[-2] )* self._pad_width_param[i][1]
                    )
                geometry.config.angles.angle_data = np.pad(a, (self._pad_width_param[i][0],self._pad_width_param[i][1]), mode='linear_ramp',end_values=end_values)

            elif dim == 'vertical':
                geometry.config.panel.num_pixels[1] += self._pad_width_param[i][0]
                geometry.config.panel.num_pixels[1] += self._pad_width_param[i][1]
                geometry.config.shift_detector_in_plane(offset, dim)

            elif dim == 'horizontal':
                geometry.config.panel.num_pixels[0] += self._pad_width_param[i][0]
                geometry.config.panel.num_pixels[0] += self._pad_width_param[i][1]
                geometry.config.shift_detector_in_plane(offset, dim)

        return geometry

    def _process_image_geometry(self):
        """
        Creates the new image geometry
        """
        geometry = self._geometry.copy()
        for i, dim in enumerate(self._labels_in):

            if not self._processed_dims[i]:
                continue

            offset = (self._pad_width_param[i][0] -self._pad_width_param[i][1])*0.5

            if dim == 'channel':
                geometry.channels += self._pad_width_param[i][0]
                geometry.channels += self._pad_width_param[i][1]
            elif dim == 'vertical':
                geometry.voxel_num_z += self._pad_width_param[i][0]
                geometry.voxel_num_z += self._pad_width_param[i][1]
                geometry.center_z -= offset * geometry.voxel_size_z
            elif dim == 'horizontal_x':
                geometry.voxel_num_x += self._pad_width_param[i][0]
                geometry.voxel_num_x += self._pad_width_param[i][1]
                geometry.center_x -= offset * geometry.voxel_size_x
            elif dim == 'horizontal_y':
                geometry.voxel_num_y += self._pad_width_param[i][0]
                geometry.voxel_num_y += self._pad_width_param[i][1]
                geometry.center_y -= offset * geometry.voxel_size_y

        return geometry


    def _process_data(self, dc_in):
        arr_in = dc_in.array.reshape(self._shape_in)

        if self.mode in ['reflect', 'symmetric', 'wrap', 'edge']:
            arr_out = np.pad(arr_in, self._pad_width_param, mode=self.mode,).squeeze()
        elif self.mode == 'constant':
            arr_out = np.pad(arr_in, self._pad_width_param, mode=self.mode, \
                constant_values=self._pad_values_param).squeeze()
        elif self.mode == 'linear_ramp':
            arr_out = np.pad(arr_in, self._pad_width_param, mode=self.mode, \
                end_values=self._pad_values_param).squeeze()

        return arr_out



    def process(self, out=None):

        data = self.get_input()

        # pad geometry
        if isinstance(self._geometry, ImageGeometry):
            new_geometry = self._process_image_geometry()
        elif isinstance(self._geometry, AcquisitionGeometry):
            new_geometry = self._process_acquisition_geometry()
        else:
            new_geometry = None

        # return if just acting on geometry
        if not self._data_array:
            return new_geometry

        # pad data
        if out is None:
            arr_out = self._process_data(data)

            if isinstance(new_geometry, ImageGeometry):
                return ImageData(arr_out,deep_copy=False, geometry=new_geometry)
            elif isinstance(new_geometry, AcquisitionGeometry):
                return AcquisitionData(arr_out,deep_copy=False, geometry=new_geometry)
            else:
                return DataContainer(arr_out,deep_copy=False, dimension_labels=data.dimension_labels)

        else:
            # check size and shape if passed out
            try:
                out.array = out.array.reshape(self._shape_out_full)
            except:
                raise ValueError("Array of `out` not compatible. Expected shape: {0}, data type: {1} Got shape: {2}, data type: {3}".format(self._shape_out_full, np.float32, out.array.shape, out.array.dtype))

            if new_geometry is not None:
                if out.geometry != new_geometry:
                    raise ValueError("Geometry of `out` not as expected. Got {0}, expected {1}".format(out.geometry, new_geometry))

            out.array = self._process_data(data)
            return out
