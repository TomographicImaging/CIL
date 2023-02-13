# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from cil.framework import DataProcessor, AcquisitionData, ImageData
from numbers import Number
from cil.framework import  DataContainer, AcquisitionGeometry, ImageGeometry
import numpy as np
import weakref

class Padder(DataProcessor):
    r'''
    Processor to pad an array, wrapping numpy.pad
    See https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    '''

    @staticmethod
    def constant(pad_width=None, constant_values=0):
        """
        Padder processor wrapping numpy.pad with mode `constant` 
        Pads the data with a constant value border.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis
        constant_values: float, tuple, dict
            The value of the border

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - An integer value will pad with a border of this size in all *spatial* dimensions and directions
         - A tuple will pad with an asymmetric border in all *spatial* dimensions i.e. (before, after)
         - A dictionary will apply the specified padding in each requested dimension: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        
        `constant_values` behaviour (value of pixels):
         - A single value will be used for borders in all padded directions and dimensions
         - A tuple of values will be used asymmetrically by each dimension i.e. (before, after)
         - A dictionary will set the values for the requested dimension only: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        """

        processor = Padder(pad_width=pad_width, mode='constant', pad_values=constant_values)
        return processor

    @staticmethod
    def edge(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `edge` 
        Pads the data by extending the edge values in to the border

        pad_width: int, tuple, dict
            The size of the border along each axis

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - An integer value will pad with a border of this size in all *spatial* dimensions and directions
         - A tuple will pad with an asymmetric border in all *spatial* dimensions i.e. (before, after)
         - A dictionary will apply the specified padding in each requested dimension: e.g.
        {'horizontal':(8, 23), 'vertical': 10}      
        """

        processor = Padder(pad_width=pad_width, mode='edge')
        return processor
    
    @staticmethod
    def linear_ramp(pad_width=None, end_values=0):
        '''Padder processor wrapping numpy.pad with mode `linear_ramp` 
        Pads the data with values calculated from a linear ramp between the array edge value and the set end_value.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis
        end_values: float, tuple, dict
            The target value of the linear_ramp

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - An integer value will pad with a border of this size in all *spatial* dimensions and directions
         - A tuple will pad with an asymmetric border in all *spatial* dimensions i.e. (before, after)
         - A dictionary will apply the specified padding in each requested dimension: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        
        `end_values` behaviour:
         - A single value will be used for borders in all padded directions and dimensions
         - A tuple of values will be used asymmetrically by each dimension i.e. (before, after)
         - A dictionary will set the values for the requested dimension only: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''
        processor = Padder(pad_width=pad_width, mode='linear_ramp', pad_values=end_values)
        return processor
    
    @staticmethod
    def reflect(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `reflect` 
        Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
        
        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - An integer value will pad with a border of this size in all *spatial* dimensions and directions
         - A tuple will pad with an asymmetric border in all *spatial* dimensions i.e. (before, after)
         - A dictionary will apply the specified padding in each requested dimension: e.g.
        {'horizontal':(8, 23), 'vertical': 10}      
        """
        processor = Padder(pad_width=pad_width, mode='reflect')
        return processor
    
    @staticmethod
    def symmetric(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `symmetric`
        Pads with the reflection of the vector mirrored along the edge of the array.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - An integer value will pad with a border of this size in all *spatial* dimensions and directions
         - A tuple will pad with an asymmetric border in all *spatial* dimensions i.e. (before, after)
         - A dictionary will apply the specified padding in each requested dimension: e.g.
        {'horizontal':(8, 23), 'vertical': 10}     
        """
        processor = Padder(pad_width=pad_width, mode='symmetric')
        return processor
    
    @staticmethod
    def wrap(pad_width=None):
        """
        Padder processor wrapping numpy.pad with mode `wrap`
        Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The size of the border along each axis

        Notes
        -----
        `pad_width` behaviour (number of pixels):
         - An integer value will pad with a border of this size in all *spatial* dimensions and directions
         - A tuple will pad with an asymmetric border in all *spatial* dimensions i.e. (before, after)
         - A dictionary will apply the specified padding in each requested dimension: e.g.
        {'horizontal':(8, 23), 'vertical': 10}     
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

        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        if (data.geometry == None):
            raise ValueError('Geometry is not defined.')

        if self.mode not in ['constant', 'edge', 'linear_ramp', 'reflect', 'symmetric', 'wrap']:
            raise Exception("Wrong mode. One of the following is expected:\n" +
                            "constant, edge, linear_ramp, reflect, symmetric, wrap")

        if (self.pad_width == None):
            raise ValueError('Please, specify pad_width')


        self._parse_input(data)
    
        return True 

    def _create_tuple(self, value):
        try:
            out = (int(value),int(value))
        except TypeError:
            try:
                out = (int(value[0]),int(value[1]))
            except:
                raise TypeError()

        return out

    def _parse_input(self, data):

        offset = 4-data.ndim

        self._labels_in = [None]*4
        self._labels_in[offset::] = data.dimension_labels

        self._shape_in = [1]*4
        self._shape_in[offset::] = data.shape

        self._shape_out = self._shape_in.copy()

        self._processed_dims = [0,0,0,0]

        self._pad_width_param = [(0,0)]*4
        self._pad_values_param = [(0,0)]*4


        # apply to spatial dimensions only by default
        # dimensions to process
        spatial_dimensions =[
            'vertical',
            'horizontal',
            'horizontal_y',
            'horizontal_x',
            'channel'
        ]
        
        if isinstance(self.pad_width, dict):
            for k, v in self.pad_width.items():
                if k == 'angle':            
                    raise NotImplementedError('Cannot use Padder to pad the angle dimension')

                try:
                    i = self._labels_in.index(k)
                except:
                    raise ValueError('Dimension label not found in data. Expected labels from {0}. Got {1}'.format(data.dimension_labels, k))

                try:
                    self._pad_width_param[i] = self._create_tuple(v)
                except TypeError:
                    raise TypeError("`pad_width` for axis {0} should be a integer or a tuple of integers. Got {1}".format(k, v))    

        else:
            try:
                pad = self._create_tuple(self.pad_width)
            except TypeError:
                raise TypeError("`pad_width` should be a integer or a tuple of integers. Got {0}".format(v))    

            for i, dim in enumerate(self._labels_in):
                if dim in spatial_dimensions:
                    self._pad_width_param[i] = pad

        # create list of processed axes and new_shape
        for i in range(4):
           if self._pad_width_param[i] != (0,0):
                self._processed_dims[i] = 1 
                self._shape_out[i] += self._pad_width_param[i][0] + self._pad_width_param[i][1]


        for i, dim in enumerate(self._labels_in):
            # only get values for axes we're processing
            if self._processed_dims[i]:
                try:
                    # if passed a dictionary must have values for each padded axis
                    values = self.pad_values[dim]
                except KeyError:
                    raise KeyError("Corresponding value not found for padded axis {0}".format(dim))
                except TypeError:
                    values = self.pad_values

                try:
                    self._pad_values_param[i] = self._create_tuple(values)
                except TypeError:
                    raise TypeError("`pad_values` should be a integer or a tuple of integers. Got {0} for axis {1}".format(v, dim))    


    def _process_acquisition_geometry(self):
        """
        Creates the new acquisition geometry
        """

        geometry = self._geometry.copy()
        for i, dim in enumerate(self._labels_in):

            if not self._processed_dims[i]:
                continue

            offset = (self._pad_width_param[i][0] -self._pad_width_param[i][1])*0.5
            system_detector = geometry.config.system.detector

            if dim == 'channel':
                geometry.set_channels(num_channels= geometry.config.channels.num_channels + \
                self._pad_width_param[i][0] + self._pad_width_param[i][1])
            elif dim == 'angle':
                raise NotImplementedError('Cannot use Padder to pad in the angle dimension')
            elif dim == 'vertical':
                geometry.config.panel.num_pixels[1] += self._pad_width_param[i][0] 
                geometry.config.panel.num_pixels[1] += self._pad_width_param[i][1]
                system_detector.position = system_detector.position - offset * system_detector.direction_y * geometry.config.panel.pixel_size[1]
            elif dim == 'horizontal':
                geometry.config.panel.num_pixels[0] += self._pad_width_param[i][0]
                geometry.config.panel.num_pixels[0] += self._pad_width_param[i][1]
                system_detector.position = system_detector.position - offset * system_detector.direction_x * geometry.config.panel.pixel_size[0]
        
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
                geometry.center_z += offset * geometry.voxel_size_z
            elif dim == 'horizontal_x':
                geometry.voxel_num_x += self._pad_width_param[i][0]
                geometry.voxel_num_x += self._pad_width_param[i][1]
                geometry.center_x += offset * geometry.voxel_size_x
            elif dim == 'horizontal_y':
                geometry.voxel_num_y += self._pad_width_param[i][0]
                geometry.voxel_num_y += self._pad_width_param[i][1]
                geometry.center_y += offset * geometry.voxel_size_y

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
                out.array = out.array.reshape(self._shape_out)
            except:
                raise ValueError("Array of `out` not compatible. Expected shape: {0}, data type: {1} Got shape: {2}, data type: {3}".format(self._shape_out, np.float32, out.array.shape, out.array.dtype))

            if new_geometry is not None:
                if out.geometry != new_geometry:
                    raise ValueError("Geometry of `out` not as expected. Got {0}, expected {1}".format(out.geometry, new_geometry))
            
            out.array = self._process_data(data)
