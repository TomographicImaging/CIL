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
import numpy
from numbers import Number


class Padder(DataProcessor):
    r'''
    Processor to pad an array, wrapping numpy.pad
    See https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    '''

    @staticmethod
    def constant(pad_width=None, constant_values=0):
        '''
        Padder processor wrapping numpy.pad with mode `constant` 
        Pads with a constant value.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The number of values padded to the edges of each axis
        constant_values: float, tuple, dict
            The values to set the padded values for each axis

        Notes
        -----
        If passed a single value it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''
        processor = Padder(pad_width=pad_width, mode='constant', constant_values=constant_values)
        return processor

    @staticmethod
    def edge(pad_width=None):
        '''Padder processor wrapping numpy.pad with mode `edge` 
        Pads with the edge values of array.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The number of values padded to the edges of each axis

        Notes
        -----
        If passed a single value it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''

        processor = Padder(pad_width=pad_width, mode='edge')
        return processor
    
    @staticmethod
    def linear_ramp(pad_width=None, end_values=0):
        '''Padder processor wrapping numpy.pad with mode `linear_ramp` 
        Pads with the linear ramp between end_value and the array edge value.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The number of values padded to the edges of each axis
        end_values: float, tuple, dict
            The values used for the ending value of the linear_ramp
 
        Notes
        -----
        If passed a single value it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''
        processor = Padder(pad_width=pad_width, mode='linear_ramp', end_values=end_values)
        return processor
    
    @staticmethod
    def reflect(pad_width=None):
        '''Padder processor wrapping numpy.pad with mode `reflect` 
        Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
        
        Parameters
        ----------
        pad_width: int, tuple, dict
            The number of values padded to the edges of each axis

        Notes
        -----
        If passed a single value it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''
        processor = Padder(pad_width=pad_width, mode='reflect')
        return processor
    
    @staticmethod
    def symmetric(pad_width=None):
        r'''Padder processor wrapping numpy.pad with mode `symmetric`
        Pads with the reflection of the vector mirrored along the edge of the array.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The number of values padded to the edges of each axis

        Notes
        -----
        If passed a single value it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''
        processor = Padder(pad_width=pad_width, mode='symmetric')
        return processor
    
    @staticmethod
    def wrap(pad_width=None):
        '''Padder processor wrapping numpy.pad with mode `wrap`
        Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

        Parameters
        ----------
        pad_width: int, tuple, dict
            The number of values padded to the edges of each axis

        Notes
        -----
        If passed a single value it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        '''
        processor = Padder(pad_width=pad_width, mode='wrap')
        return processor

    def __init__(self,
                 mode='constant',
                 pad_width=None,
                 constant_values=0,
                 end_values=0):

        kwargs = {'mode': mode,
                'pad_width': pad_width,
                'constant_values': constant_values,
                'end_values': end_values}

        super(Padder, self).__init__(**kwargs)


    def check_input(self, data):

        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise TypeError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        elif (data.geometry == None):
            raise ValueError('Geometry is not defined.')

        elif self.mode not in ['constant', 'edge', 'linear_ramp', 'reflect', 'symmetric', 'wrap']:
            raise Exception("Wrong mode. One of the following is expected:\n" +
                            "constant, edge, linear_ramp, reflect, symmetric, wrap")

        elif (self.pad_width == None):
            raise ValueError('Please, specify pad_width')

        else:
            return True 


    def process(self, out=None):

        data = self.get_input()
        ndim = data.number_of_dimensions
        dimension_labels = data.dimension_labels
        geometry_0 = data.geometry
        # create a new geometry for the new dataset
        geometry = geometry_0.copy()

        pad_width_param = self._parse_param(data, self.pad_width, ndim, 'pad_width')
        
        constant_values_param = self._parse_param(data, self.constant_values, ndim, \
             'constant_values')

        end_values_param = self._parse_param(data, self.end_values, ndim, 'end_values')

        for dim in range(ndim):

            if (isinstance(data, ImageData)):
                if dimension_labels[dim] == 'channel':
                    geometry.channels += pad_width_param[dim][0]
                    geometry.channels += pad_width_param[dim][1]
                elif dimension_labels[dim] == 'vertical':
                    geometry.voxel_num_z += pad_width_param[dim][0]
                    geometry.voxel_num_z += pad_width_param[dim][1]
                elif dimension_labels[dim] == 'horizontal_x':
                    geometry.voxel_num_x += pad_width_param[dim][0]
                    geometry.voxel_num_x += pad_width_param[dim][1]
                elif dimension_labels[dim] == 'horizontal_y':
                    geometry.voxel_num_y += pad_width_param[dim][0]
                    geometry.voxel_num_y += pad_width_param[dim][1]
            
            # if AcquisitionData
            else:
                if dimension_labels[dim] == 'channel':
                    geometry.set_channels(num_channels=geometry_0.config.channels.num_channels + \
                        pad_width_param[dim][0] + pad_width_param[dim][1])
                elif dimension_labels[dim] == 'angle':
                    # pad angles vector
                    pad_width_param[dim] = (0,0)
                elif dimension_labels[dim] == 'vertical':
                    geometry.config.panel.num_pixels[1] += pad_width_param[dim][0]
                    geometry.config.panel.num_pixels[1] += pad_width_param[dim][1]
                elif dimension_labels[dim] == 'horizontal':
                    geometry.config.panel.num_pixels[0] += pad_width_param[dim][0]
                    geometry.config.panel.num_pixels[0] += pad_width_param[dim][1]
        
        if out == None:
            data_padded = geometry.allocate()
        else:
            if out.geometry != geometry:
                raise ValueError('The geometry in the argument out we received is not consistent with the requested padding.')
            data_padded = out
        
        
        if self.mode in ['reflect', 'symmetric', 'wrap', 'edge']:
            data_padded.fill(numpy.pad(data.as_array(), pad_width_param, mode=self.mode))
        elif self.mode == 'constant':
            data_padded.fill(numpy.pad(data.as_array(), pad_width_param, mode=self.mode, \
                constant_values=constant_values_param))
        elif self.mode == 'linear_ramp':
            data_padded.fill(numpy.pad(data.as_array(), pad_width_param, mode=self.mode, \
                end_values=end_values_param))
        
        if out == None:
            return data_padded


    def _parse_param(self, data, param, ndim, descr):

        if isinstance(param, Number):
            pad_param = [(int(param), int(param))] * ndim
        elif isinstance(param, tuple) and len(param) == 2:
            # create a list of tuples containing 2 ints in each tuple
            pad_param = [ tuple([int(el) for el in param]) ] * ndim
        elif isinstance(param, dict):
            pad_param = [(0,0)] * ndim
            for key in param.keys():
                if key == 'angle':
                    raise NotImplementedError('Cannot use Padder to pad the angle dimension')
                idx = data.dimension_labels.index(key)
                if isinstance(param[key], Number):
                    pad_param[idx] = (int(param[key]), int(param[key]))
                elif isinstance(param[key], tuple) and len(param[key]) == 2:
                    # create a tuple containing 2 ints
                    pad_param[idx] = tuple([int(el) for el in param[key]])
                else:
                    raise ValueError('Cannot parse provided {}. Expecting a number or tuple of length 2. Got {}'\
                        .format(descr, param[key]))
        else:
            raise ValueError('Cannot parse provided {}. Expecting int, tuple or dictionary with dimension lables. Got'\
                .format(descr, type(param)))

        return pad_param

