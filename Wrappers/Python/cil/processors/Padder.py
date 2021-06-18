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
    '''

    @staticmethod
    def constant(pad_width=None, constant_values=0):
        r'''Padder processor wrapping numpy.pad with mode `linear_ramp` 
        
        :param pad_width: number of values padded to the edges of each axis.
        If passed an int it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        :type pad_width: int or tuple or dict
        :param constant_values: pads with a constant value.
        edge of the padded array. They can be passed in the same format as `pad_width`.
        :type constant_values: float or tuple or dict

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        
        
        This creates a Padder processor which pads with a constant value.

        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        :param constant_values: The values to set the padded values for each axis, specified as a dictionary containing a tuple (before, after). A tuple (before, after) yields same before and after pad for each axis. float is a shortcut for before = after = pad value for all axes. Default is 0.
        :type constant_values: float or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='constant', constant_values=constant_values)
        return processor

    @staticmethod
    def edge(pad_width=None):
        r'''Padder processor wrapping numpy.pad with mode `edge` 
        
        :param pad_width: number of values padded to the edges of each axis.
        If passed an int it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        :type pad_width: int or tuple or dict

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        '''
        processor = Padder(pad_width=pad_width, mode='edge')
        return processor
    
    @staticmethod
    def linear_ramp(pad_width=None, end_values=0):
        r'''Padder processor wrapping numpy.pad with mode `linear_ramp` 
        
        :param pad_width: number of values padded to the edges of each axis.
        If passed an int it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        :type pad_width: int or tuple or dict
        :param end_values: The values used for the ending value of the linear_ramp and that will form the 
        edge of the padded array. They can be passed in the same format as `pad_width`.
        :type end_values: float or tuple or dict

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        '''
        processor = Padder(pad_width=pad_width, mode='linear_ramp', end_values=end_values)
        return processor
    
    @staticmethod
    def reflect(pad_width=None):
        r'''Padder processor wrapping numpy.pad with mode `reflect` 
        
        :param pad_width: number of values padded to the edges of each axis.
        If passed an int it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        :type pad_width: int or tuple or dict

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        '''
        processor = Padder(pad_width=pad_width, mode='reflect')
        return processor
    
    @staticmethod
    def symmetric(pad_width=None):
        r'''Padder processor wrapping numpy.pad with mode `symmetric`
        
        :param pad_width: number of values padded to the edges of each axis.
        If passed an int it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        :type pad_width: int or tuple or dict

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

        '''
        processor = Padder(pad_width=pad_width, mode='symmetric')
        return processor
    
    @staticmethod
    def wrap(pad_width=None):
        r'''Padder processor wrapping numpy.pad with mode `wrap`
        
        :param pad_width: number of values padded to the edges of each axis.
        If passed an int it will pad symmetrically in all dimensions.
        If passed a tuple it will apply asymmetric padding in all dimensions. (before, after)
        If passed a dictionary it will apply the specified padding to the required dimension label: e.g.
        {'horizontal':(8, 23), 'vertical': 10}
        :type pad_width: int or tuple or dict

        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

        '''
        processor = Padder(pad_width=pad_width, mode='wrap')
        return processor

    def __init__(self,
                 mode='constant',
                 pad_width=None,
                 constant_values=0,
                 end_values=0):
        r'''
        Processor to pad an array wrapping numpy.pad.

        :param mode: specifies the method to use for padding. Available methods: constant, 
        edge, linear_ramp, reflect, symmetric, wrap.
        :type mode: string, default 'constant'.
        :param pad_width: number of values padded to the edges of each axis, specified as 
        a dictionary containing a tuple (before, after) or an int (before=after). The dictionary keys
        must be the dimension_labels of the DataContainer (ImageData or AcquisitionData) that one wants
        to pad.
         A tuple (before, after) yields same before and after pad for each axis. 
         int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        :param constant_values: Used in 'constant' mode. The values to set the padded values for each axis, 
        specified as a dictionary containing a tuple (before, after). 
        A tuple (before, after) yields same before and after pad for each axis. 
        float is a shortcut for before = after = pad value for all axes. Default is 0.
        :type constant_values: float or tuple or dict
        :param edge_values: Used in 'linear_ramp'. The values used for the ending value of the linear_ramp
         and that will form the edge of the padded array, specified as a dictionary containing a tuple
         (before, after). A tuple (before, after) yields same before and after pad for each axis.
         float is a shortcut for before = after = pad value for all axes. Default is 0.
        :type edge_values: float or tuple or dict
        '''

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

