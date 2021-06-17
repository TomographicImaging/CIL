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
import warnings
import numpy


class Padder(DataProcessor):
    r'''
    Processor to pad an array. Please use the desiried method to configure a processor for your needs.
    '''

    @staticmethod
    def constant(pad_width=None, constant_values=0):
        r'''This creates a Padder processor which pads with a constant value.
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        :param constant_values: The values to set the padded values for each axis, specified as a dictionary containing a tuple (before, after). A tuple (before, after) yields same before and after pad for each axis. float is a shortcut for before = after = pad value for all axes. Default is 0.
        :type constant_values: float or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='constant', constant_values=constant_values)
        return processor

    @staticmethod
    def edge(pad_width=None):
        r'''This creates a Padder processor which pads with the edge values of array.
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='edge')
        return processor
    
    @staticmethod
    def linear_ramp(pad_width=None, end_values=0):
        r'''This creates a Padder processor which pads with the linear ramp between end_value and the array edge value.
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        :param end_values: The values used for the ending value of the linear_ramp and that will form the edge of the padded array., specified as a dictionary containing a tuple (before, after). A tuple (before, after) yields same before and after pad for each axis. float is a shortcut for before = after = pad value for all axes. Default is 0.
        :type end_values: float or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='linear_ramp', end_values=end_values)
        return processor
    
    @staticmethod
    def reflect(pad_width=None):
        r'''This creates a Padder processor which pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='reflect')
        return processor
    
    @staticmethod
    def symmetric(pad_width=None):
        r'''This creates a Padder processor which pads with the reflection of the vector mirrored along the edge of the array.
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='symmetric')
        return processor
    
    @staticmethod
    def wrap(pad_width=None):
        r'''This creates a Padder processor which pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        '''
        processor = Padder(pad_width=pad_width, mode='wrap')
        return processor

    def __init__(self,
                 mode='constant',
                 pad_width=None,
                 constant_values=0,
                 end_values=0):
        r'''
        Processor to pad an array. 
        :param pad_width: number of values padded to the edges of each axis, specified as a dictionary containing a tuple (before, after) or an int (before=after). A tuple (before, after) yields same before and after pad for each axis. int is a shortcut for before = after = pad width for all axes.
        :type pad_width: int or tuple or dict
        :param constant_values: Used in 'constant' mode. The values to set the padded values for each axis, specified as a dictionary containing a tuple (before, after). A tuple (before, after) yields same before and after pad for each axis. float is a shortcut for before = after = pad value for all axes. Default is 0.
        :type constant_values: float or tuple or dict
        :param edge_values: Used in 'linear_ramp'. The values used for the ending value of the linear_ramp and that will form the edge of the padded array., specified as a dictionary containing a tuple (before, after). A tuple (before, after) yields same before and after pad for each axis. float is a shortcut for before = after = pad value for all axes. Default is 0.
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
        geometry = geometry_0.copy()

        res, pad_width_param = self._parse_param(data, self.pad_width, ndim, dimension_labels)
        
        if res == False:
            raise ValueError('Cannot parse provided pad_width. Please, provide int, tuple or dictionary with dimension lables.')

        res, constant_values_param = self._parse_param(data, self.constant_values, ndim, dimension_labels)

        if res == False:
            raise ValueError('Cannot parse provided constant_values. Please, provide float, tuple or dictionary with dimension lables.')
        
        res, end_values_param = self._parse_param(data, self.end_values, ndim, dimension_labels)

        if res == False:
            raise ValueError('Cannot parse provided end_values. Please, provide float, tuple or dictionary with dimension lables.')

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
                    geometry.set_channels(num_channels=geometry_0.config.channels.num_channels+pad_width_param[dim][0]+pad_width_param[dim][1])
                elif dimension_labels[dim] == 'angle':
                    # pad angles vector
                    initial_angle = geometry_0.config.angles.initial_angle
                    angle_unit = geometry_0.config.angles.angle_unit
                    angles_0 = geometry_0.config.angles.angle_data
                    if pad_width_param[dim] != (0,0):
                        if self.mode in ['reflect', 'symmetric', 'wrap', 'edge']:
                            angles = numpy.pad(angles_0, pad_width_param[dim], mode=self.mode)
                        elif self.mode == 'constant':
                            angles = numpy.pad(angles_0, pad_width_param[dim], mode=self.mode, constant_values=constant_values_param[dim])
                        elif self.mode == 'linear_ramp':
                            angles = numpy.pad(angles_0, pad_width_param[dim], mode=self.mode, end_values=end_values_param[dim])
                    else:
                        angles = angles_0.copy()
                    geometry.set_angles(angles, initial_angle=initial_angle, angle_unit=angle_unit)
                elif dimension_labels[dim] == 'vertical':
                    geometry.config.panel.num_pixels[1] += pad_width_param[dim][0]
                    geometry.config.panel.num_pixels[1] += pad_width_param[dim][1]
                elif dimension_labels[dim] == 'horizontal':
                    geometry.config.panel.num_pixels[0] += pad_width_param[dim][0]
                    geometry.config.panel.num_pixels[0] += pad_width_param[dim][1]
        
        data_padded = geometry.allocate()
        
        if self.mode in ['reflect', 'symmetric', 'wrap', 'edge']:
            data_padded.fill(numpy.pad(data.as_array(), pad_width_param, mode=self.mode))
        elif self.mode == 'constant':
            data_padded.fill(numpy.pad(data.as_array(), pad_width_param, mode=self.mode, constant_values=constant_values_param))
        elif self.mode == 'linear_ramp':
            data_padded.fill(numpy.pad(data.as_array(), pad_width_param, mode=self.mode, end_values=end_values_param))
        
        if out == None:
            return data_padded
        else:
            out = data_padded


    def _parse_param(self, data, param, ndim, dimension_labels):

        res = True
        pad_param = []

        if not hasattr(param, "__len__"):
            pad_param = [(param, param)] * ndim
        elif isinstance(param, tuple) and len(param) == 2:
            pad_param = [param] * ndim
        elif isinstance(param, dict):
            pad_param = [(0,0)] * ndim
            for key in param.keys():
                idx = data.dimension_labels.index(key)
                if not hasattr(param[key], "__len__"):
                    pad_param[idx] = (param[key], param[key])
                elif isinstance(param[key], tuple) and len(param[key]) == 2:
                    pad_param[idx] = param[key]
        else:
            res = False

        return res, pad_param

