# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.framework import DataProcessor, AcquisitionData, ImageData
import numpy as np
import warnings


class Padder(DataProcessor):

    def __init__(self,
                 pad_width = None,
                 mode = 'constant',
                 stat_length = None,
                 constant_values = None,
                 end_values = None,
                 reflect_type='even'):
        
        '''
        Constructor
        
        Input:
            
            pad_width       number of pixels to pad, specified as a dictionary
                            containing tuple (before, after)
                            shortcuts: if pad_width is specified as a single int, then all
                            axes will be symmetrically padded by this int; 
                            if pad_width is pecified as tuple, then all axes will 
                            be padded by this tuple
        '''

        kwargs = {'pad_width': pad_width,
                  'mode': mode,
                  'stat_length': stat_length,
                  'constant_values': constant_values,
                  'end_values': end_values,
                  'reflect_type': reflect_type}

        super(Padder, self).__init__(**kwargs)
    
    def check_input(self, data):
        
        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise Exception('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        elif (data.geometry == None):
            raise Exception('Geometry is not defined.')
        else:
            return True 
    
    def process(self):

        data = self.get_input()
        ndim = len(data.dimension_labels)
        
        geometry_0 = data.geometry
        geometry = geometry_0.clone()
        
        if self.pad_width != None:
            if isinstance(self.pad_width, dict):
                for key in self.pad_width.keys():
                    if key not in data.dimension_labels.values():
                        raise Exception('Wrong label is specified for pad_width')
        
        if self.stat_length != None:
            if isinstance(self.stat_length, dict):
                for key in self.stat_length.keys():
                    if key not in data.dimension_labels.values():
                        raise Exception('Wrong label is specified for stat_length')
        
        if self.constant_values != None:
            if isinstance(self.constant_values, dict):
                for key in self.constant_values.keys():
                    if key not in data.dimension_labels.values():
                        raise Exception('Wrong label is specified for constant_values') 
        
        if self.end_values != None:
            if isinstance(self.end_values, dict):
                for key in self.end_values.keys():
                    if key not in data.dimension_labels.values():
                        raise Exception('Wrong label is specified for end_values') 
        
        if self.reflect_type not in ['even', 'odd']:
            raise Exception('Wrong reflect_type, even or odd is expected')
        
        if self.constant_values != None and not(isinstance(self.constant_values, int) or isinstance(self.constant_values, tuple)):
            constant_values = []
            for i in range(ndim):
                constant_values.append((0, 0))
            for key in self.constant_values.keys():
                idx = data.get_dimension_axis(key)
                if isinstance(self.constant_values[key], int):
                    constant_values[idx] = (self.constant_values[key], self.constant_values[key])
                elif isinstance(self.constant_values[key], tuple):
                    tmp = [0,0]
                    for i in range(2):
                        if self.constant_values[key][i] != None:
                            tmp[i] = self.constant_values[key][i]
                    constant_values[idx] = tuple(tmp)
                else:
                    raise Exception('Wrong constant_values parameter is specified. Excpected int or tuple(int,int)')
            constant_values = tuple(constant_values)
        else:
            constant_values = self.constant_values
        
        if self.end_values != None and not(isinstance(self.end_values, int) or isinstance(self.end_values, tuple)):
            end_values = []
            for i in range(ndim):
                end_values.append((0, 0))
            for key in self.end_values.keys():
                idx = data.get_dimension_axis(key)
                if isinstance(self.end_values[key], int):
                    end_values[idx] = (self.end_values[key], self.end_values[key])
                elif isinstance(self.end_values[key], tuple):
                    tmp = [0,0]
                    for i in range(2):
                        if self.end_values[key][i] != None:
                            tmp[i] = self.end_values[key][i]
                    end_values[idx] = tuple(tmp)
                else:
                    raise Exception('Wrong end_values parameter is specified. Excpected int or tuple(int,int)')
            end_values = tuple(end_values)
        else:
            end_values = self.end_values
            
        if self.stat_length != None and not(isinstance(self.stat_length, int) or isinstance(self.stat_length, tuple)):
            stat_length = []
            for i in range(ndim):
                stat_length.append((data.shape[i], data.shape[i]))
            for key in self.stat_length.keys():
                idx = data.get_dimension_axis(key)
                if isinstance(self.stat_length[key], int):
                    stat_length[idx] = (self.stat_length[key], self.stat_length[key])
                elif isinstance(self.stat_length[key], tuple):
                    tmp = [0,0]
                    for i in range(2):
                        if self.stat_length[key][i] != None:
                            tmp[i] = self.stat_length[key][i]
                    stat_length[idx] = tuple(tmp)
                else:
                    raise Exception('Wrong pad_width parameter is specified. Excpected int or tuple(int,int)')
            stat_length = tuple(stat_length)
        else:
            stat_length = self.stat_length
        
        pad_param = []
        for i in range(ndim):
            pad_param.append((0,0))
        if self.pad_width != None:
            # if shortcut, construct dictionary
            if isinstance(self.pad_width, int):
                pad_width = {}
                for value in data.dimension_labels.values():
                    pad_width[value] = (self.pad_width, self.pad_width)
            elif isinstance(self.pad_width, tuple):
                pad_width = {}
                for value in data.dimension_labels.values():
                    pad_width[value] = self.pad_width
            else:
                pad_width = self.pad_width
   
            for key in pad_width.keys():
                idx = data.get_dimension_axis(key)
                if isinstance(pad_width[key], int):
                    pad_param[idx] = (pad_width[key], pad_width[key])
                elif isinstance(pad_width[key], tuple):
                    tmp = [0,0]
                    for i in range(2):
                        if pad_width[key][i] != None:
                            tmp[i] = pad_width[key][i]
                    pad_param[idx] = tuple(tmp)
                else:
                    raise Exception('Wrong pad_width parameter is specified. Excpected int or tuple(int,int)')
                if (isinstance(data, ImageData)):
                    if key == 'channel':
                        geometry.channels += pad_param[idx][0]+pad_param[idx][1]
                    elif key == 'horizontal_x':
                        geometry.voxel_num_x += pad_param[idx][0]+pad_param[idx][1]
                    elif key == 'horizontal_y':
                        geometry.voxel_num_y += pad_param[idx][0]+pad_param[idx][1]
                    elif key == 'vertical':
                        geometry.voxel_num_z += pad_param[idx][0]+pad_param[idx][1]
                else: #AcquisitionData
                    if key == 'channel':
                        geometry.channels += pad_param[idx][0]+pad_param[idx][1]
                    elif key == 'horizontal':
                        geometry.pixel_num_h += pad_param[idx][0]+pad_param[idx][1]
                    elif key == 'vertical':
                        geometry.pixel_num_v += pad_param[idx][0]+pad_param[idx][1]
                    elif key == 'angle':
                        geometry.angles = np.pad(geometry_0.angles, tuple(pad_param[idx]))
        pad_param = tuple(pad_param)
        
        if self.mode in ['maximum', 'minimum', 'median', 'mean'] and stat_length is not None:
            data_resized = np.pad(data.as_array(), 
                                  pad_param, 
                                  mode=self.mode,
                                  stat_length=stat_length)
        elif self.mode == 'constant' and constant_values is not None:
            data_resized = np.pad(data.as_array(), 
                                  pad_param, 
                                  mode=self.mode,
                                  constant_values=constant_values)
        elif self.mode == 'linear_ramp' and end_values is not None:
            data_resized = np.pad(data.as_array(), 
                                  pad_param, 
                                  mode=self.mode,
                                  end_values=end_values)
        elif self.mode in ['reflect', 'symmetric']:
            data_resized = np.pad(data.as_array(), 
                                  pad_param, 
                                  mode=self.mode,
                                  reflect_type=self.reflect_type)
        else:
            data_resized = np.pad(data.as_array(), 
                                  pad_param, 
                                  mode=self.mode)
        
        out = type(data)(array = data_resized, 
                         deep_copy = False,
                         dimension_labels = data.dimension_labels,
                         geometry = geometry)
        
        return out


'''

# usage example
from ccpi.processors import Resizer
from ccpi.io import NikonDataReader
import matplotlib.pyplot as plt

xtek_file = '/media/newhd/shared/Data/SophiaBeads/SophiaBeads_256_averaged/SophiaBeads_256_averaged.xtekct'
reader = NikonDataReader()
reader.set_up(xtek_file = xtek_file,
              normalize = True,
              roi = {'vertical': -1, 
                     'horizontal': -1},
              fliplr = False)

data = reader.load_projections()
#print(data)
ag = reader.get_geometry()
#print(ag)

plt.imshow(data.as_array()[1, :, :])
plt.show()

padder = Padder(pad_width = {'vertical': 100},
                mode = 'linear_ramp',
                end_values = {'vertical': 100})

padder.input = data
data_resized = padder.process()

plt.imshow(data_resized.as_array()[120, :, :])
plt.show()

padder = Padder(pad_width = 100,
                mode = 'constant',
                constant_values = {'vertical': (0,100), 'horizontal': 1000})

padder.input = data
data_resized = padder.process()

plt.imshow(data_resized.as_array()[120, :, :])
plt.show()
'''
