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
                 pad_width = None):
        
        '''
        Constructor
        
        Input:
            
            pad_width       number of pixels to pad, specified as a dictionary
                            containing tuple (before, after)
        '''

        kwargs = {'pad_width': pad_width}

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
            for key in self.pad_width.keys():
                if key not in data.dimension_labels.values():
                    raise Exception('Wrong label is specified for pad_width')
        
        pad_width = [1] * ndim
            
        if self.pad_width != None:
            for key in self.pad_width.keys():
                idx = data.get_dimension_axis(key)
                if isinstance(self.pad_width[key], int):
                    pad_width[idx] = (self.pad_width[key],self.pad_width[key])
                elif len(self.pad_width[key]) == 2:
                    tmp = [0,0]
                    for i in range(2):
                        if self.pad_width[key][i] != None:
                            tmp[i] = self.pad_width[key][i]
                    pad_width[idx] = tuple(tmp)
                else:
                    raise Exception('Wrong pad_width parameter is specified. Excpected int or tuple(int,int)')
                if (isinstance(data, ImageData)):
                    if key == 'channel':
                        geometry.channels += pad_width[idx][0]+pad_width[idx][1]
                    elif key == 'horizontal_x':
                        geometry.voxel_num_x += pad_width[idx][0]+pad_width[idx][1]
                    elif key == 'horizontal_y':
                        geometry.voxel_num_y += pad_width[idx][0]+pad_width[idx][1]
                    elif key == 'vertical':
                        geometry.voxel_num_z += pad_width[idx][0]+pad_width[idx][1]
                else: #AcquisitionData
                    if key == 'channel':
                        geometry.channels += pad_width[idx][0]+pad_width[idx][1]
                    elif key == 'horizontal':
                        geometry.pixel_num_h += pad_width[idx][0]+pad_width[idx][1]
                    elif key == 'vertical':
                        geometry.pixel_num_v += pad_width[idx][0]+pad_width[idx][1]
                    elif key == 'angle':
                        geometry.angles = np.pad(geometry_0.angles, tuple(pad_width[idx]))

        pad_width = tuple(pad_width)

        data_resized = np.pad(data.as_array(),pad_width)
        
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

resizer = Slicer(roi = {'vertical': (-1500, -1000), 
                     'horizontal': (None, None, 5),
                     'angle': (None, None, 2)})
resizer.input = data
data_resized = resizer.process()

plt.imshow(data_resized.as_array()[1, :, :])
plt.show()

print(data.geometry.angles)
print(data_resized.geometry.angles)
'''
