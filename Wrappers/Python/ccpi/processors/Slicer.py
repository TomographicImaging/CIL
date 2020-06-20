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


class Slicer(DataProcessor):

    def __init__(self,
                 roi = None):
        
        '''
        Constructor
        
        Input:
            
            roi             region-of-interest to crop, specified as a dictionary
                            containing tuple (Start, Stop, Step)
        '''

        kwargs = {'roi': roi}

        super(Slicer, self).__init__(**kwargs)
    
    def check_input(self, data):
        
        if not ((isinstance(data, ImageData)) or 
                (isinstance(data, AcquisitionData))):
            raise ValueError('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        elif (data.geometry == None):
            raise ValueError('Geometry is not defined.')
        else:
            return True 
    
    def process(self):

        data = self.get_input()
        ndim = len(data.dimension_labels)
        
        geometry_0 = data.geometry
        geometry = geometry_0.clone()
        
        if self.roi != None:
            for key in self.roi.keys():
                if key not in data.dimension_labels.values():
                    raise ValueError('Wrong label is specified for roi')
        
        roi = []
        sliceobj = []
        for i in range(ndim):
            roi.append([0, data.shape[i], 1])
            sliceobj.append(slice(None, None, None))
            
        if self.roi != None:
            for key in self.roi.keys():
                idx = data.get_dimension_axis(key)
                if (self.roi[key] != -1):
                    for i in range(2):
                        if self.roi[key][i] != None:
                            if self.roi[key][i] >= 0:
                                roi[idx][i] = self.roi[key][i]
                            else:
                                roi[idx][i] = roi[idx][1] + self.roi[key][i]
                    if len(self.roi[key]) > 2:
                        if self.roi[key][2] != None:
                            if self.roi[key][2] > 0:
                                roi[idx][2] = self.roi[key][2]
                            else:
                                raise ValueError("Negative step is not allowed")
                    sliceobj[idx] = slice(roi[idx][0], roi[idx][1], roi[idx][2])
                if (isinstance(data, ImageData)):
                    if key == 'channel':
                        geometry.channels = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                    elif key == 'horizontal_x':
                        geometry.voxel_num_x = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.voxel_size_x = geometry_0.voxel_size_x
                    elif key == 'horizontal_y':
                        geometry.voxel_num_y = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.voxel_size_y = geometry_0.voxel_size_y
                    elif key == 'vertical':
                        geometry.voxel_num_z = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.voxel_size_z = geometry_0.voxel_size_z
                else: #AcquisitionData
                    if key == 'channel':
                        geometry.channels = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                    elif key == 'horizontal':
                        geometry.pixel_num_h = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.pixel_size_h = geometry_0.pixel_size_h
                    elif key == 'vertical':
                        geometry.pixel_num_v = np.int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.pixel_size_v = geometry_0.pixel_size_v
                    elif key == 'angle':
                        geometry.angles = geometry_0.angles[sliceobj[idx]]
        
        data_resized = data.as_array()[tuple(sliceobj)]
        
        out = geometry.allocate()
        out.fill(data_resized)
        
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
