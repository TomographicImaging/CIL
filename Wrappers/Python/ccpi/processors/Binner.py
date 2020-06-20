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
import warnings


class Binner(DataProcessor):

    def __init__(self,
                 roi = None):
        
        '''
        Constructor
        
        Input:
            
            roi             region-of-interest to crop, specified as a dictionary
                            containing tuple (Start, Stop, Step)
        '''

        kwargs = {'roi': roi}

        super(Binner, self).__init__(**kwargs)
    
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
        
        if self.roi != None:
            for key in self.roi.keys():
                if key not in data.dimension_labels.values():
                    raise ValueError('Wrong label is specified for roi')
        
        roi = []
        for i in range(ndim):
            roi.append([0, data.shape[i], 1])
            
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
                if (isinstance(data, ImageData)):
                    if key == 'channel':
                        geometry.channels = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                    elif key == 'horizontal_x':
                        geometry.voxel_num_x = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                        geometry.voxel_size_x = geometry_0.voxel_size_x * roi[idx][2]
                    elif key == 'horizontal_y':
                        geometry.voxel_num_y = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                        geometry.voxel_size_y = geometry_0.voxel_size_y * roi[idx][2]
                    elif key == 'vertical':
                        geometry.voxel_num_z = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                        geometry.voxel_size_z = geometry_0.voxel_size_z * roi[idx][2]
                else: #AcquisitionData
                    if key == 'channel':
                        geometry.channels = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                    elif key == 'horizontal':
                        geometry.pixel_num_h = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                        geometry.pixel_size_h = geometry_0.pixel_size_h * roi[idx][2]
                    elif key == 'vertical':
                        geometry.pixel_num_v = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                        geometry.pixel_size_v = geometry_0.pixel_size_v * roi[idx][2]
                    elif key == 'angle':
                        n_elem = (roi[idx][1] - roi[idx][0]) // roi[idx][2]
                        shape = (n_elem, roi[idx][2])
                        geometry.angles = geometry_0.angles[roi[idx][0]:(roi[idx][0] + n_elem * roi[idx][2])].reshape(shape).mean(1)
                
        if ndim == 2:
            n_pix_0 = (roi[0][1] - roi[0][0]) // roi[0][2]
            n_pix_1 = (roi[1][1] - roi[1][0]) // roi[1][2]
            shape = (n_pix_0, roi[0][2], 
                     n_pix_1, roi[1][2])
            data_resized = data.as_array()[roi[0][0]:(roi[0][0] + n_pix_0 * roi[0][2]), 
                                           roi[1][0]:(roi[1][0] + n_pix_1 * roi[1][2])].reshape(shape).mean(-1).mean(1)
        if ndim == 3:
            n_pix_0 = (roi[0][1] - roi[0][0]) // roi[0][2]
            n_pix_1 = (roi[1][1] - roi[1][0]) // roi[1][2]
            n_pix_2 = (roi[2][1] - roi[2][0]) // roi[2][2]
            shape = (n_pix_0, roi[0][2], 
                     n_pix_1, roi[1][2],
                     n_pix_2, roi[2][2])
            data_resized = data.as_array()[roi[0][0]:(roi[0][0] + n_pix_0 * roi[0][2]), 
                                           roi[1][0]:(roi[1][0] + n_pix_1 * roi[1][2]), 
                                           roi[2][0]:(roi[2][0] + n_pix_2 * roi[2][2])].reshape(shape).mean(-1).mean(1).mean(2)
        if ndim == 4:
            n_pix_0 = (roi[0][1] - roi[0][0]) // roi[0][2]
            n_pix_1 = (roi[1][1] - roi[1][0]) // roi[1][2]
            n_pix_2 = (roi[2][1] - roi[2][0]) // roi[2][2]
            n_pix_3 = (roi[3][1] - roi[3][0]) // roi[3][2]
            shape = (n_pix_0, roi[0][2], 
                     n_pix_1, roi[1][2],
                     n_pix_2, roi[2][2],
                     n_pix_3, roi[3][2])
            data_resized = data.as_array()[roi[0][0]:(roi[0][0] + n_pix_0 * roi[0][2]), 
                                           roi[1][0]:(roi[1][0] + n_pix_1 * roi[1][2]), 
                                           roi[2][0]:(roi[2][0] + n_pix_2 * roi[2][2]), 
                                           roi[3][0]:(roi[3][0] + n_pix_3 * roi[3][2])].reshape(shape).mean(-1).mean(1).mean(2).mean(3)

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

resizer = Binner(roi = {'vertical': (-1500, -1000), 
                     'horizontal': (None, None, 5),
                     'angle': (None, None, 2)})
resizer.input = data
data_resized = resizer.process()

plt.imshow(data_resized.as_array()[1, :, :])
plt.show()

print(data.geometry.angles)
print(data_resized.geometry.angles)
'''
