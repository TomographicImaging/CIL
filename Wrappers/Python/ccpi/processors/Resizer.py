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


class Resizer(DataProcessor):

    def __init__(self,
                 roi = None,
                 binning = None):
        
        '''
        Constructor
        
        Input:
            
            roi             region-of-interest to crop, specified as a dictionary
                            containing axis lables and tuples. 
                            
            binning         number of pixels to bin (combine), specified as a dictionary
                            containing axis lables and int. 
        '''

        kwargs = {'roi': roi,
                  'binning': binning}

        super(Resizer, self).__init__(**kwargs)
    
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
                    raise Exception('Wrong label is specified for roi')
        
        if self.binning != None:
            for key in self.binning.keys():
                if key not in data.dimension_labels.values():
                    raise Exception('Wrong label is specified for binning')
        
        roi = [-1] * ndim
        binning = [1] * ndim
        
        if self.roi != None:
            for key in data.dimension_labels.keys():
                if data.dimension_labels[key] in self.roi.keys() and self.roi[data.dimension_labels[key]] != -1:
                    roi[key] = self.roi[data.dimension_labels[key]]
        
        if self.binning != None:
            for key in data.dimension_labels.keys():
                print(data.dimension_labels[key])
                print(self.binning.keys())
                if data.dimension_labels[key] in self.binning.keys():
                    binning[key] = self.binning[data.dimension_labels[key]]

        if (isinstance(data, ImageData)):
            for key in data.dimension_labels:
                if data.dimension_labels[key] == 'channel':
                    if (roi[key] != -1):
                        geometry.channels = (roi[key][1] - roi[key][0]) // binning[key]
                        roi[key] = (roi[key][0], roi[key][0] + ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.channels = geometry_0.channels // binning[key]
                        roi[key] = (0, geometry.channels * binning[key])
                elif data.dimension_labels[key] == 'horizontal_y':
                    if (roi[key] != -1):
                        geometry.voxel_num_y = (roi[key][1] - roi[key][0]) // binning[key]
                        geometry.voxel_size_y = geometry_0.voxel_size_y * binning[key]
                        roi[key] = (roi[key][0], roi[key][0] + ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.voxel_num_y = geometry_0.voxel_num_y // binning[key]
                        geometry.voxel_size_y = geometry_0.voxel_size_y * binning[key]
                        roi[key] = (0, geometry.voxel_num_y * binning[key])
                elif data.dimension_labels[key] == 'vertical':
                    if (roi[key] != -1):
                        geometry.voxel_num_z = (roi[key][1] - roi[key][0]) // binning[key]
                        geometry.voxel_size_z = geometry_0.voxel_size_z * binning[key]
                        roi[key] = (roi[key][0], roi[key][0] + ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.voxel_num_z = geometry_0.voxel_num_z // binning[key]
                        geometry.voxel_size_z = geometry_0.voxel_size_z * binning[key]
                        roi[key] = (0, geometry.voxel_num_z * binning[key])
                elif data.dimension_labels[key] == 'horizontal_x':
                    if (roi[key] != -1):
                        geometry.voxel_num_x = (roi[key][1] - roi[key][0]) // binning[key]
                        geometry.voxel_size_x = geometry_0.voxel_size_x * binning[key]
                        roi[key] = (roi[key][0], roi[key][0]+ ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.voxel_num_x = geometry_0.voxel_num_x // binning[key]
                        geometry.voxel_size_x = geometry_0.voxel_size_x * binning[key]
                        roi[key] = (0, geometry.voxel_num_x * binning[key])

        else: # AcquisitionData
            for key in data.dimension_labels:
                if data.dimension_labels[key] == 'channel':
                    if (roi[key] != -1):
                        geometry.channels = (roi[key][1] - roi[key][0]) // binning[key]
                        roi[key] = (roi[key][0], roi[key][0] + ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.channels = geometry_0.channels // binning[key]
                        roi[key] = (0, geometry.channels * binning[key])
                elif data.dimension_labels[key] == 'angle':
                    if (roi[key] != -1):
                        geometry.angles = geometry_0.angles[roi[key][0]:roi[key][1]]
                    else:
                        geometry.angles = geometry_0.angles
                        roi[key] = (0, len(geometry.angles))
                    if (binning[key] != 1):
                        binning[key] = 1
                        warnings.warn('Rebinning in angular dimensions is not supported: \n binning[{}] is set to 1.'.format(key))
                elif data.dimension_labels[key] == 'vertical':
                    if (roi[key] != -1):
                        geometry.pixel_num_v = (roi[key][1] - roi[key][0]) // binning[key]
                        geometry.pixel_size_v = geometry_0.pixel_size_v * binning[key]
                        roi[key] = (roi[key][0], roi[key][0] + ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.pixel_num_v = geometry_0.pixel_num_v // binning[key]
                        geometry.pixel_size_v = geometry_0.pixel_size_v * binning[key]
                        roi[key] = (0, geometry.pixel_num_v * binning[key])
                elif data.dimension_labels[key] == 'horizontal':
                    if (roi[key] != -1):
                        geometry.pixel_num_h = (roi[key][1] - roi[key][0]) // binning[key]
                        geometry.pixel_size_h = geometry_0.pixel_size_h * binning[key]
                        roi[key] = (roi[key][0], roi[key][0] + ((roi[key][1] - roi[key][0]) // binning[key]) * binning[key])
                    else:
                        geometry.pixel_num_h = geometry_0.pixel_num_h // binning[key]
                        geometry.pixel_size_h = geometry_0.pixel_size_h * binning[key]
                        roi[key] = (0, geometry.pixel_num_h * binning[key])
                            
        if ndim == 2:
            n_pix_0 = (roi[0][1] - roi[0][0]) // binning[0]
            n_pix_1 = (roi[1][1] - roi[1][0]) // binning[1]
            shape = (n_pix_0, binning[0], 
                     n_pix_1, binning[1])
            data_resized = data.as_array()[roi[0][0]:(roi[0][0] + n_pix_0 * binning[0]), 
                                           roi[1][0]:(roi[1][0] + n_pix_1 * binning[1])].reshape(shape).mean(-1).mean(1)
        if ndim == 3:
            n_pix_0 = (roi[0][1] - roi[0][0]) // binning[0]
            n_pix_1 = (roi[1][1] - roi[1][0]) // binning[1]
            n_pix_2 = (roi[2][1] - roi[2][0]) // binning[2]
            shape = (n_pix_0, binning[0], 
                     n_pix_1, binning[1],
                     n_pix_2, binning[2])
            data_resized = data.as_array()[roi[0][0]:(roi[0][0] + n_pix_0 * binning[0]), 
                                           roi[1][0]:(roi[1][0] + n_pix_1 * binning[1]), 
                                           roi[2][0]:(roi[2][0] + n_pix_2 * binning[2])].reshape(shape).mean(-1).mean(1).mean(2)
        if ndim == 4:
            n_pix_0 = (roi[0][1] - roi[0][0]) // binning[0]
            n_pix_1 = (roi[1][1] - roi[1][0]) // binning[1]
            n_pix_2 = (roi[2][1] - roi[2][0]) // binning[2]
            n_pix_3 = (roi[3][1] - roi[3][0]) // binning[3]
            shape = (n_pix_0, binning[0], 
                     n_pix_1, binning[1],
                     n_pix_2, binning[2],
                     n_pix_3, binning[3])
            data_resized = data.as_array()[roi[0][0]:(roi[0][0] + n_pix_0 * binning[0]), 
                                           roi[1][0]:(roi[1][0] + n_pix_1 * binning[1]), 
                                           roi[2][0]:(roi[2][0] + n_pix_2 * binning[2]), 
                                           roi[3][0]:(roi[3][0] + n_pix_3 * binning[3])].reshape(shape).mean(-1).mean(1).mean(2).mean(3)

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
              binning = {'horizontal': 1},
              roi = {'vertical': -1},
              fliplr = False)

data = reader.load_projections()
print(data)
ag = reader.get_geometry()
print(ag)

plt.imshow(data.as_array()[1, :, :])
plt.show()

resizer = Resizer(binning = {'horizontal': 2}, 
                  roi = {'vertical': (200,400)})
resizer.input = data
data_resized = resizer.process()

plt.imshow(data_resized.as_array()[1, :, :])
plt.show()

print(data_resized)
'''
