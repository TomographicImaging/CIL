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
from ccpi.framework import DataProcessor, AcquisitionData, ImageData
import warnings


class Resizer(DataProcessor):

    def __init__(self,
                 roi = -1,
                 binning = 1):
        
        '''
        Constructor
        
        Input:
            
            roi             region-of-interest to crop. If roi = -1 (default), then no crop. 
                            Otherwise roi is given by a list with ndim elements, 
                            where each element is either -1 if no crop along this 
                            dimension or a tuple with beginning and end coodinates to crop to.
                            Example:
                                to crop 4D array along 2nd dimension:
                                roi = [-1, -1, (100, 900), -1]
                            
            binning         number of pixels to bin (combine) along each dimension.
                            If binning = 1, then projections in original resolution are loaded. 
                            Otherwise, binning is given by a list with ndim integers. 
                            Example:
                                to rebin 3D array along 1st direction:
                                binning = [1, 5, 1]
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
        
        if (self.roi == -1):
            roi_par = [-1] * ndim
        else:
            roi_par = list(self.roi)
            if (len(roi_par) != ndim):
                raise Exception('Number of dimensions and number of elements in roi parameter do not match')

        if (self.binning == 1):
            binning = [1] * ndim
        else:
            binning = list(self.binning)
            if (len(binning) != ndim):
                raise Exception('Number of dimensions and number of elements in binning parameter do not match')
                
        if (isinstance(data, ImageData)):
            if ((all(x == -1 for x in roi_par)) and (all(x == 1 for x in binning))):
                for key in data.dimension_labels:
                    if data.dimension_labels[key] == 'channel':
                        geometry.channels = geometry_0.channels
                        roi_par[key] = (0, geometry.channels)
                    elif data.dimension_labels[key] == 'horizontal_y':
                        geometry.voxel_size_y = geometry_0.voxel_size_y
                        geometry.voxel_num_y = geometry_0.voxel_num_y
                        roi_par[key] = (0, geometry.voxel_num_y)
                    elif data.dimension_labels[key] == 'vertical':
                        geometry.voxel_size_z = geometry_0.voxel_size_z
                        geometry.voxel_num_z = geometry_0.voxel_num_z
                        roi_par[key] = (0, geometry.voxel_num_z)
                    elif data.dimension_labels[key] == 'horizontal_x':
                        geometry.voxel_size_x = geometry_0.voxel_size_x
                        geometry.voxel_num_x = geometry_0.voxel_num_x
                        roi_par[key] = (0, geometry.voxel_num_x)
            else:
                for key in data.dimension_labels:
                    if data.dimension_labels[key] == 'channel':
                        if (roi_par[key] != -1):
                            geometry.channels = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0] + ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.channels = geometry_0.channels // binning[key]
                            roi_par[key] = (0, geometry.channels * binning[key])
                    elif data.dimension_labels[key] == 'horizontal_y':
                        if (roi_par[key] != -1):
                            geometry.voxel_num_y = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            geometry.voxel_size_y = geometry_0.voxel_size_y * binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0] + ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.voxel_num_y = geometry_0.voxel_num_y // binning[key]
                            geometry.voxel_size_y = geometry_0.voxel_size_y * binning[key]
                            roi_par[key] = (0, geometry.voxel_num_y * binning[key])
                    elif data.dimension_labels[key] == 'vertical':
                        if (roi_par[key] != -1):
                            geometry.voxel_num_z = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            geometry.voxel_size_z = geometry_0.voxel_size_z * binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0] + ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.voxel_num_z = geometry_0.voxel_num_z // binning[key]
                            geometry.voxel_size_z = geometry_0.voxel_size_z * binning[key]
                            roi_par[key] = (0, geometry.voxel_num_z * binning[key])
                    elif data.dimension_labels[key] == 'horizontal_x':
                        if (roi_par[key] != -1):
                            geometry.voxel_num_x = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            geometry.voxel_size_x = geometry_0.voxel_size_x * binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0]+ ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.voxel_num_x = geometry_0.voxel_num_x // binning[key]
                            geometry.voxel_size_x = geometry_0.voxel_size_x * binning[key]
                            roi_par[key] = (0, geometry.voxel_num_x * binning[key])
            
        else: # AcquisitionData
            if ((all(x == -1 for x in roi_par)) and (all(x == 1 for x in binning))):
                for key in data.dimension_labels:
                    if data.dimension_labels[key] == 'channel':
                        geometry.channels = geometry_0.channels
                        roi_par[key] = (0, geometry.channels)
                    elif data.dimension_labels[key] == 'angle':
                        geometry.angles = geometry_0.angles
                        roi_par[key] = (0, len(geometry.angles))
                    elif data.dimension_labels[key] == 'vertical':
                        geometry.pixel_size_v = geometry_0.pixel_size_v
                        geometry.pixel_num_v = geometry_0.pixel_num_v
                        roi_par[key] = (0, geometry.pixel_num_v)
                    elif data.dimension_labels[key] == 'horizontal':
                        geometry.pixel_size_h = geometry_0.pixel_size_h
                        geometry.pixel_num_h = geometry_0.pixel_num_h
                        roi_par[key] = (0, geometry.pixel_num_h)
            else:
                for key in data.dimension_labels:
                    if data.dimension_labels[key] == 'channel':
                        if (roi_par[key] != -1):
                            geometry.channels = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0] + ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.channels = geometry_0.channels // binning[key]
                            roi_par[key] = (0, geometry.channels * binning[key])
                    elif data.dimension_labels[key] == 'angle':
                        if (roi_par[key] != -1):
                            geometry.angles = geometry_0.angles[roi_par[key][0]:roi_par[key][1]]
                        else:
                            geometry.angles = geometry_0.angles
                            roi_par[key] = (0, len(geometry.angles))
                        if (binning[key] != 1):
                            binning[key] = 1
                            warnings.warn('Rebinning in angular dimensions is not supported: \n binning[{}] is set to 1.'.format(key))
                    elif data.dimension_labels[key] == 'vertical':
                        if (roi_par[key] != -1):
                            geometry.pixel_num_v = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            geometry.pixel_size_v = geometry_0.pixel_size_v * binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0] + ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.pixel_num_v = geometry_0.pixel_num_v // binning[key]
                            geometry.pixel_size_v = geometry_0.pixel_size_v * binning[key]
                            roi_par[key] = (0, geometry.pixel_num_v * binning[key])
                    elif data.dimension_labels[key] == 'horizontal':
                        if (roi_par[key] != -1):
                            geometry.pixel_num_h = (roi_par[key][1] - roi_par[key][0]) // binning[key]
                            geometry.pixel_size_h = geometry_0.pixel_size_h * binning[key]
                            roi_par[key] = (roi_par[key][0], roi_par[key][0] + ((roi_par[key][1] - roi_par[key][0]) // binning[key]) * binning[key])
                        else:
                            geometry.pixel_num_h = geometry_0.pixel_num_h // binning[key]
                            geometry.pixel_size_h = geometry_0.pixel_size_h * binning[key]
                            roi_par[key] = (0, geometry.pixel_num_h * binning[key])
                            
        if ndim == 2:
            n_pix_0 = (roi_par[0][1] - roi_par[0][0]) // binning[0]
            n_pix_1 = (roi_par[1][1] - roi_par[1][0]) // binning[1]
            shape = (n_pix_0, binning[0], 
                     n_pix_1, binning[1])
            data_resized = data.as_array()[roi_par[0][0]:(roi_par[0][0] + n_pix_0 * binning[0]), 
                                           roi_par[1][0]:(roi_par[1][0] + n_pix_1 * binning[1])].reshape(shape).mean(-1).mean(1)
        if ndim == 3:
            n_pix_0 = (roi_par[0][1] - roi_par[0][0]) // binning[0]
            n_pix_1 = (roi_par[1][1] - roi_par[1][0]) // binning[1]
            n_pix_2 = (roi_par[2][1] - roi_par[2][0]) // binning[2]
            shape = (n_pix_0, binning[0], 
                     n_pix_1, binning[1],
                     n_pix_2, binning[2])
            data_resized = data.as_array()[roi_par[0][0]:(roi_par[0][0] + n_pix_0 * binning[0]), 
                                           roi_par[1][0]:(roi_par[1][0] + n_pix_1 * binning[1]), 
                                           roi_par[2][0]:(roi_par[2][0] + n_pix_2 * binning[2])].reshape(shape).mean(-1).mean(1).mean(2)
        if ndim == 4:
            n_pix_0 = (roi_par[0][1] - roi_par[0][0]) // binning[0]
            n_pix_1 = (roi_par[1][1] - roi_par[1][0]) // binning[1]
            n_pix_2 = (roi_par[2][1] - roi_par[2][0]) // binning[2]
            n_pix_3 = (roi_par[3][1] - roi_par[3][0]) // binning[3]
            shape = (n_pix_0, binning[0], 
                     n_pix_1, binning[1],
                     n_pix_2, binning[2],
                     n_pix_3, binning[3])
            data_resized = data.as_array()[roi_par[0][0]:(roi_par[0][0] + n_pix_0 * binning[0]), 
                                           roi_par[1][0]:(roi_par[1][0] + n_pix_1 * binning[1]), 
                                           roi_par[2][0]:(roi_par[2][0] + n_pix_2 * binning[2]), 
                                           roi_par[3][0]:(roi_par[3][0] + n_pix_3 * binning[3])].reshape(shape).mean(-1).mean(1).mean(2).mean(3)

        out = type(data)(array = data_resized, 
                         deep_copy = False,
                         dimension_labels = data.dimension_labels,
                         geometry = geometry)
        
        return out


'''
#usage exaample
ig = ImageGeometry(voxel_num_x = 200, 
                   voxel_num_y = 200, 
                   voxel_num_z = 200, 
                   voxel_size_x = 1, 
                   voxel_size_y = 1, 
                   voxel_size_z = 1, 
                   center_x = 0, 
                   center_y = 0, 
                   center_z = 0, 
                   channels = 200)

im = ImageData(array = numpy.zeros((200, 200, 200, 200)),
               geometry = ig,
               deep_copy = False, 
               dimension_labels = ['channel',\
                                   'vertical',\
                                   'horizontal_y',\
                                   'horizontal_x'])
            
            
resizer = Resizer(binning = [1, 1, 7, 1], roi = -1)
resizer.input = im
data_resized = resizer.process()
print(data_resized)
'''
