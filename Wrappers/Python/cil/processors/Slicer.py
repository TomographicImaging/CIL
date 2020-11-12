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

from ccpi.framework import DataProcessor, AcquisitionData, ImageData, DataContainer, AcquisitionGeometry
import numpy as np
import warnings


class Slicer(DataProcessor):

    def __init__(self,
                 roi = None,
                 force = False):
        
        '''
        Constructor
        
        Input:
            
            roi             region-of-interest to crop, specified as a dictionary
                            containing tuple (Start, Stop, Step)
        '''

        kwargs = {'roi': roi,
                  'force': force}

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
        
        dimension_labels = list(geometry_0.dimension_labels)
        
        if self.roi != None:
            for key in self.roi.keys():
                if key not in data.dimension_labels.values():
                    raise ValueError('Wrong label is specified for roi')
        
        roi = []
        sliceobj = []
        for i in range(ndim):
            roi.append([0, data.shape[i], 1])
            sliceobj.append(slice(None, None, None))
                        
        if self.roi == None:
            geometry = geometry_0.clone()
        else:
            if (isinstance(data, AcquisitionData)):
                if geometry_0.config.system.geometry == 'parallel':
                    ray_direction = geometry_0.config.system.ray.direction[:]
                else:
                    source_position = geometry_0.config.system.source.position[:]
                detector_position = geometry_0.config.system.detector.position[:]
                detector_direction_row = geometry_0.config.system.detector.direction_row[:]
                if geometry_0.config.system.dimension == '3D':
                    detector_direction_col = geometry_0.config.system.detector.direction_col[:]
                    rotation_axis_direction = geometry_0.config.system.rotation_axis.direction[:]
                rotation_axis_position = geometry_0.config.system.rotation_axis.position[:]
                num_channels = geometry_0.config.channels.num_channels
                pixel_size_x = geometry_0.config.panel.pixel_size[0]
                pixel_size_y = geometry_0.config.panel.pixel_size[1]
                num_pixels_x = geometry_0.config.panel.num_pixels[0]
                num_pixels_y = geometry_0.config.panel.num_pixels[1]
                angles = geometry_0.config.angles.angle_data[:]
            else: #ImageData
                geometry = geometry_0.clone()
                
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
                    
                    flag_warn = False
                    
                    if key == 'channel':
                        geometry.channels = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        
                        if geometry.channels == 1:
                            dimension_labels.remove('channel')
                        
                    elif key == 'horizontal_x':
                        geometry.voxel_num_x = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.voxel_size_x = geometry_0.voxel_size_x
                        
                        if geometry.voxel_num_x == 1:
                            dimension_labels.remove('horizontal_x')
                            
                        if ((roi[idx][0] != (geometry_0.voxel_num_x - roi[idx][1])) or \
                            ((roi[idx][1] - roi[idx][0]) % roi[idx][2] != 0)):
                            flag_warn = True
                            
                    elif key == 'horizontal_y':
                        geometry.voxel_num_y = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.voxel_size_y = geometry_0.voxel_size_y
                        
                        if geometry.voxel_num_y == 1:
                            dimension_labels.remove('horizontal_y')
                        
                        if ((roi[idx][0] != (geometry_0.voxel_num_y - roi[idx][1])) or \
                            ((roi[idx][1] - roi[idx][0]) % roi[idx][2] != 0)):
                            flag_warn = True
                            
                    elif key == 'vertical':
                        geometry.voxel_num_z = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        geometry.voxel_size_z = geometry_0.voxel_size_z
                        
                        if geometry.voxel_num_z == 1:
                            dimension_labels.remove('vertical')
                            
                        if ((roi[idx][0] != (geometry_0.voxel_num_z - roi[idx][1])) or \
                            ((roi[idx][1] - roi[idx][0]) % roi[idx][2] != 0)):
                            flag_warn = True
                    
                    if flag_warn == True:
                        warnings.warn('Geometrical center of the ImageData has been positioned at ({})'.format([geometry_0.center_x, geometry_0.center_y, geometry_0.center_z]))
                        
                else: #AcquisitionData
                    
                    detector_warn = False
                    data_container_warn = False
                    geometry = None
                    
                    if key == 'channel':
                        num_channels = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        
                        if num_channels == 1:
                            dimension_labels.remove('channel')
                            
                    elif key == 'horizontal':
                        num_pixels_x = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        pixel_size_x = geometry_0.config.panel.pixel_size[0]
                        
                        if num_pixels_x == 1:
                            data_container_warn = True
                            dimension_labels.remove('horizontal')
                            
                        if ((roi[idx][0] != (geometry_0.config.panel.num_pixels[0] - roi[idx][1])) or \
                            ((roi[idx][1] - roi[idx][0]) % roi[idx][2] != 0)):
                            detector_warn = True
                        
                    elif key == 'vertical':
                        num_pixels_y = int(np.ceil((roi[idx][1] - roi[idx][0]) / roi[idx][2]))
                        pixel_size_y = geometry_0.config.panel.pixel_size[1]
                        
                        if num_pixels_y == 1:
                            dimension_labels.remove('vertical')
                        
                            if geometry_0.config.system.geometry == 'parallel':
                                ray_direction = geometry_0.config.system.ray.direction[0:2]
                            else:
                                source_position = geometry_0.config.system.source.position[0:2]
                            detector_position = geometry_0.config.system.detector.position[0:2]
                            detector_direction_row = geometry_0.config.system.detector.direction_row[0:2]
                            rotation_axis_position = geometry_0.config.system.rotation_axis.position[0:2]
                            
                        if ((roi[idx][0] != (geometry_0.config.panel.num_pixels[1] - roi[idx][1])) or \
                            ((roi[idx][1] - roi[idx][0]) % roi[idx][2] != 0)):
                            detector_warn = True
                        
                        if geometry_0.config.system.geometry == 'cone' and num_pixels_y == 1 and (roi[idx][1] + roi[idx][0]) // 2 != geometry_0.config.panel.num_pixels[1] // 2:
                            data_container_warn = True
                            
                    elif key == 'angle':
                        angles = angles[sliceobj[idx]]
                        
                        if angles.shape[0] == 1:
                            dimension_labels.remove('angle')

            if (isinstance(data, AcquisitionData)):
                if data_container_warn == True:
                    if self.force == True:
                        geometry = None
                    else:
                        raise ValueError ("Unable to slice requested geometry. Use 'force=True' to return DataContainer instead.")
                else:
                    if geometry_0.config.system.geometry == 'cone':
                        if num_pixels_y == 1:
                            geometry = AcquisitionGeometry.create_Cone2D(source_position=source_position,
                                                                         rotation_axis_position=rotation_axis_position,
                                                                         detector_position=detector_position,
                                                                         detector_direction_row=detector_direction_row)
                            
                            geometry.set_panel(num_pixels_x, pixel_size=pixel_size_x)
                            
                        else:
                            geometry = AcquisitionGeometry.create_Cone3D(source_position=source_position,
                                                                         rotation_axis_position=rotation_axis_position,
                                                                         rotation_axis_direction=rotation_axis_direction,
                                                                         detector_position=detector_position,
                                                                         detector_direction_row=detector_direction_row,
                                                                         detector_direction_col=detector_direction_col)
                            
                            geometry.set_panel((num_pixels_x, num_pixels_y), 
                                               pixel_size=(pixel_size_x, pixel_size_y))
                    else:
                        if num_pixels_y == 1:
                            geometry = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction,
                                                                             rotation_axis_position=rotation_axis_position,
                                                                             detector_position=detector_position,
                                                                             detector_direction_row=detector_direction_row)
                            
                            geometry.set_panel(num_pixels_x, pixel_size=pixel_size_x)
                            
                        else:
                            geometry = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction,
                                                                             rotation_axis_position=rotation_axis_position,
                                                                             rotation_axis_direction=rotation_axis_direction,
                                                                             detector_position=detector_position,
                                                                             detector_direction_row=detector_direction_row,
                                                                             detector_direction_col=detector_direction_col)
                            
                            geometry.set_panel((num_pixels_x, num_pixels_y), 
                                               pixel_size=(pixel_size_x, pixel_size_y))
                            
                    geometry.set_angles(angles, 
                                        angle_unit=geometry_0.config.angles.angle_unit, 
                                        initial_angle=geometry_0.config.angles.initial_angle)
            
                    geometry.set_channels(num_channels=num_channels)
                            
                    if detector_warn == True:
                        warnings.warn('Geometrical center of the detector has been positioned at ({})'.format(geometry_0.config.system.detector.position))
                                                        
            data_resized = np.squeeze(data.as_array()[tuple(sliceobj)])
            
            if geometry == None:
                return DataContainer(data_resized, deep_copy=False, dimension_labels=dimension_labels, suppress_warning=True)
            else:
                return type(data)(data_resized, deep_copy=False, geometry=geometry, dimension_labels=dimension_labels, suppress_warning=True)