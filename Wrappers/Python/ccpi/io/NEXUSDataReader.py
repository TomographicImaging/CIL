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
import numpy as np
import os
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry

h5pyAvailable = True
try:
    import h5py 
except:
    h5pyAvailable = False


class NEXUSDataReader(object):
    
    def __init__(self,
                 **kwargs):
        
        '''
        Constructor
        
        Input:
            
            file_name      full path to NEXUS file
        '''
        
        self.file_name = kwargs.get('file_name', None)
        
        if self.file_name is not None:
            self.set_up(file_name = self.file_name)
            
    def set_up(self, 
               file_name = None):
        
        self.file_name = file_name
        
        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot load NEXUS files.')
            
        if self.file_name == None:
            raise Exception('Path to nexus file is required.')
        
        # check if nexus file exists
        if not(os.path.isfile(self.file_name)):
            raise Exception('File\n {}\n does not exist.'.format(self.file_name))  
        
        self._geometry = None
        
    def get_geometry(self):
        
        '''
        Parse NEXUS file and returns either ImageData or Acquisition Data 
        depending on file content
        '''
        
        try:
            with h5py.File(self.file_name,'r') as file:
                
                if np.string_(file.attrs['creator']) != np.string_('NEXUSDataWriter.py'):
                    raise Exception('We can parse only files created by NEXUSDataWriter.py')
                
                ds_data = file['entry1/tomo_entry/data/data']
                
                if ds_data.attrs['data_type'] == 'ImageData':
                    
                    self._geometry = ImageGeometry(voxel_num_x = int(ds_data.attrs['voxel_num_x']),
                                                   voxel_num_y = int(ds_data.attrs['voxel_num_y']),
                                                   voxel_num_z = int(ds_data.attrs['voxel_num_z']),
                                                   voxel_size_x = ds_data.attrs['voxel_size_x'],
                                                   voxel_size_y = ds_data.attrs['voxel_size_y'],
                                                   voxel_size_z = ds_data.attrs['voxel_size_z'],
                                                   center_x = ds_data.attrs['center_x'],
                                                   center_y = ds_data.attrs['center_y'],
                                                   center_z = ds_data.attrs['center_z'],
                                                   channels = ds_data.attrs['channels'])
                    
                    if ds_data.attrs.__contains__('channel_spacing') == True:
                        self._geometry.channel_spacing = ds_data.attrs['channel_spacing']
                        
                    dimension_labels = []
                        
                    # if 4D
                    if ds_data.attrs['voxel_num_z'] > 1 and ds_data.attrs['channels'] > 1:
                        ndim = 4
                    elif (ds_data.attrs['voxel_num_z'] == 1 and ds_data.attrs['channels'] > 1) or (ds_data.attrs['voxel_num_z'] > 1 and ds_data.attrs['channels'] == 1):
                        ndim = 3
                    else:
                        ndim = 2
                        
                    for i in range(ndim):
                            dimension_labels.append(ds_data.attrs['dim{}'.format(i)])
                        
                    
                else:   # AcquisitionData
                    
                    # if old file 
                    if ds_data.attrs.__contains__('geom_type'):
                        
                        dimension_labels = []
                        
                        # if 4D
                        if ds_data.attrs['pixel_num_v'] > 1 and ds_data.attrs['channels'] > 1:
                            ndim = 4
                        elif (ds_data.attrs['pixel_num_v'] == 1 and ds_data.attrs['channels'] > 1) or (ds_data.attrs['pixel_num_v'] > 1 and ds_data.attrs['channels'] == 1):
                            ndim = 3
                        else:
                            ndim = 2
                            
                        for i in range(ndim):
                                dimension_labels.append(ds_data.attrs['dim{}'.format(i)])
                        
                        # if cone 3D
                        if ds_data.attrs.__contains__('dist_source_center') == True and ds_data.attrs['pixel_num_v'] > 1:
                            self._geometry = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, 0],
                                                         rotation_axis_position=[0, ds_data.attrs['dist_source_center'], 0],
                                                         detector_position=[0, ds_data.attrs['dist_source_center'] + ds_data.attrs['dist_center_detector'], 0])
                            self._geometry.set_angles(np.array(file['entry1/tomo_entry/data/rotation_angle'], dtype = 'float32'))
                            
                            self._geometry.set_panel((int(ds_data.attrs['pixel_num_h']), int(ds_data.attrs['pixel_num_v'])),
                                                     pixel_size=(ds_data.attrs['pixel_size_h'], ds_data.attrs['pixel_size_v']))
                        
                        # if cone 2D
                        elif ds_data.attrs.__contains__('dist_source_center') == True and ds_data.attrs['pixel_num_v'] == 1:
                            self._geometry = AcquisitionGeometry.create_Cone2D(source_position=[0, 0],
                                                         rotation_axis_position=[0, ds_data.attrs['dist_source_center']],
                                                         detector_position=[0, ds_data.attrs['dist_source_center'] + ds_data.attrs['dist_center_detector']])
                            self._geometry.set_angles(np.array(file['entry1/tomo_entry/data/rotation_angle'], dtype = 'float32'))
                            
                            self._geometry.set_panel(int(ds_data.attrs['pixel_num_h']),
                                                     pixel_size=ds_data.attrs['pixel_size_h'])
                        
                        # if parallel 3D
                        elif ds_data.attrs.__contains__('dist_source_center') == False and ds_data.attrs['pixel_num_v'] > 1:
                            self._geometry = AcquisitionGeometry.create_Parallel3D()
                            self._geometry.set_angles(np.array(file['entry1/tomo_entry/data/rotation_angle'], dtype = 'float32'))
                            
                            self._geometry.set_panel((int(ds_data.attrs['pixel_num_h']), int(ds_data.attrs['pixel_num_v'])),
                                                     pixel_size=(ds_data.attrs['pixel_size_h'], ds_data.attrs['pixel_size_v']))
                        # if parallel 2D
                        elif ds_data.attrs.__contains__('dist_source_center') == False and ds_data.attrs['pixel_num_v'] == 1:
                            self._geometry = AcquisitionGeometry.create_Parallel2D()
                            self._geometry.set_angles(np.array(file['entry1/tomo_entry/data/rotation_angle'], dtype = 'float32'))
                            
                            self._geometry.set_panel(int(ds_data.attrs['pixel_num_h']),
                                                     pixel_size=ds_data.attrs['pixel_size_h'])
                        
                        # set channels
                        self._geometry.set_channels(num_channels = int(ds_data.attrs['channels']))
                        
                    # new file
                    else:
                        rotation_axis_position = np.array(file['entry1/tomo_entry/config/rotation_axis/position'], dtype = 'float32')
                        detector_position = np.array(file['entry1/tomo_entry/config/detector/position'], dtype = 'float32')
                        detector_direction_row = np.array(file['entry1/tomo_entry/config/detector/direction_row'], dtype = 'float32')
                        ds_angles = file['entry1/tomo_entry/config/angles']
                        angles = np.array(ds_angles, dtype = 'float32')
                        angle_unit = ds_angles.attrs['angle_unit']
                        initial_angle = ds_angles.attrs['initial_angle']
                        
                        dimension_labels = []
                        
                        # if 4D
                        if ds_data.attrs['dimension'] == '3D' and ds_data.attrs['num_channels'] > 1:
                            ndim = 4
                        elif (ds_data.attrs['dimension'] == '3D' and ds_data.attrs['num_channels'] == 1) or (ds_data.attrs['dimension'] == '2D' and ds_data.attrs['num_channels'] > 1):
                            ndim = 3
                        else:
                            ndim = 2
                            
                        for i in range(ndim):
                                dimension_labels.append(ds_data.attrs['dim{}'.format(i)])
                        
                        # if cone beam geometry
                        if ds_data.attrs['geometry'] == 'cone':
                            source_position = np.array(file['entry1/tomo_entry/config/source/position'], dtype = 'float32')
                            # if Cone3D
                            if ds_data.attrs['dimension'] == '3D':
                                detector_direction_col = np.array(file['entry1/tomo_entry/config/detector/direction_col'], dtype = 'float32')
                                rotation_axis_direction = np.array(file['entry1/tomo_entry/config/rotation_axis/direction'], dtype = 'float32')
                                self._geometry = AcquisitionGeometry.create_Cone3D(source_position=source_position,
                                                                                   rotation_axis_position=rotation_axis_position,
                                                                                   rotation_axis_direction=rotation_axis_direction,
                                                                                   detector_position=detector_position,
                                                                                   detector_direction_row=detector_direction_row,
                                                                                   detector_direction_col=detector_direction_col)
                            
                                self._geometry.set_panel((int(ds_data.attrs['num_pixels_h']), int(ds_data.attrs['num_pixels_v'])),
                                                         pixel_size=(ds_data.attrs['pixel_size_h'], ds_data.attrs['pixel_size_v']))
                            # if Cone2D
                            else:
                                self._geometry = AcquisitionGeometry.create_Cone2D(source_position=source_position,
                                                                                   rotation_axis_position=rotation_axis_position,
                                                                                   detector_position=detector_position,
                                                                                   detector_direction_row=detector_direction_row)
                            
                                self._geometry.set_panel(int(ds_data.attrs['num_pixels_h']),
                                                         pixel_size=ds_data.attrs['pixel_size_h'])
                        
                        # if parallel beam geometry
                        else:
                            ray_direction = np.array(file['entry1/tomo_entry/config/ray/direction'], dtype = 'float32')
                            # if Parallel3D
                            if ds_data.attrs['dimension'] == '3D':
                                detector_direction_col = np.array(file['entry1/tomo_entry/config/detector/direction_col'], dtype = 'float32')
                                rotation_axis_direction = np.array(file['entry1/tomo_entry/config/rotation_axis/direction'], dtype = 'float32')
                                self._geometry = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction,
                                                                                       rotation_axis_position=rotation_axis_position,
                                                                                       rotation_axis_direction=rotation_axis_direction,
                                                                                       detector_position=detector_position,
                                                                                       detector_direction_row=detector_direction_row,
                                                                                       detector_direction_col=detector_direction_col)
                            
                                self._geometry.set_panel((int(ds_data.attrs['num_pixels_h']), int(ds_data.attrs['num_pixels_v'])),
                                                         pixel_size=(ds_data.attrs['pixel_size_h'], ds_data.attrs['pixel_size_v']))
                            # if Parallel2D
                            else:
                                self._geometry = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction,
                                                                                       rotation_axis_position=rotation_axis_position,
                                                                                       detector_position=detector_position,
                                                                                       detector_direction_row=detector_direction_row)
                            
                                self._geometry.set_panel(int(ds_data.attrs['num_pixels_h']),
                                                         pixel_size=ds_data.attrs['pixel_size_h'])
                                
                        self._geometry.set_angles(angles, 
                                                  angle_unit=angle_unit,
                                                  initial_angle=initial_angle)
                            
                        self._geometry.set_channels(num_channels = int(ds_data.attrs['num_channels']))
            
            self._geometry.dimension_labels = dimension_labels
                        
        except:
            print('Error reading {}'.format(self.file_name))
            raise
                        
        return self._geometry
    
    
    def load(self):
        
        if self._geometry == None:
            self.get_geometry()
            
        with h5py.File(self.file_name,'r') as file:
                
            ds_data = file['entry1/tomo_entry/data/data']
            data = np.array(ds_data, dtype = 'float32')
            
            if ds_data.attrs['data_type'] == 'ImageData':
    
                return ImageData(array = data,
                                 deep_copy = False,
                                 geometry = self._geometry,
                                 dimension_labels = self._geometry.dimension_labels)
            
            else:
                
                return AcquisitionData(array = data,
                                       deep_copy = False,
                                       geometry = self._geometry,
                                       dimension_labels = self._geometry.dimension_labels)
                        