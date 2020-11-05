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
        
        self.file_name = os.path.abspath(file_name)
        
        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot load NEXUS files.')
            
        if self.file_name == None:
            raise Exception('Path to nexus file is required.')
        
        # check if nexus file exists
        if not(os.path.isfile(self.file_name)):
            raise Exception('File\n {}\n does not exist.'.format(self.file_name))  
        
        self._geometry = None
    
    def read_dimension_labels(self, attrs):
        dimension_labels = [None] * 4
        for k,v in attrs.items():
            if k in ['dim0', 'dim1', 'dim2' , 'dim3']:
                dimension_labels[int(k[3:])] = v
        if all (map(lambda x: x is not None, dimension_labels)):
            # remove Nones
            if dimension_labels[3] is None:
                dimension_labels.pop(3)
            if dimension_labels[2] is None:
                dimension_labels.pop(2)
        else:
            #assign default values => don't set dimension_labels
            dimension_labels = None
        return dimension_labels

    def get_geometry(self):
        
        '''
        Parse NEXUS file and returns either ImageData or Acquisition Data 
        depending on file content
        '''
        
        with h5py.File(self.file_name,'r') as dfile:
            
            if np.string_(dfile.attrs['creator']) != np.string_('NEXUSDataWriter.py'):
                raise Exception('We can parse only files created by NEXUSDataWriter.py')
            
            ds_data = dfile['entry1/tomo_entry/data/data']
            
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
                                
                # read the dimension_labels from dim{}
                dimension_labels = self.read_dimension_labels(ds_data.attrs)
                
            else:   # AcquisitionData
                
                # if old file 
                if self.is_old_file_version():
                    
                    dimension_labels = []
                    
                    dimension_labels = self.read_dimension_labels(ds_data.attrs)
                    
                    # if cone 3D
                    if ds_data.attrs.__contains__('dist_source_center') == True and ds_data.attrs['pixel_num_v'] > 1:
                        self._geometry = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, 0],
                                                        rotation_axis_position=[0, ds_data.attrs['dist_source_center'], 0],
                                                        detector_position=[0, ds_data.attrs['dist_source_center'] + ds_data.attrs['dist_center_detector'], 0])
                        
                        self._geometry.set_panel((int(ds_data.attrs['pixel_num_h']), int(ds_data.attrs['pixel_num_v'])),
                                                    pixel_size=(ds_data.attrs['pixel_size_h'], ds_data.attrs['pixel_size_v']))
                    
                    # if cone 2D
                    elif ds_data.attrs.__contains__('dist_source_center') == True and ds_data.attrs['pixel_num_v'] == 1:
                        self._geometry = AcquisitionGeometry.create_Cone2D(source_position=[0, 0],
                                                        rotation_axis_position=[0, ds_data.attrs['dist_source_center']],
                                                        detector_position=[0, ds_data.attrs['dist_source_center'] + ds_data.attrs['dist_center_detector']])
                        self._geometry.set_angles(np.array(dfile['entry1/tomo_entry/data/rotation_angle'], dtype = 'float32'))
                        
                        self._geometry.set_panel(int(ds_data.attrs['pixel_num_h']),
                                                    pixel_size=ds_data.attrs['pixel_size_h'])
                    
                    # if parallel 3D
                    elif ds_data.attrs.__contains__('dist_source_center') == False and ds_data.attrs['pixel_num_v'] > 1:
                        self._geometry = AcquisitionGeometry.create_Parallel3D()
                        
                        self._geometry.set_panel((int(ds_data.attrs['pixel_num_h']), int(ds_data.attrs['pixel_num_v'])),
                                                    pixel_size=(ds_data.attrs['pixel_size_h'], ds_data.attrs['pixel_size_v']))
                    # if parallel 2D
                    elif ds_data.attrs.__contains__('dist_source_center') == False and ds_data.attrs['pixel_num_v'] == 1:
                        self._geometry = AcquisitionGeometry.create_Parallel2D()
                        
                        self._geometry.set_panel(int(ds_data.attrs['pixel_num_h']),
                                                    pixel_size=ds_data.attrs['pixel_size_h'])
                    
                    # set channels
                    self._geometry.set_channels(num_channels = int(ds_data.attrs['channels']))
                    
                    # set angles
                    ds_angles = dfile['entry1/tomo_entry/data/rotation_angle']
                    if ds_angles.attrs.__contains__('units'):
                        self._geometry.set_angles(np.array(ds_angles, dtype = 'float32'),
                                                    angle_unit=ds_angles.attrs['units'])
                    else:
                        self._geometry.set_angles(np.array(ds_angles, dtype = 'float32'))
                    
                # new file
                else:
                    rotation_axis_position = np.array(dfile['entry1/tomo_entry/config/rotation_axis/position'], dtype = 'float32')
                    detector_position = np.array(dfile['entry1/tomo_entry/config/detector/position'], dtype = 'float32')
                    detector_direction_row = np.array(dfile['entry1/tomo_entry/config/detector/direction_row'], dtype = 'float32')
                    ds_angles = dfile['entry1/tomo_entry/config/angles']
                    angles = np.array(ds_angles, dtype = 'float32')
                    angle_unit = ds_angles.attrs['angle_unit']
                    initial_angle = ds_angles.attrs['initial_angle']
                    
                    # dimension_labels = []
                    
                    # # if 4D
                    # if ds_data.attrs['dimension'] == '3D' and ds_data.attrs['num_channels'] > 1:
                    #     ndim = 4
                    # elif (ds_data.attrs['dimension'] == '3D' and ds_data.attrs['num_channels'] == 1) or (ds_data.attrs['dimension'] == '2D' and ds_data.attrs['num_channels'] > 1):
                    #     ndim = 3
                    # else:
                    #     ndim = 2
                        
                    # for i in range(ndim):
                    #         dimension_labels.append(ds_data.attrs['dim{}'.format(i)])
                    dimension_labels = self.read_dimension_labels(ds_data.attrs)
                    
                    # if cone beam geometry
                    if ds_data.attrs['geometry'] == 'cone':
                        source_position = np.array(dfile['entry1/tomo_entry/config/source/position'], dtype = 'float32')
                        # if Cone3D
                        if ds_data.attrs['dimension'] == '3D':
                            detector_direction_col = np.array(dfile['entry1/tomo_entry/config/detector/direction_col'], dtype = 'float32')
                            rotation_axis_direction = np.array(dfile['entry1/tomo_entry/config/rotation_axis/direction'], dtype = 'float32')
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
                        ray_direction = np.array(dfile['entry1/tomo_entry/config/ray/direction'], dtype = 'float32')
                        # if Parallel3D
                        if ds_data.attrs['dimension'] == '3D':
                            detector_direction_col = np.array(dfile['entry1/tomo_entry/config/detector/direction_col'], dtype = 'float32')
                            rotation_axis_direction = np.array(dfile['entry1/tomo_entry/config/rotation_axis/direction'], dtype = 'float32')
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
        
        self._geometry.set_labels(dimension_labels)
        return self._geometry
    
    
    def read(self):
        
        if self._geometry is None:
            self.get_geometry()
            
        with h5py.File(self.file_name,'r') as dfile:
                
            ds_data = dfile['entry1/tomo_entry/data/data']
            data = np.array(ds_data, dtype = np.float32)
            
            # handle old files?
            if self.is_old_file_version():
                if isinstance(self._geometry , AcquisitionGeometry):
                    return AcquisitionData(data, True, geometry=self._geometry, suppress_warning=True)
                elif isinstance(self._geometry , ImageGeometry):
                    return ImageData(data, True, geometry=self._geometry, suppress_warning=True)
                else:
                    raise TypeError("Unsupported geometry. Expected ImageGeometry or AcquisitionGeometry, got {}"\
                        .format(type(self._geometry)))

            output = self._geometry.allocate(None)
            output.fill(data)
            return output
    
          
    def load_data(self):
        '''alias of read'''
        return self.read()
    def is_old_file_version(self):
        #return ds_data.attrs.__contains__('geom_type')
        with h5py.File(self.file_name,'r') as dfile:
                
            if np.string_(dfile.attrs['creator']) != np.string_('NEXUSDataWriter.py'):
                raise Exception('We can parse only files created by NEXUSDataWriter.py')
            
            ds_data = dfile['entry1/tomo_entry/data/data']

            return 'geom_type' in ds_data.attrs.keys()
            # return True