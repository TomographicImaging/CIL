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
import numpy as np
import os
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry

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
        
        # remove Nones
        dimension_labels = [i for i in dimension_labels if i] 

        if len(dimension_labels) == 0:
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
                if ds_data.attrs.__contains__('dist_source_center') or dfile['entry1/tomo_entry'].__contains__('config/source/position'):
                    geom_type = 'cone'
                else:
                    geom_type = 'parallel'

                if ds_data.attrs.__contains__('num_pixels_v'):
                    num_pixels_v = ds_data.attrs.get('num_pixels_v')
                elif ds_data.attrs.__contains__('pixel_num_v'):
                    num_pixels_v = ds_data.attrs.get('pixel_num_v')
                else:
                    num_pixels_v = 1

                if num_pixels_v > 1:
                    dim = 3
                else:
                    dim = 2


                if self.is_old_file_version():
                    num_pixels_h = ds_data.attrs.get('pixel_num_h', 1)
                    num_channels = ds_data.attrs['channels']
                    ds_angles = dfile['entry1/tomo_entry/data/rotation_angle']

                    if geom_type == 'cone' and dim == 3:
                        self._geometry = AcquisitionGeometry.create_Cone3D(source_position=[0, -ds_data.attrs['dist_source_center'], 0],
                                                                               detector_position=[0, ds_data.attrs['dist_center_detector'],0])
                    elif geom_type == 'cone' and dim == 2:
                        self._geometry = AcquisitionGeometry.create_Cone2D(source_position=[0, -ds_data.attrs['dist_source_center']],
                                                        detector_position=[0, ds_data.attrs['dist_center_detector']])
                    elif geom_type == 'parallel' and dim == 3:
                        self._geometry = AcquisitionGeometry.create_Parallel3D()
                    elif geom_type == 'parallel' and dim == 2:
                        self._geometry = AcquisitionGeometry.create_Parallel2D()

    
                else:
                    num_pixels_h = ds_data.attrs.get('num_pixels_h', 1)
                    num_channels = ds_data.attrs['num_channels']
                    ds_angles = dfile['entry1/tomo_entry/config/angles']

                    rotation_axis_position = list(dfile['entry1/tomo_entry/config/rotation_axis/position'])
                    detector_position = list(dfile['entry1/tomo_entry/config/detector/position'])

                    ds_detector = dfile['entry1/tomo_entry/config/detector']
                    if ds_detector.__contains__('direction_x'):
                        detector_direction_x = list(dfile['entry1/tomo_entry/config/detector/direction_x'])
                    else:
                        detector_direction_x = list(dfile['entry1/tomo_entry/config/detector/direction_row'])
 
                    if ds_detector.__contains__('direction_y'):
                        detector_direction_y = list(dfile['entry1/tomo_entry/config/detector/direction_y'])
                    elif ds_detector.__contains__('direction_col'):
                        detector_direction_y = list(dfile['entry1/tomo_entry/config/detector/direction_col'])

                    ds_rotate = dfile['entry1/tomo_entry/config/rotation_axis']
                    if ds_rotate.__contains__('direction'):
                        rotation_axis_direction = list(dfile['entry1/tomo_entry/config/rotation_axis/direction'])

                    if geom_type == 'cone':
                        source_position = list(dfile['entry1/tomo_entry/config/source/position'])

                        if dim == 2:
                            self._geometry = AcquisitionGeometry.create_Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position)
                        else:
                            self._geometry = AcquisitionGeometry.create_Cone3D(source_position,\
                                                detector_position, detector_direction_x, detector_direction_y,\
                                                rotation_axis_position, rotation_axis_direction)
                    else:
                        ray_direction = list(dfile['entry1/tomo_entry/config/ray/direction'])

                        if dim == 2:
                            self._geometry = AcquisitionGeometry.create_Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)
                        else:
                            self._geometry = AcquisitionGeometry.create_Parallel3D(ray_direction,\
                                                detector_position, detector_direction_x, detector_direction_y,\
                                                rotation_axis_position, rotation_axis_direction)

                # for all Aquisition data
                #set angles
                angles = list(ds_angles)
                angle_unit = ds_angles.attrs.get('angle_unit','degree')
                initial_angle = ds_angles.attrs.get('initial_angle',0)
                self._geometry.set_angles(angles, initial_angle=initial_angle, angle_unit=angle_unit)

                #set panel
                pixel_size_v = ds_data.attrs.get('pixel_size_v', ds_data.attrs['pixel_size_h'])
                origin = ds_data.attrs.get('panel_origin','bottom-left')
                self._geometry.set_panel((num_pixels_h, num_pixels_v),\
                                        pixel_size=(ds_data.attrs['pixel_size_h'], pixel_size_v),\
                                        origin=origin)

                # set channels
                self._geometry.set_channels(num_channels)

                dimension_labels = []
                dimension_labels = self.read_dimension_labels(ds_data.attrs)
            
        #set labels
        self._geometry.set_labels(dimension_labels)

        return self._geometry
    
    def read(self):
        
        if self._geometry is None:
            self.get_geometry()
            
        with h5py.File(self.file_name,'r') as dfile:
                
            ds_data = dfile['entry1/tomo_entry/data/data']
            # If AcquisitionData, data may contain dark and flat fields:
            if ds_data.attrs['data_type'] == 'AcquisitionData' and 'entry1/tomo_entry/data/image_key' in dfile:
                all_data = np.array(ds_data, dtype = np.float32)
                image_keys = np.array(dfile['entry1/tomo_entry/data/image_key'])
                if 0 not in image_keys:
                    raise ValueError("Projections are not in the data. Data Path ",
                                    self.file_name)
                else:
                    data = all_data[image_keys == 0]
            else:
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

    def load_dark(self):
        '''
        Loads the dark field data from the nexus file, if
        present.
        returns: numpy array with flat field data
        '''
        return self._load_with_image_id(2)

    def load_flat(self):
        '''
        Loads the flat field data from the nexus file, if
        present.
        returns: numpy array with flat field data
        '''
        return self._load_with_image_id(1)

    def _load_with_image_id(self, image_key_id):
        '''
        This is generic loading function for loading flat field, dark field and
        projection data, if an image_key array is saved in the data file.
        Loads data with image key id of image_key_id
        '''
        with h5py.File(self.file_name,'r') as dfile:  
            ds_data = dfile['entry1/tomo_entry/data/data']
            # If AcquisitionData, data may contain dark and flat fields:
            if ds_data.attrs['data_type'] == 'AcquisitionData' and 'entry1/tomo_entry/data/image_key' in dfile:
                all_data = np.array(ds_data, dtype = np.float32)
                image_keys = np.array(dfile['entry1/tomo_entry/data/image_key'])
                if image_key_id not in image_keys:
                    raise ValueError("Data with image key: ", image_key_id, "is not in the data. Data Path ",
                                    self.file_name)
                else:
                    print("Image keys: ", image_keys)
                    data = all_data[image_keys == image_key_id]
                    if len(data==1):
                        data = data[0]
                    print(data)
            else:
                raise ValueError("Dark and Flat fields are not saved in the data. Data Path ",
                                    self.file_name)

            return data