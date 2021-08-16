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
from cil.framework import AcquisitionData, ImageData
import datetime


h5pyAvailable = True
try:
    import h5py 
except:
    h5pyAvailable = False


class NEXUSDataWriter(object):
    
    def __init__(self,
                 **kwargs):
        '''
        Constructor

        input:

        data: Acquistion or Image Data

        file_name: name of nexus file to write to (string)

        flat_field: flat field projections (Acquisition Data or numpy array)

        flat_field_key: numpy array of 0s and 1s, with same shape as flat_field.
        0 indicates flat field was imaged before the projections, 1 after. 

        dark_field: dark field projections (Acquisition Data or numpy array)

        dark_field_key:numpy array of 0s and 1s, with same shape as dark_field.
        0 indicates dark field was imaged before the projections, 1 after. 
        
        '''
        
        self.data = kwargs.get('data', None)
        self.file_name = kwargs.get('file_name', None)
        self.flat_field = kwargs.get('flat_field', None)
        self.flat_field_key = kwargs.get('flat_field_key', None)
        self.dark_field = kwargs.get('dark_field', None)
        self.dark_field_key = kwargs.get('dark_field_key', None)
        
        if ((self.data is not None) and (self.file_name is not None)):
            self.set_up(data=self.data,
                        file_name=self.file_name,
                        flat_field=self.flat_field,
                        flat_field_key=self.flat_field_key,
                        dark_field=self.dark_field,
                        dark_field_key=self.dark_field_key)
        
    def set_up(self, data=None, file_name=None,
               flat_field=None, flat_field_key=None,
               dark_field=None, dark_field_key=None):
        '''
        data: Acquistion or Image Data

        file_name: name of nexus file to write to (string)

        flat_field: flat field projections (Acquisition Data or numpy array)

        flat_field_key: numpy array of 0s and 1s, with same shape as flat_field.
        0 indicates flat field was imaged before the projections, 1 after. 

        dark_field: dark field projections (Acquisition Data or numpy array)

        dark_field_key:numpy array of 0s and 1s, with same shape as dark_field.
        0 indicates dark field was imaged before the projections, 1 after. 
        
        '''
        self.data = data
        self.file_name = file_name
        if flat_field is not None:
            self.set_flat_field(flat_field)
            self.set_flat_field_key(flat_field_key)
        if dark_field is not None:
            self.set_dark_field(dark_field)
            self.set_dark_field_key(dark_field_key)
        
        if not ((isinstance(self.data, ImageData)) or 
                (isinstance(self.data, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        
        # check that h5py library is installed
        if h5pyAvailable is False:
            raise Exception('h5py is not available, cannot load NEXUS files.')

        for data in [self.flat_field, self.dark_field]:
            if not (data is None or isinstance(data, AcquisitionData)
                    or isinstance(data, np.ndarray)):
                raise Exception('Flat and dark fields must be one of the following data types:\n' +
                            ' - np.ndarray\n - AcquisitionData')

    def set_flat_field(self, flat_field):
        '''
        Sets the flat field projections.
        flat_field: flat field projections (Acquisition Data or numpy array)
        '''
        if isinstance(flat_field, AcquisitionData):
            flat_field = flat_field.as_array()
        elif type(flat_field) is not np.ndarray:
            raise TypeError('Flat fields must be one of the following data types:\n' +
                            ' - np.ndarray\n - AcquisitionData')
        if self.data is not None:
            self._validate_field_shape(flat_field)
        self.flat_field = flat_field

    def set_dark_field(self, dark_field):
        '''
        Sets the dark field projections
        dark_field: dark field projections (Acquisition Data or numpy array)
        '''
        if isinstance(dark_field, AcquisitionData):
            dark_field = dark_field.as_array()
        elif type(dark_field) is not np.ndarray:
            raise TypeError('Dark fields must be one of the following data types:\n' +
                            ' - np.ndarray\n - AcquisitionData')
        if self.data is not None:
            self._validate_field_shape(dark_field)
        self.dark_field = dark_field

    def set_flat_field_key(self, key):
        '''
        Sets a key to identify the ordering of the flat fields relative to the
        projections.

        key: numpy array of 0s and 1s, with same shape as flat_field.
        0 indicates flat field was imaged before the projections, 1 after.
        '''
        if self.flat_field is None:
            raise ValueError('Flat field must be set before flat field key')
        if key is None:
            shape = self.flat_field.shape
            key = np.zeros(shape)
        key = self._validate_key_shape(key, self.flat_field)
        self.flat_field_key = key

    def set_dark_field_key(self, key):
        '''
        Sets a key to identify the ordering of the dark fields relative to the
        projections.

        key: numpy array of 0s and 1s, with same shape as dark_field.
        0 indicates dark field was imaged before the projections, 1 after.
        '''
        if self.dark_field is None:
            raise ValueError('Dark field must be set before dark field key')
        key = self._validate_key_shape(key, self.dark_field)
        self.dark_field_key = key

    def _validate_field_shape(self, field_data):
        ''' Checks that the shape of the projection data is the same
        as that of the dark/flat field data, in the horizontal and 
        vertical dimensions'''

        angles_id = list(self.data.dimension_labels).index('angle')
        try:
            channels_id = list(self.data.dimension_labels).index('channel')
        except ValueError:
            channels_id = None
        projection_shape = list(self.data.as_array().shape)
        field_shape = list(field_data.shape)
        projection_shape.pop(angles_id)
        field_shape.pop(angles_id)
        if channels_id is not None:
            projection_shape.pop(channels_id)
            field_shape.pop(channels_id)
        if not (projection_shape == field_shape):
            raise ValueError('Flats/Dark and projections size do not match: {},{}'.format(field_shape, projection_shape))

    def _validate_key_shape(self, key, field_data):
        ''' Checks that there are the same number of entries in the key,
         as there are projections in the field data.
        If key is set to None, it is assumed that the fields were taken
        before the projections, and the key is set to a numpy array of 0s,
        with the same number of entries as there are projections in the field data.'''
        if key is not None:
            if type(key) not in [np.ndarray, list]:
                raise TypeError('Dark/flat field key must be one of the following data types:\n' +
                                ' - np.ndarray\n - list')
        field_shape = list(field_data.shape)
        horizontal_id = list(self.data.dimension_labels).index('horizontal')
        field_shape.pop(horizontal_id)
        try:
            vertical_id = list(self.data.dimension_labels).index('vertical')
            field_shape.pop(vertical_id)
        except ValueError:
            pass

        if key is None:
            key = np.zeros(field_shape)

        num_field_projections = field_shape
        num_key_entries = list(key.shape)
        if not (num_field_projections == num_key_entries):
            raise ValueError('Number of Flats/Dark and key entries do not match: {},{}'.format(num_field_projections, num_key_entries))

        return key

    def write(self):
        ''' writes the data and flat and dark field information
        (if present) to self.file_name'''
        
        # if the folder does not exist, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))
            
        # create the file
        with h5py.File(self.file_name, 'w') as f:
            
            # give the file some important attributes
            f.attrs['file_name'] = self.file_name
            f.attrs['file_time'] = str(datetime.datetime.utcnow())
            f.attrs['creator'] = np.string_('NEXUSDataWriter.py')
            f.attrs['NeXus_version'] = '4.3.0'
            f.attrs['HDF5_Version'] = h5py.version.hdf5_version
            f.attrs['h5py_version'] = h5py.version.version
            
            # create the NXentry group
            nxentry = f.create_group('entry1/tomo_entry')
            nxentry.attrs['NX_class'] = 'NXentry'

            ds_data = f.create_dataset('entry1/tomo_entry/data/data',
                            (self.data.as_array().shape),
                            dtype='float32',
                            data=self.data.as_array())

            if (isinstance(self.data, AcquisitionData)):
                # create group to store configuration
                f.create_group('entry1/tomo_entry/config')
                f.create_group('entry1/tomo_entry/config/source')
                f.create_group('entry1/tomo_entry/config/detector')
                f.create_group('entry1/tomo_entry/config/rotation_axis')
            
                ds_data.attrs['data_type'] = 'AcquisitionData'
                ds_data.attrs['geometry'] = self.data.geometry.config.system.geometry
                ds_data.attrs['dimension'] = self.data.geometry.config.system.dimension   
                ds_data.attrs['num_channels'] = self.data.geometry.config.channels.num_channels

                if self.flat_field is not None:
                    self._validate_field_shape(self.flat_field)
                    f.create_dataset('entry1/tomo_entry/data/flat_field/data',
                                    (self.flat_field.shape),
                                    dtype='float32',
                                    data=self.flat_field)
                    f.create_dataset('entry1/tomo_entry/data/flat_field/position_key',
                                    (self.flat_field_key.shape),
                                    dtype='int32',
                                    data=self.flat_field_key)
                if self.dark_field is not None:
                    self._validate_field_shape(self.dark_field)
                    f.create_dataset('entry1/tomo_entry/data/dark_field/data',
                                    (self.dark_field.shape),
                                    dtype='float32',
                                    data=self.dark_field)
                    f.create_dataset('entry1/tomo_entry/data/dark_field/position_key',
                                    (self.dark_field_key.shape),
                                    dtype='int32',
                                    data=self.dark_field_key)

                f.create_dataset('entry1/tomo_entry/config/detector/direction_x',
                                 (self.data.geometry.config.system.detector.direction_x.shape), 
                                 dtype = 'float32', 
                                 data = self.data.geometry.config.system.detector.direction_x)
                
                f.create_dataset('entry1/tomo_entry/config/detector/position',
                                 (self.data.geometry.config.system.detector.position.shape), 
                                 dtype = 'float32', 
                                 data = self.data.geometry.config.system.detector.position)
                
                if self.data.geometry.config.system.geometry == 'cone':
                    f.create_dataset('entry1/tomo_entry/config/source/position', 
                                     (self.data.geometry.config.system.source.position.shape), 
                                     dtype = 'float32', 
                                     data = self.data.geometry.config.system.source.position)
                else:
                    f.create_dataset('entry1/tomo_entry/config/ray/direction',
                                     (self.data.geometry.config.system.ray.direction.shape), 
                                     dtype = 'float32', 
                                     data = self.data.geometry.config.system.ray.direction)
                
                f.create_dataset('entry1/tomo_entry/config/rotation_axis/position', 
                                 (self.data.geometry.config.system.rotation_axis.position.shape), 
                                 dtype = 'float32', 
                                 data = self.data.geometry.config.system.rotation_axis.position)
                
                
                ds_data.attrs['num_pixels_h'] = self.data.geometry.config.panel.num_pixels[0]
                ds_data.attrs['pixel_size_h'] = self.data.geometry.config.panel.pixel_size[0]
                ds_data.attrs['panel_origin'] = self.data.geometry.config.panel.origin

                if self.data.geometry.config.system.dimension == '3D':
                    f.create_dataset('entry1/tomo_entry/config/detector/direction_y', 
                                     (self.data.geometry.config.system.detector.direction_y.shape), 
                                     dtype = 'float32', 
                                     data = self.data.geometry.config.system.detector.direction_y)
                    
                    f.create_dataset('entry1/tomo_entry/config/rotation_axis/direction', 
                                 (self.data.geometry.config.system.rotation_axis.direction.shape), 
                                 dtype = 'float32', 
                                 data = self.data.geometry.config.system.rotation_axis.direction)
                
                    ds_data.attrs['num_pixels_v'] = self.data.geometry.config.panel.num_pixels[1]
                    ds_data.attrs['pixel_size_v'] = self.data.geometry.config.panel.pixel_size[1]

                angle_data = self.data.geometry.config.angles.angle_data
                    
                angles = f.create_dataset('entry1/tomo_entry/config/angles', 
                                                 (angle_data.shape), 
                                                 dtype = 'float32', 
                                                 data = angle_data)
                angles.attrs['angle_unit'] = self.data.geometry.config.angles.angle_unit   
                angles.attrs['initial_angle'] = self.data.geometry.config.angles.initial_angle
            
            else:   # ImageData
                ds_data.attrs['data_type'] = 'ImageData'
                ds_data.attrs['voxel_num_x'] = self.data.geometry.voxel_num_x
                ds_data.attrs['voxel_num_y'] = self.data.geometry.voxel_num_y
                ds_data.attrs['voxel_num_z'] = self.data.geometry.voxel_num_z
                ds_data.attrs['voxel_size_x'] = self.data.geometry.voxel_size_x
                ds_data.attrs['voxel_size_y'] = self.data.geometry.voxel_size_y
                ds_data.attrs['voxel_size_z'] = self.data.geometry.voxel_size_z
                ds_data.attrs['center_x'] = self.data.geometry.center_x
                ds_data.attrs['center_y'] = self.data.geometry.center_y
                ds_data.attrs['center_z'] = self.data.geometry.center_z
                ds_data.attrs['channels'] = self.data.geometry.channels
                ds_data.attrs['channel_spacing'] = self.data.geometry.channel_spacing

            # set up dataset attributes    
            for i in range(self.data.number_of_dimensions):
                ds_data.attrs['dim{}'.format(i)] = self.data.dimension_labels[i]