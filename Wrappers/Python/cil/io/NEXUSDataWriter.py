# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy as np
import os
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from cil.version import version
import datetime
from cil.io import utilities

h5pyAvailable = True
try:
    import h5py 
except:
    h5pyAvailable = False


class NEXUSDataWriter(object):
    ''' Create a writer for NEXUS files.
    
    Parameters
    ----------
    data: AcquisitionData, ImageData, 
        The dataset to write to file
    file_name: os.path or string, default None
        The file name to write
    compression: str, {'uint8', 'uint16', None}, default None
        The lossy compression to apply, default None will not compress data.
        uint8 or unit16 will compress to 8 and 16 bit dtypes respectively.
    '''
    
    def __init__(self, data=None, file_name=None, compression=None):

        self.data = data
        self.file_name = file_name

        if ((data is not None) and (file_name is not None)):
            self.set_up(data = data, file_name = file_name, compression=compression)
        
    def set_up(self,
               data = None,
               file_name = None,
               compression = None):

        '''
        Set up the writer

        data: AcquisitionData, ImageData, 
            The dataset to write to file
        file_name: os.path or string, default None
            The file name to write
        compression: int, default 0
            The lossy compression to apply, default 0 will not compress data.
            8 or 16 will compress to 8 and 16 bit dtypes respectively.
        '''
        self.data = data
        self.file_name = file_name
        
        if self.file_name is None:
            raise Exception('Path to write file is required.')
        else:
            self.file_name = os.path.abspath(file_name)

        if self.data is None:
            raise Exception('Data to write is required.')

        if not self.file_name.endswith('nxs') and not self.file_name.endswith('nex'):
            self.file_name+='.nxs'
        
        # Deal with compression
        self.compress           = utilities.get_compress(compression)
        self.dtype              = utilities.get_compressed_dtype(data, compression)
        self.compression        = compression
        
        if not ((isinstance(self.data, ImageData)) or 
                (isinstance(self.data, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        
        
        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot write NEXUS files.')
    
    def write(self):

        '''
        write dataset to disk
        '''   
        # check filename and data have been set:
        if self.file_name is None:
            raise TypeError('Path to nexus file to write to is required.')
        if self.data is None:
            raise TypeError('Data to write out must be set.')
        
        # if the folder does not exist, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))

        if self.compress is True:
            scale, offset = utilities.get_compression_scale_offset(self.data, self.compression)
                
        # create the file
        with h5py.File(self.file_name, 'w') as f:
            
            # give the file some important attributes
            f.attrs['file_name'] = self.file_name
            f.attrs['cil_version'] = version
            f.attrs['file_time'] = str(datetime.datetime.utcnow())
            f.attrs['creator'] = np.string_('NEXUSDataWriter.py')
            f.attrs['NeXus_version'] = '4.3.0'
            f.attrs['HDF5_Version'] = h5py.version.hdf5_version
            f.attrs['h5py_version'] = h5py.version.version
            
            # create the NXentry group
            nxentry = f.create_group('entry1/tomo_entry')
            nxentry.attrs['NX_class'] = 'NXentry'

            #create empty data entry
            ds_data = f.create_dataset('entry1/tomo_entry/data/data',shape=self.data.shape, dtype=self.dtype)

            if self.compress:
                ds_data.attrs['scale'] = scale
                ds_data.attrs['offset'] = offset

                for i in range(self.data.shape[0]):
                    ds_data[i:(i+1)] = self.data.array[i] * scale + offset
            else:
                ds_data.write_direct(self.data.array)

            # set up dataset attributes
            if (isinstance(self.data, ImageData)):
                ds_data.attrs['data_type'] = 'ImageData'
            else:
                ds_data.attrs['data_type'] = 'AcquisitionData'
            
            for i in range(self.data.number_of_dimensions):
                ds_data.attrs['dim{}'.format(i)] = self.data.dimension_labels[i]
            
            if (isinstance(self.data, AcquisitionData)): 
                
                # create group to store configuration
                f.create_group('entry1/tomo_entry/config')
                f.create_group('entry1/tomo_entry/config/source')
                f.create_group('entry1/tomo_entry/config/detector')
                f.create_group('entry1/tomo_entry/config/rotation_axis')
                
                ds_data.attrs['geometry'] = self.data.geometry.config.system.geometry
                ds_data.attrs['dimension'] = self.data.geometry.config.system.dimension   
                ds_data.attrs['num_channels'] = self.data.geometry.config.channels.num_channels
                
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
                    
                angles = f.create_dataset('entry1/tomo_entry/config/angles', 
                                                 (self.data.geometry.config.angles.angle_data.shape), 
                                                 dtype = 'float32', 
                                                 data = self.data.geometry.config.angles.angle_data)
                angles.attrs['angle_unit'] = self.data.geometry.config.angles.angle_unit   
                angles.attrs['initial_angle'] = self.data.geometry.config.angles.initial_angle
            
            else:   # ImageData
                
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
