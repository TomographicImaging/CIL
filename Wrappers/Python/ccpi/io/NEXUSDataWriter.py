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
import numpy
import os
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import datetime


h5pyAvailable = True
try:
    import h5py 
except:
    h5pyAvailable = False


class NEXUSDataWriter(object):
    
    def __init__(self,
                 **kwargs):
        
        self.data_container = kwargs.get('data_container', None)
        self.file_name = kwargs.get('file_name', None)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data_container = self.data_container,
                        file_name = self.file_name)
        
    def set_up(self,
               data_container = None,
               file_name = None):
        
        self.data_container = data_container
        self.file_name = file_name
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        
        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot load NEXUS files.')
    
    def write_file(self):
        
        # if the folder does not exist, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))
            
        # create the file
        with h5py.File(self.file_name, 'w') as f:
            
            # give the file some important attributes
            f.attrs['file_name'] = self.file_name
            f.attrs['file_time'] = str(datetime.datetime.utcnow())
            f.attrs['creator'] = 'NEXUSDataWriter.py'
            f.attrs['NeXus_version'] = '4.3.0'
            f.attrs['HDF5_Version'] = h5py.version.hdf5_version
            f.attrs['h5py_version'] = h5py.version.version
            
            # create the NXentry group
            nxentry = f.create_group('entry1/tomo_entry')
            nxentry.attrs['NX_class'] = 'NXentry'
            
            # create dataset to store data
            ds_data = f.create_dataset('entry1/tomo_entry/data/data', 
                                       (self.data_container.as_array().shape), 
                                       dtype = 'float32', 
                                       data = self.data_container.as_array())
            
            # set up dataset attributes
            if (isinstance(self.data_container, ImageData)):
                ds_data.attrs['data_type'] = 'ImageData'
            else:
                ds_data.attrs['data_type'] = 'AcquisitionData'
            
            for i in range(self.data_container.as_array().ndim):
                ds_data.attrs['dim{}'.format(i)] = self.data_container.dimension_labels[i]
            
            if (isinstance(self.data_container, AcquisitionData)):      
                ds_data.attrs['geom_type'] = self.data_container.geometry.geom_type
                ds_data.attrs['dimension'] = self.data_container.geometry.dimension
                if self.data_container.geometry.dist_source_center is not None:
                    ds_data.attrs['dist_source_center'] = self.data_container.geometry.dist_source_center
                if self.data_container.geometry.dist_center_detector is not None:
                    ds_data.attrs['dist_center_detector'] = self.data_container.geometry.dist_center_detector
                ds_data.attrs['pixel_num_h'] = self.data_container.geometry.pixel_num_h
                ds_data.attrs['pixel_size_h'] = self.data_container.geometry.pixel_size_h
                ds_data.attrs['pixel_num_v'] = self.data_container.geometry.pixel_num_v
                ds_data.attrs['pixel_size_v'] = self.data_container.geometry.pixel_size_v
                ds_data.attrs['channels'] = self.data_container.geometry.channels
                ds_data.attrs['n_angles'] = self.data_container.geometry.angles.shape[0]
                
                # create the dataset to store rotation angles
                ds_angles = f.create_dataset('entry1/tomo_entry/data/rotation_angle', 
                                             (self.data_container.geometry.angles.shape), 
                                             dtype = 'float32', 
                                             data = self.data_container.geometry.angles)
                
                #ds_angles.attrs['units'] = self.data_container.geometry.angle_unit
            
            else:   # ImageData
                
                ds_data.attrs['voxel_num_x'] = self.data_container.geometry.voxel_num_x
                ds_data.attrs['voxel_num_y'] = self.data_container.geometry.voxel_num_y
                ds_data.attrs['voxel_num_z'] = self.data_container.geometry.voxel_num_z
                ds_data.attrs['voxel_size_x'] = self.data_container.geometry.voxel_size_x
                ds_data.attrs['voxel_size_y'] = self.data_container.geometry.voxel_size_y
                ds_data.attrs['voxel_size_z'] = self.data_container.geometry.voxel_size_z
                ds_data.attrs['center_x'] = self.data_container.geometry.center_x
                ds_data.attrs['center_y'] = self.data_container.geometry.center_y
                ds_data.attrs['center_z'] = self.data_container.geometry.center_z
                ds_data.attrs['channels'] = self.data_container.geometry.channels


'''
# usage example
xtek_file = '/home/evelina/nikon_data/SophiaBeads_256_averaged.xtekct'
reader = NikonDataReader()
reader.set_up(xtek_file = xtek_file,
              binning = [3, 1],
              roi = [200, 500, 1500, 2000],
              normalize = True)

data = reader.load_projections()
ag = reader.get_geometry()

writer = NEXUSDataWriter()
writer.set_up(file_name = '/home/evelina/test_nexus.nxs',
              data_container = data)

writer.write_file()

ig = ImageGeometry(voxel_num_x = 100,
                   voxel_num_y = 100)
im = ImageData(array = numpy.zeros((100, 100), dtype = 'float'),
               geometry = ig)
im_writer = NEXUSDataWriter()

im_writer.set_up(file_name = '/home/evelina/test_nexus_im.nxs',
                 data_container = im)
im_writer.write_file()

ag = AcquisitionGeometry(geom_type = 'parallel', 
                         dimension = '2D', 
                         angles = numpy.array([0, 1]), 
                         pixel_num_h = 200, 
                         pixel_size_h = 1, 
                         pixel_num_v = 100, 
                         pixel_size_v = 1)

ad = ag.allocate()
ag_writer = NEXUSDataWriter()
ag_writer.set_up(file_name = '/home/evelina/test_nexus_ag.nxs',
                 data_container = ad)
ag_writer.write_file()
'''