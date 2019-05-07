#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:11:20 2019

@author: evelina
"""


import numpy
import os
import matplotlib.pyplot as plt
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from NikonDataReader import NikonDataReader
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
        self.image_geometry = kwargs.get('image_geometry', None)
        self.acquisition_geometry = kwargs.get('acquisition_geometry', None)
        self.flat_images = kwargs.get('flat_images', None)
        self.dark_images = kwargs.get('dark_images', None)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data_container = self.data_container,
                        file_name = self.file_name,
                        image_geometry = self.image_geometry,
                        acquisition_geometry = self.acquisition_geometry,
                        flat_images = self.flat_images,
                        dark_images = self.dark_images)
        
    def set_up(self,
               data_container = None,
               file_name = None,
               image_geometry = None,
               acquisition_geometry = None,
               flat_images = None,
               dark_images = None):
        
        self.data_container = data_container
        self.file_name = file_name
        self.image_geometry = image_geometry
        self.acquisition_geometry = acquisition_geometry
        self.flat_images = flat_images
        self.dark_images = dark_images
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData)) or 
                ((self.flat_images is not None) and (isinstance(self.flat_images, AcquisitionData))) or
                ((self.dark_images is not None) and (isinstance(self.dark_images, AcquisitionData)))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
            
        if ((isinstance(self.data_container, ImageData) and 
             ((self.flat_images is not None) or (self.dark_images is not None)))):
            raise Exception('ImageData and flat/ dark images are not compatible.\n' +
                            'Please pass AcquisitionData')
        
        if ((isinstance(self.data_container, ImageData)) and (self.image_geometry == None)):
            raise Exception('image_geometry is required.')
        
        if ((isinstance(self.data_container, AcquisitionData)) and (self.acquisition_geometry == None)):
            raise Exception('acquisition_geometry is required.')
            
    
    def write_file(self):
        # if the folder does not exist, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))
            
        # create the file
        with h5py.File(self.file_name, "w") as f:
            
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
                ds_data.attrs['geom_type'] = self.acquisition_geometry.geom_type
                ds_data.attrs['dimension'] = self.acquisition_geometry.dimension
                ds_data.attrs['dist_source_center'] = self.acquisition_geometry.dist_source_center
                ds_data.attrs['dist_center_detector'] = self.acquisition_geometry.dist_center_detector
                ds_data.attrs['pixel_num_h'] = self.acquisition_geometry.pixel_num_h
                ds_data.attrs['pixel_size_h'] = self.acquisition_geometry.pixel_size_h
                ds_data.attrs['pixel_num_v'] = self.acquisition_geometry.pixel_num_v
                ds_data.attrs['pixel_size_v'] = self.acquisition_geometry.pixel_size_v
                ds_data.attrs['channels'] = self.acquisition_geometry.channels
                ds_data.attrs['n_angles'] = self.acquisition_geometry.angles.shape[0]
                
                # create the dataset to store rotation angles
                ds_angles = f.create_dataset('entry1/tomo_entry/data/rotation_angle', 
                                             (self.acquisition_geometry.angles.shape), 
                                             dtype = 'float32', 
                                             data = self.acquisition_geometry.angles)
                
                #ds_angles.attrs['units'] = self.acquisition_geometry.angle_unit
                
                # create the dataset to store the flat images
                if (self.flat_images is not None):
                    ds_flat = f.create_dataset('entry1/tomo_entry/flat/data',
                                               (self.flat_images.as_array().shape), 
                                               dtype = 'float32', 
                                               data = self.flat_images.as_array())
                    
                    ds_flat.attrs['data_type'] = 'AcquisitionData'
                    
                    for i in range(self.flat_images.as_array().ndim):
                        ds_flat.attrs['dim{}'.format(i)] = self.flat_images.dimension_labels[i]
                    
                    ds_flat.attrs['pixel_num_h'] = self.acquisition_geometry.pixel_num_h
                    ds_flat.attrs['pixel_size_h'] = self.acquisition_geometry.pixel_size_h
                    ds_flat.attrs['pixel_num_v'] = self.acquisition_geometry.pixel_num_v
                    ds_flat.attrs['pixel_size_v'] = self.acquisition_geometry.pixel_size_v
                    ds_flat.attrs['channels'] = self.acquisition_geometry.channels
                    ds_flat.attrs['n_flat_images'] = self.flat_images.as_array().shape[self.flat_images.dimension_labels == 'angle']
                
                # create the dataset to store dark images
                if (self.dark_images is not None):
                    ds_dark = f.create_dataset('entry1/tomo_entry/dark/data',
                                               (self.dark_images.as_array().shape), 
                                               dtype = 'float32', 
                                               data = self.dark_images.as_array())
                    
                    ds_dark.attrs['data_type'] = 'AcquisitionData'
                    
                    for i in range(self.dark_images.as_array().ndim):
                        ds_dark.attrs['dim{}'.format(i)] = self.dark_images.dimension_labels[i]
                    
                    ds_dark.attrs['pixel_num_h'] = self.acquisition_geometry.pixel_num_h
                    ds_dark.attrs['pixel_size_h'] = self.acquisition_geometry.pixel_size_h
                    ds_dark.attrs['pixel_num_v'] = self.acquisition_geometry.pixel_num_v
                    ds_dark.attrs['pixel_size_v'] = self.acquisition_geometry.pixel_size_v
                    ds_dark.attrs['channels'] = self.acquisition_geometry.channels
                    ds_dark.attrs['n_dark_images'] = self.dark_images.as_array().shape[self.dark_images.dimension_labels == 'angle']
            
            else:   # ImageData
                
                ds_data.attrs['voxel_num_x'] = self.image_geometry.voxel_num_x
                ds_data.attrs['voxel_num_y'] = self.image_geometry.voxel_num_y
                ds_data.attrs['voxel_num_z'] = self.image_geometry.voxel_num_z
                ds_data.attrs['voxel_size_x'] = self.image_geometry.voxel_size_x
                ds_data.attrs['voxel_size_y'] = self.image_geometry.voxel_size_y
                ds_data.attrs['voxel_size_z'] = self.image_geometry.voxel_size_z
                ds_data.attrs['center_x'] = self.image_geometry.center_x
                ds_data.attrs['center_y'] = self.image_geometry.center_y
                ds_data.attrs['center_z'] = self.image_geometry.center_z
                ds_data.attrs['channels'] = self.image_geometry.channels
            
            
# usage example
xtek_file = '/home/evelina/nikon_data/SophiaBeads_256_averaged.xtekct'
reader = NikonDataReader()
reader.set_up(xtek_file = xtek_file,
              binning = [3, 1],
              roi = [200, 500, 1500, 2000],
              normalize = True)

data = reader.load_projections()
ag = reader.get_acquisition_geometry()
flat = AcquisitionData(array = numpy.zeros((10,10,10), dtype = float),
                       dimension_lables = ['angles', 'vertical', 'horizontal'])
dark = AcquisitionData(array = numpy.zeros((10,20,30), dtype = float),
                       dimension_lables = ['angles', 'vertical', 'horizontal'])

writer = NEXUSDataWriter()
writer.set_up(file_name = '/home/evelina/test_nexus.nxs',
              data_container = data,
              acquisition_geometry = ag,
              flat_images = flat,
              dark_images = dark)

writer.write_file()