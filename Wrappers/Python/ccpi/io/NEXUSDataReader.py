#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:25 2019

@author: evelina
"""


import numpy
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
            
            nexus_file      full path to NEXUS file
        '''
        
        self.nexus_file = kwargs.get('nexus_file', None)
        
        if self.nexus_file is not None:
            self.set_up(nexus_file = self.nexus_file,
                        roi = self.roi,
                        binning = self.binning)
            
    def set_up(self, 
               nexus_file = None):
        
        self.nexus_file = nexus_file
        
        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot load NEXUS files.')
            
        if self.nexus_file == None:
            raise Exception('Path to nexus file is required.')
        
        # check if nexus file exists
        if not(os.path.isfile(self.nexus_file)):
            raise Exception('File\n {}\n does not exist.'.format(self.nexus_file))  
        
    def load_data(self):
        
        '''
        Parse NEXUS file and returns either ImageData or Acquisition Data 
        depending on file content
        '''
        
        try:
            with h5py.File(self.nexus_file,'r') as file:
                
                if (file.attrs['creator'] != 'NEXUSDataWriter.py'):
                    raise Exception('We can parse only files created by NEXUSDataWriter.py')
                
                ds_data = file['entry1/tomo_entry/data/data']
                data = numpy.array(ds_data, dtype = 'float32')
                
                dimension_labels = []
                
                for i in range(data.ndim):
                    dimension_labels.append(ds_data.attrs['dim{}'.format(i)])
                
                if ds_data.attrs['data_type'] == 'ImageData':
                    self._geometry = ImageGeometry(voxel_num_x = ds_data.attrs['voxel_num_x'],
                                                   voxel_num_y = ds_data.attrs['voxel_num_y'],
                                                   voxel_num_z = ds_data.attrs['voxel_num_z'],
                                                   voxel_size_x = ds_data.attrs['voxel_size_x'],
                                                   voxel_size_y = ds_data.attrs['voxel_size_y'],
                                                   voxel_size_z = ds_data.attrs['voxel_size_z'],
                                                   center_x = ds_data.attrs['center_x'],
                                                   center_y = ds_data.attrs['center_y'],
                                                   center_z = ds_data.attrs['center_z'],
                                                   channels = ds_data.attrs['channels'])
                    
                    return ImageData(array = data,
                                     deep_copy = False,
                                     geometry = self._geometry,
                                     dimension_labels = dimension_labels)
                    
                else:   # AcquisitionData
                    self._geometry = AcquisitionGeometry(geom_type = ds_data.attrs['geom_type'],
                                                         dimension = ds_data.attrs['dimension'],
                                                         dist_source_center = ds_data.attrs['dist_source_center'],
                                                         dist_center_detector = ds_data.attrs['dist_center_detector'],
                                                         pixel_num_h = ds_data.attrs['pixel_num_h'],
                                                         pixel_size_h = ds_data.attrs['pixel_size_h'],
                                                         pixel_num_v = ds_data.attrs['pixel_num_v'],
                                                         pixel_size_v = ds_data.attrs['pixel_size_v'],
                                                         channels = ds_data.attrs['channels'],
                                                         angles = numpy.array(file['entry1/tomo_entry/data/rotation_angle'], dtype = 'float32'))
                                                         #angle_unit = file['entry1/tomo_entry/data/rotation_angle'].attrs['units'])
                                             
                    return AcquisitionData(array = data,
                                           deep_copy = False,
                                           geometry = self._geometry,
                                           dimension_labels = dimension_labels)
                    
        except:
            print("Error reading nexus file")
            raise
                
    def get_geometry(self):
        
        '''
        Return either ImageGeometry or AcquisitionGeometry 
        depepnding on file content
        '''
        
        return self._geometry


'''
# usage example
reader = NEXUSDataReader()
reader.set_up(nexus_file = '/home/evelina/test_nexus.nxs')
acquisition_data = reader.load_data()
print(acquisition_data)
ag = reader.get_geometry()
print(ag)

reader = NEXUSDataReader()
reader.set_up(nexus_file = '/home/evelina/test_nexus_im.nxs')
image_data = reader.load_data()
print(image_data)
ig = reader.get_geometry()
print(ig)
'''
