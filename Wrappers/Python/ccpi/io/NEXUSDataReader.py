#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:25 2019

@author: evelina
"""


import numpy
import os
import matplotlib.pyplot as plt
from ccpi.framework import AcquisitionData, AcquisitionGeometry


h5pyAvailable = True
try:
    import h5py 
except:
    h5pyAvailable = False


class NEXUSDataReader(object):
    
    def __init__(self,
                 **kwargs):
        
        self.nexus_file = kwargs.get('nexus_file', None)
        self.pixel_size_h_0 = kwargs.get('pixel_size_h_0', 1)
        self.pixel_size_v_0 = kwargs.get('pixel_size_v_0', 1)
        self.key_path = kwargs.get('key_path', 'entry1/tomo_entry/instrument/detector/image_key')
        self.data_path = kwargs.get('data_path', 'entry1/tomo_entry/data/data')
        self.angle_path =  kwargs.get('angle_path', 'entry1/tomo_entry/data/rotation_angle')
        self.roi = kwargs.get('roi', -1)
        self.binning = kwargs.get('binning', [1, 1])
        
        if self.nexus_file is not None:
            self.set_up(nexus_file = self.nexus_file,
                        pixel_size_h_0 = self.pixel_size_h_0,
                        pixel_size_v_0 = self.pixel_size_v_0,
                        key_path = self.key_path,
                        data_path = self.data_path,
                        angle_path = self.angle_path,
                        roi = self.roi,
                        binning = self.binning)
            
    def set_up(self, 
               nexus_file = None, 
               pixel_size_h_0 = 1,
               pixel_size_v_0 = 1,
               key_path = 'entry1/tomo_entry/instrument/detector/image_key',
               data_path = 'entry1/tomo_entry/data/data',
               angle_path = 'entry1/tomo_entry/data/rotation_angle',
               roi = -1, 
               binning = [1, 1]):
        
        self.nexus_file = nexus_file
        self.pixel_size_h_0 = pixel_size_h_0
        self.pixel_size_v_0 = pixel_size_v_0
        self.key_path = key_path
        self.data_path = data_path
        self.angle_path = angle_path
        self.roi = roi
        self.binning = binning
        
        if self.nexus_file == None:
            raise Exception('Path to nexus file is required.')
        
        # check if nexus file exists
        if not(os.path.isfile(self.nexus_file)):
            raise Exception('File {} does not exist'.format(self.nexus_file))  
        
        # check ROI
        if (self.roi != -1): 
            if not ((isinstance(self.roi, list)) or 
                    (len(self.roi) == 4) or 
                    (self.roi[0] < self.roi[2]) or 
                    (self.roi[1] < self.roi[3])):
                raise Exception('Not valid ROI. ROI must be defined as [row0, column0, row1, column1] \
                                such that ((row0 < row1) and (column0 < column1))')
        
        # check binning parameters
        if (not(isinstance(self.binning, list)) or 
            (len(self.binning) != 2)):
            raise Exception('Not valid binning parameters. \
                            Binning must be defined as [int, int]')
        
        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot read NEXUS files')
        
        # read metadata
        try:
            with h5py.File(self.nexus_file,'r') as file:        
                self._image_keys = numpy.array(file[self.key_path])                
                angles = numpy.array(file[self.angle_path], dtype = float)[self._image_keys == 0]
                pixel_num_v_0, pixel_num_h_0 =  file['entry1/tomo_entry/data/data/'].shape[1:]
        except:
            print('Error reading NEXUS file')
            raise
        
        # calculate number of pixels and pixel size
        if ((self.binning == [1, 1]) and (self.roi == -1)):
            pixel_num_v = pixel_num_v_0
            pixel_num_h = pixel_num_h_0
            pixel_size_v = pixel_size_v_0
            pixel_size_h = pixel_size_h_0
            
        elif ((self.binning == [1, 1]) and (self.roi != -1)):
            pixel_num_v = self.roi[2] - self.roi[0]
            pixel_num_h = self.roi[3] - self.roi[1]
            pixel_size_v = pixel_size_v_0
            pixel_size_h = pixel_size_h_0
            
        elif ((self.binning > [1, 1]) and (self.roi == -1)):
            pixel_num_v = pixel_num_v_0 // self.binning[0]
            pixel_num_h = pixel_num_h_0 // self.binning[1]
            pixel_size_v = pixel_size_v_0 * self.binning[0]
            pixel_size_h = pixel_size_h_0 * self.binning[1]
            
        elif ((self.binning > [1, 1]) and (self.roi != -1)):
            pixel_num_v = (self.roi[2] - self.roi[0]) // self.binning[0]
            pixel_num_h = (self.roi[3] - self.roi[1]) // self.binning[1]
            pixel_size_v = pixel_size_v_0 * self.binning[0]
            pixel_size_h = pixel_size_h_0 * self.binning[1]
            
        self._ag = AcquisitionGeometry(geom_type = 'parallel', 
                                       dimension = '3D', 
                                       angles = angles, 
                                       pixel_num_h = pixel_num_h, 
                                       pixel_size_h = pixel_size_h, 
                                       pixel_num_v = pixel_num_v, 
                                       pixel_size_v = pixel_size_v,
                                       channels = 1,
                                       angle_unit = 'degree')
    
    def get_acquisition_geometry(self):
        '''
        Return AcquisitionGeometry object
        '''
        
        return self._ag
    
    def load_projections(self):
        '''
        Load projections and returns AcquisitionData object
        '''
        
        if 0 not in self._image_keys:
            raise ValueError('Projections are not in the data. Data Path ', self._data_path)
        
        data = self._load(0)
        
        return AcquisitionData(array = data, 
                               geometry = self._ag,
                               dimension_labels = ['angle', 'vertical', 'horizontal'])
        
    def load_flat_images(self):
        '''
        Load flat field images and returns numpy array
        '''
        
        if 1 not in self._image_keys:
            raise ValueError('Flat field images are not in the data. Data Path ', self._data_path)
        
        data = self._load(1)
        
        return data
    
    def load_dark_images(self):
        '''
        Load dark field images and returns numpy array
        '''
        
        if 2 not in self._image_keys:
            raise ValueError('Dark field images are not in the data. Data Path ', self._data_path)
        
        data = self._load(2)
        
        return data
    
    def _load(self, key_id = 0):
        '''
        Generic loader for projections, flat and dark images. Returns numpy array.
        '''
        
        try:
            with h5py.File(self.nexus_file,'r') as file:
                if ((self.binning == [1, 1]) and (self.roi == -1)):
                    data = numpy.array(file[self.data_path][self._image_keys == key_id, :, :])
                
                elif ((self.binning == [1, 1]) and (self.roi != -1)):
                    data = numpy.array(file[self.data_path][self._image_keys == key_id, self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]])
                
                elif ((self.binning > [1, 1]) and (self.roi == -1)):
                    shape = (len(self._image_keys[self._image_keys == key_id]),
                             self._ag.pixel_num_v, self.binning[0], 
                             self._ag.pixel_num_h, self.binning[1])
                    data = numpy.array(file[self.data_path][self._image_keys == key_id, \
                                                            :(self._ag.pixel_num_v * self.binning[0]), \
                                                            :(self._ag.pixel_num_h * self.binning[1])]).reshape(shape).mean(-1).mean(2)
                
                elif ((self.binning > [1, 1]) and (self.roi != -1)):
                    shape = (len(self._image_keys[self._image_keys == key_id]),
                             self._ag.pixel_num_v, self.binning[0], 
                             self._ag.pixel_num_h, self.binning[1])
                    data = numpy.array(file[self.data_path])[self._image_keys == key_id, \
                                                             self.roi[0]:(self.roi[0] + (((self.roi[2] - self.roi[0]) // self.binning[0]) * self.binning[0])), \
                                                             self.roi[1]:(self.roi[1] + (((self.roi[3] - self.roi[1]) // self.binning[1]) * self.binning[1]))].reshape(shape).mean(-1).mean(2)
                    
        except:
            print('Error reading NEXUS file')
            raise
            
        return data

# usage example
nexus_file = '/media/newhd/shared/Data/CCPi-data/CIL-demodata/24737_fd.nxs'
reader = NEXUSDataReader()
reader.set_up(nexus_file = nexus_file,
              roi = [10, 30, 120, 110],
              binning = [2, 2])

ag = reader.get_acquisition_geometry()
print(ag)

data = reader.load_projections()

plt.imshow(data.as_array()[0, :, :])
plt.show()

flat = reader.load_flat_images()

plt.imshow(flat[0, :, :])
plt.show()

dark = reader.load_dark_images()

plt.imshow(dark[0, :, :])
plt.show()
