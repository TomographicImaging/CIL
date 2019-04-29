#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:25 2019

@author: evelina
"""


import numpy
import os
from ccpi.framework import AcquisitionData, AcquisitionGeometry


h5pyAvailable = True
try:
    import h5py 
except:
    h5pyAvailable = False
    

''' 
filename = '/media/newhd/shared/Data/CCPi-data/CIL-demodata/24737_fd.nxs'
f = h5py.File(filename, 'r')
print(list(f.keys()))
tmp = numpy.array(f['entry1/tomo_entry/instrument/detector/image_key'])
print(tmp)
data = numpy.array(f['entry1/tomo_entry/data/data/'])
print(data)
rotation_angle = numpy.array(f['entry1/tomo_entry/data/rotation_angle/'])
print(rotation_angle)
'''

class NEXUSDataReader(object):
    
    def __init__(self,
                 **kwargs):
        
        self.nexus_file = kwargs.get('nexus_file', None)
        self.pixel_size_h_0 = kwargs.get('pixel_size_h_0', 1)
        self.pixel_size_v_0 = kwargs.get('pixel_size_v_0', 1)
        self.roi = kwargs.get('roi', -1)
        self.binning = kwargs.get('binning', [1, 1])
        
        if self.nexus_file is not None:
            self.set_up(nexus_file = self.nexus_file,
                        pixel_size_h_0 = self.pixel_size_h_0,
                        pixel_size_v_0 = self.pixel_size_v_0,
                        roi = self.roi,
                        binning = self.binning)
            
    def set_up(self, 
               nexus_file = None, 
               pixel_size_h_0 = 1,
               pixel_size_v_0 = 1,
               roi = -1, 
               binning = [1, 1]):
        
        self.nexus_file = nexus_file
        self.pixel_size_h_0 = pixel_size_h_0
        self.pixel_size_v_0 = pixel_size_v_0
        self.roi = roi
        self.binning = binning
        
        self._key_path = 'entry1/tomo_entry/instrument/detector/image_key'
        self._data_path = 'entry1/tomo_entry/data/data'
        self._angle_path = 'entry1/tomo_entry/data/rotation_angle'

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
            raise Exception("h5py is not available, cannot load nexus files")
        
        # read metadata    
        with h5py.File(self.nexus_file,'r') as file: 
                image_keys = numpy.array(file[self._key_path])                
                angles = numpy.array(file[self._angle_path], dtype = float)[image_keys==0]
                pixel_num_v_0, pixel_num_h_0 =  list(file['entry1/tomo_entry/data/data/'].shape[1,2])
        
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