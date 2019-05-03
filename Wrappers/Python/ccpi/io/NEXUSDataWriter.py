#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:11:20 2019

@author: evelina
"""


import numpy
import os
import matplotlib.pyplot as plt
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageData
from NikonDataReader import NikonDataReader


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
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data_container = self.data_container,
                        file_name = self.file_name,
                        image_geometry = self.image_geometry,
                        acquisition_geometry = self.acquisition_geometry)
        
    def set_up(self,
               data_container = None,
               file_name = None,
               image_geometry = None,
               acquisition_geometry = None):
        
        self.data_container = data_container
        self.file_name = file_name
        self.image_geometry = image_geometry
        self.acquisition_geometry = acquisition_geometry
        
        if not (isinstance(self.data_container, ImageData) or (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        
        if ((isinstance(self.data_container, ImageData)) and (self.image_geometry == None)):
            raise Exception('image_geometry is required.')
        
        if ((isinstance(self.data_container, AcquisitionData)) and (self.acquisition_geometry == None)):
            raise Exception('acquisition_geometry is required.')
            
    
    def write_file(self):
        # if a folder not exists, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))
            
        # create a file
        with h5py.File(self.file_name, "w") as f:
            f.create_group('entry1/tomo_entry')
            f.create_dataset('entry1/tomo_entry/data/data', 
                             (self.data_container.as_array().shape), 
                             dtype = 'float', 
                             data = self.data_container.as_array())
            
            if (isinstance(self.data_container, AcquisitionData)):
                ds_angles = f.create_dataset('entry1/tomo_entry/data/rotation_angle', 
                                             (self.acquisition_geometry.angles.shape), 
                                             dtype = 'float', 
                                             data = self.acquisition_geometry.angles)
                ds_angles.attrs['units'] = self.acquisition_geometry.angle_unit
                
                
            
            
# usage example
xtek_file = '/home/evelina/nikon_data/SophiaBeads_256_averaged.xtekct'
reader = NikonDataReader()
reader.set_up(xtek_file = xtek_file,
              binning = [3, 1],
              roi = [200, 500, 1500, 2000],
              normalize = True)

data = reader.load_projections()
ag = reader.get_acquisition_geometry()

writer = NEXUSDataWriter()
writer.set_up(file_name = '/home/evelina/test_nexus.nxs',
              data_container = data,
              acquisition_geometry = ag)

writer.write_file()