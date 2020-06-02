import numpy as np
from ccpi.io import *
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData

import os
import sys

from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import datetime
from PIL import Image

import functools


class TIFFWriter(object):
    
    def __init__(self,
                 **kwargs):
        
        self.data_container = kwargs.get('data_container', None)
        self.file_name = kwargs.get('file_name', None)
        counter_offset = kwargs.get('counter_offset', 0)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data_container = self.data_container,
                        file_name = self.file_name, 
                        counter_offset=counter_offset)
        
    def set_up(self,
               data_container = None,
               file_name = None,
               counter_offset = -1):
        
        self.data_container = data_container
        self.file_name = os.path.splitext(os.path.basename(file_name))[0]
        self.dir_name = os.path.dirname(file_name)
        self.counter_offset = counter_offset
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

    def write_file(self):
        '''alias of write'''
        return self.write()
    
    def write(self):
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)

        ndim = len(self.data_container.shape)
        if ndim == 2:
            # save single slice
            
            if self.counter_offset >= 0:
                fname = "{}_idx_{:04d}.tiff".format(os.path.join(self.dir_name, self.file_name), self.counter_offset)
            else:
                fname = "{}.tiff".format(os.path.join(self.dir_name, self.file_name))
            with open(fname, 'wb') as f:
                Image.fromarray(self.data_container.as_array()).save(f, 'tiff')
        elif ndim == 3:
            for sliceno in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = self.data_container.dimension_labels[0]
                fname = "{}_idx_{:04d}.tiff".format(
                    os.path.join(self.dir_name, self.file_name),
                    sliceno + self.counter_offset)
                with open(fname, 'wb') as f:
                    Image.fromarray(self.data_container.as_array()[sliceno]).save(f, 'tiff')
        elif ndim == 4:
            for sliceno1 in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = [ self.data_container.dimension_labels[0] ]
                for sliceno2 in range(self.data_container.shape[1]):
                    idx = self.data_container.shape[0] * sliceno2 + sliceno1 + self.counter_offset
                    fname = "{}_{}_{}_idx_{}.tiff".format(os.path.join(self.dir_name, self.file_name), 
                        self.data_container.shape[0], self.data_container.shape[1], idx)
                    with open(fname, 'wb') as f:
                        Image.fromarray(self.data_container.as_array()[sliceno1][sliceno2]).save(f, 'tiff')
        else:
            raise ValueError('Cannot handle more than 4 dimensions')
