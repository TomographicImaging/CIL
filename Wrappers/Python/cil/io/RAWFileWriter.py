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
#   limitations under the License.import numpy as np

from cil.framework import AcquisitionData, ImageData
import os, re
import sys
from cil.framework import AcquisitionData, ImageData
import re
from cil.io import utilities
import configparser

import logging

logger = logging.getLogger(__name__)

def compress_and_save(data, compress, scale, offset, dtype, fname):
    '''Compress and save numpy array to file
    
    Parameters
    ----------
    data : numpy array
    scale : float
    offset : float
    dtype : numpy dtype
    fname : string
    '''
    if compress:
        d = utilities.compress_data(data, scale, offset, dtype)
    else:
        d = data
    # Data is always written in ‘C’ order, independent of the order of d. 
    d.tofile(fname)    

class RAWFileWriter(object):
    '''Write a DataSet to disk as a binary blob'''
    
    def __init__(self,
                 **kwargs):
        '''

        Parameters
        ----------
        data : DataContainer, AcquisitionData or ImageData
            This represents the data to save to TIFF file(s)
        file_name : string
            This defines the file name prefix, i.e. the file name without the extension.
        compression : str, default None. Accepted values None, 'uint8', 'uint16'
            The lossy compression to apply. The default None will not compress data. 
            'uint8' or 'unit16' will compress to unsigned int 8 and 16 bit respectively.


        Note:
        -----

        If compression 'uint8' or 'unit16' are used, the scale and offset used to compress the data are saved 
        in a file called `scaleoffset.json` in the same directory as the TIFF file(s).

        The original data can be obtained by: `original_data = (compressed_data - offset) / scale`
        '''
        
        self.data_container = kwargs.get('data', None)
        self.file_name = kwargs.get('file_name', None)
        compression = kwargs.get('compression', None)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data = self.data_container,
                        file_name = self.file_name, 
                        compression=compression)
        
    def set_up(self,
               data = None,
               file_name = None,
               compression=0):
        
        self.data_container = data
        file_name = os.path.abspath(file_name)
        self.file_name = os.path.splitext(
            os.path.basename(
                file_name
                )
            )[0]
        
        self.dir_name = os.path.dirname(file_name)
        logger.info ("dir_name {}".format(self.dir_name))
        logger.info ("file_name {}".format(self.file_name))
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        # Deal with compression
        self.compress           = utilities.get_compress(compression)
        self.dtype              = utilities.get_compressed_dtype(data, compression)
        self.scale, self.offset = utilities.get_compression_scale_offset(data, compression)
        self.compression        = compression

    
    def write(self):
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)
        
        fname = os.path.join(self.dir_name, self.file_name + '.raw')

        # write to disk
        compress_and_save(self.data_container.as_array(), self.compress, self.scale, self.offset, self.dtype, fname)

        # save information about the file we just saved
        config = configparser.ConfigParser()
        config['MINIMAL INFO'] = {
            'file_name': self.file_name,
            'data_type': str(self.dtype),
            'shape': self.data_container.shape,
            # Data is always written in ‘C’ order, independent of the order of d. 
            'isFortran': False
        }
        
        if self.compress:
            config['COMPRESSION'] = {
                'scale': self.scale,
                'offset': self.offset,
            }
        logging.info("Saving to {}".format(self.file_name))
        logging.info(str(config))
        # write the configuration to an ini file
        with open(os.path.join(self.dir_name, self.file_name + '.ini'), 'w') as configfile:
            config.write(configfile)
