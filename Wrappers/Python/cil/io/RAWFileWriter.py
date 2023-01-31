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
import numpy as np

logger = logging.getLogger(__name__)

def compress_and_save(data, compress, scale, offset, dtype, fname):
    '''Compress and save numpy array to file
    
    Parameters
    ----------
    data : numpy array
    compress : bool
    scale : float
    offset : float
    dtype : numpy dtype
    fname : string
    '''
    if compress:
        d = utilities.compress_data(data, scale, offset, dtype)
    else:
        d = data
    
    # This is slightly silly, but tofile will only export float (i.e. 64 bit)
    # which is not what we want.
    # So, we write a npy file with the correct type, then 
    # we copy only the binary part and dispose of the npy file.
    np.save(fname+'.npy', d)
    
    buffer_size = 1024*1024
    with open(fname+'.npy', 'rb') as f:
        
        vM, vm = np.lib.format.read_magic(f)
        if vM == 1:
            header = np.lib.format.read_array_header_1_0(f)
        elif vM == 2:
            header = np.lib.format.read_array_header_2_0(f)

        if header[0] != data.shape:
            raise ValueError('Shape mismatch')
        if header[2] != d.dtype:
            raise ValueError('dtype mismatch')

        with open(fname, 'wb') as f2:
            buffer = f.read(buffer_size)
            while buffer:
                f2.write(buffer)
                buffer = f.read(buffer_size)
    
    # finally remove the npy file.
    os.remove(fname+'.npy')
    return header
        



class RAWFileWriter(object):
    '''
        Writer to write DataSet to disk as a binary blob

        This writer will write a text file with the minimal information necessary to 
        read the data back in.
        
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
    
    def __init__(self, data, file_name, compression=None):
        
        if not isinstance(data, (ImageData, AcquisitionData) ):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        self.data_container = data
        file_name = os.path.abspath(file_name)
        self.file_name = os.path.splitext(
            os.path.basename( file_name )
            )[0]
        
        self.dir_name = os.path.dirname(file_name)
        logger.info ("dir_name {}".format(self.dir_name))
        logger.info ("file_name {}".format(self.file_name))
        
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
        header = \
            compress_and_save(self.data_container.as_array(), self.compress, self.scale, self.offset, self.dtype, fname)

        shape = header[0]
        fortran_order = header[1]
        read_dtype = header[2]
        if len(header) > 3:
            max_header_length = header[3]

        # save information about the file we just saved
        config = configparser.ConfigParser()
        config['MINIMAL INFO'] = {
            'file_name': self.file_name,
            'data_type': str(self.dtype),
            'shape': self.data_container.shape,
            # Data is always written in ‘C’ order, independent of the order of d. 
            'isFortran': fortran_order
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
