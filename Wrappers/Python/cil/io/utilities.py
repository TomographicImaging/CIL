# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Author(s): Edoardo Pasca

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import json
import logging
import struct

logger = logging.getLogger(__name__)

def get_compress(compression=None):
    '''Returns whether the data needs to be compressed and to which numpy type
    
    Parameters:
    -----------
    compression : string, int. Default is None, no compression.
        It specifies the number of bits to use for compression, allowed values are None, 'uint8', 'uint16' and deprecated 0, 8, 16. 
    
    Returns:
    --------
    compress : bool, True if compression is required, False otherwise

    Note:
    -----

    The use of int is deprecated and will be removed in the future. Use string instead.
    
    '''
    if isinstance(compression, int):
        logger.warning("get_compress: The use of int is deprecated and will be removed in the future. Use string instead.")

    if compression is None or compression == 0:
        compress = False
    elif compression in [ 8, 'uint8']:
        compress = True
    elif compression in [ 16, 'uint16']:
        compress = True
    else:
        raise ValueError('Compression bits not valid. Got {0} expected value in {1}'.format(compression, [0,8,16, None, 'uint8', 'uint16']))

    return compress

def get_compressed_dtype(data, compression=None):
    '''Returns whether the data needs to be compressed and to which numpy type
    
    Given the data and the compression level, returns the numpy type to be used for compression.

    Parameters:
    -----------
    data : DataContainer, numpy array
        the data to be compressed
    compression : string, int. Default is None, no compression.
        It specifies the number of bits to use for compression, allowed values are None, 'uint8', 'uint16' and deprecated 0, 8, 16. 

    Returns:
    --------
    dtype : numpy type, the numpy type to be used for compression
    '''
    if isinstance(compression, int):
        logger.warning("get_compressed_dtype: The use of int is deprecated and will be removed in the future. Use string instead.")
    if compression is None or compression == 0:
        dtype = data.dtype
    elif compression in [ 8, 'uint8']:
        dtype = np.uint8
    elif compression in [ 16, 'uint16']:
        dtype = np.uint16
    else:
        raise ValueError('Compression bits not valid. Got {0} expected value in {1}'.format(compression, [0,8,16]))

    return dtype

def get_compression_scale_offset(data, compression=0):
    '''Returns the scale and offset to be applied to the data to compress it
    
    Parameters:
    -----------
    data : DataContainer, numpy array
        The data to be compressed
    compression : string, int. Default is None, no compression.
        It specifies the number of bits to use for compression, allowed values are None, 'uint8', 'uint16' and deprecated 0, 8, 16. 

    Returns:
    --------
    scale : float, the scale to be applied to the data for compression to the specified number of bits
    offset : float, the offset to be applied to the data for compression to the specified number of bits
    '''

    if isinstance(compression, int):
        logger.warning("get_compression_scale_offset: The use of int is deprecated and will be removed in the future. Use string instead.")

    if compression is None or compression == 0:
        # no compression
        # return scale 1.0 and offset 0.0
        return 1.0, 0.0

    dtype = get_compressed_dtype(data, compression)
    save_range = np.iinfo(dtype).max

    data_min = data.min()
    data_range = data.max() - data_min

    if data_range > 0:
        scale = save_range / data_range
        offset = - data_min * scale
    else:
        scale = 1.0
        offset = 0.0
    return scale, offset

def save_dict_to_file(fname, dictionary):
    '''Save scale and offset to file
    
    Parameters
    ----------
    fname : string
    dictionary : dictionary
        dictionary to write to file
    '''

    with open(fname, 'w') as configfile:
        json.dump(dictionary, configfile)

def compress_data(data, scale, offset, dtype):
    '''Compress data to dtype using scale and offset
    
    Parameters
    ----------
    data : numpy array
    scale : float
    offset : float
    dtype : numpy dtype
    
    returns compressed casted data'''
    if dtype == data.dtype:
        return data
    if data.ndim > 2:
        # compress each slice
        tmp = np.empty(data.shape, dtype=dtype)
        for i in range(data.shape[0]):
            tmp[i] = compress_data(data[i], scale, offset, dtype)
    else:
        tmp = data * scale + offset
        tmp = tmp.astype(dtype)
    return tmp
