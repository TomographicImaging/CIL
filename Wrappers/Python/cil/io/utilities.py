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

def get_compression(data, compression=0):
    '''Returns whether the data needs to be compressed and to which numpy type'''
    if compression == 0:
        dtype = data.dtype
        compress = False
    elif compression == 8:
        dtype = np.uint8
        compress = True
    elif compression == 16:
        dtype = np.uint16
        compress = True
    else:
        raise ValueError('Compression bits not valid. Got {0} expected value in {1}'.format(compression, [0,8,16]))

    return compress, dtype

def get_compression_scale_offset(data, compression=0):
    '''Returns the scale and offset to be applied to the data to compress it'''

    compress, dtype = get_compression(data, compression)
    if compression == 0:
        # no compression
        # return scale 1.0 and offset 0.0
        return 1.0, 0.0
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