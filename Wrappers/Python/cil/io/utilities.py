#  Copyright 2023 United Kingdom Research and Innovation
#  Copyright 2023 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy as np
import json
import h5py
from warnings import warn


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
        warn("Use string instead of int", DeprecationWarning, stacklevel=2)

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
        warn("Use string instead of int", DeprecationWarning, stacklevel=2)
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
        warn("Use string instead of int", DeprecationWarning, stacklevel=2)

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

class HDF5_utilities(object):

    """
    Utility methods to read in from a generic HDF5 file and extract the relevant data
    """
    def __init__(self):
        pass


    @staticmethod
    def _descend_obj(obj, sep='\t', depth=-1):
        """
        Parameters
        ----------
        obj: str
            The initial group to print the metadata for
        sep: str, default '\t'
            The separator to use for the output
        depth: int
            depth to print from starting object. Values 1-N, if -1 will print all
        """
        if depth != 0:
            if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
                for key in obj.keys():
                    print(sep, '-', key, ':', obj[key])
                    HDF5_utilities._descend_obj(obj[key], sep=sep+'\t', depth=depth-1)
            elif type(obj) == h5py._hl.dataset.Dataset:
                for key in obj.attrs.keys():
                    print(sep+'\t', '-', key, ':', obj.attrs[key])


    @staticmethod
    def print_metadata(filename, group='/', depth=-1):
        """
        Prints the file metadata

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        group: (str), default: '/'
            a specific group to print the metadata for, this defaults to the root group
        depth: int, default -1
            depth of group to output the metadata for, -1 is fully recursive
        """
        with h5py.File(filename, 'r') as f:
            HDF5_utilities._descend_obj(f[group], depth=depth)


    @staticmethod
    def get_dataset_metadata(filename, dset_path):
        """
        Returns the dataset metadata as a dictionary

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        dset_path: str
            The internal path to the requested dataset

        Returns
        -------
        A dictionary containing keys, values are `None` if attribute can't be read.:
            ndim, shape, size, dtype, nbytes, compression, chunks, is_virtual
        """
        with h5py.File(filename, 'r') as f:
                dset = f.get(dset_path, )

                attribs = {
                    'ndim':None,
                    'shape':None,
                    'size':None,
                    'dtype':None,
                    'compression':None,
                    'chunks':None,
                    'is_virtual':None}

                for x in attribs.keys():
                    try:
                        attribs[x] = getattr(dset, x)
                    except AttributeError:
                        pass

                return attribs



    @staticmethod
    def read(filename, dset_path, source_sel=None, dtype=np.float32):
        """
        Reads a dataset entry and returns a numpy array with the requested data

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        dset_path: str
            The internal path to the requested dataset
        source_sel: tuple of slice objects, optional
            The selection of slices in each source dimension to return
        dtype: numpy type, default np.float32
            the numpy data type for the returned array


        Returns
        -------
        numpy.ndarray
            The requested data

        Note
        ----
        source_sel takes a tuple of slice objects to defining crop and slicing behaviour

        This can be constructed using numpy indexing, i.e. the following lines are equivalent.

        >>> source_sel = (slice(2, 4, None), slice(2, 10, 2))

        >>> source_sel = np.s_[2:4,2:10:2]
        """

        with h5py.File(filename, 'r') as f:
            dset = f.get(dset_path)

            if source_sel == None:
                source_sel = tuple([slice(None)]*dset.ndim)

            arr = np.asarray(dset[source_sel],dtype=dtype, order='C')

        return arr


    @staticmethod
    def read_to(filename, dset_path, out, source_sel=None, dest_sel=None):
        """
        Reads a dataset entry and directly fills a numpy array with the requested data

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        dset_path: str
            The internal path to the requested dataset
        out: numpy.ndarray
            The output array to be filled
        source_sel: tuple of slice objects, optional
            The selection of slices in each source dimension to return
        dest_sel: tuple of slice objects, optional
            The selection of slices in each destination dimension to fill


        Note
        ----
        source_sel and dest_sel take a tuple of slice objects to defining crop and slicing behaviour

        This can be constructed using numpy indexing, i.e. the following lines are equivalent.

        >>> source_sel = (slice(2, 4, None), slice(2, 10, 2))

        >>> source_sel = np.s_[2:4,2:10:2]
        """

        with h5py.File(filename, 'r') as f:
            dset = f.get(dset_path)
            dset.read_direct(out, source_sel, dest_sel)
