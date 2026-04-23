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

from cil.framework import AcquisitionData, ImageData, DataContainer
import os
from cil.io import utilities
import configparser

import logging

log = logging.getLogger(__name__)


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

    Note:
    -----
    Data is always written in ‘C’ order, independent of the order of d.
    '''
    if compress:
        d = utilities.compress_data(data, scale, offset, dtype)
    else:
        d = data

    log.info(
        "Data is always written in ‘C’ order, independent of the order of d.")
    d.tofile(fname)

    # return shape, fortran order, dtype
    return d.shape, False, d.dtype.str


class RAWFileWriter(object):
    '''
        Writer to write DataContainer (or subclass AcquisitionData, ImageData) to disk as a binary blob

        Parameters
        ----------
        data : DataContainer, AcquisitionData or ImageData
            This represents the data to save to TIFF file(s)
        file_name : string
            This defines the file name prefix, i.e. the file name without the extension.
        compression : str, default None. Accepted values None, 'uint8', 'uint16'
            The lossy compression to apply. The default None will not compress data.
            'uint8' or 'unit16' will compress to unsigned int 8 and 16 bit respectively.


        This writer will also write a text file with the minimal information necessary to
        read the data back in. This text file will need to reside in the same directory as the raw file.

        The text file will look something like this::

            [MINIMAL INFO]
            file_name = filename.raw
            data_type = <u2
            shape = (6, 5, 4)
            is_fortran = False

            [COMPRESSION]
            scale = 550.7142857142857
            offset = -0.0

        The ``data_type`` describes the data layout when packing and unpacking data. This can be
        read as numpy dtype with ``np.dtype('<u2')``.


        Example
        -------

        Example of using the writer with compression to ``uint8``:

        >>> from cil.io import RAWFileWriter
        >>> writer = RAWFileWriter(data=data, file_name=fname, compression='uint8')
        >>> writer.write()

        Example
        -------

        Example of reading the data from the ini file:

        >>> config = configparser.ConfigParser()
        >>> inifname = "file_name.ini"
        >>> config.read(inifname)
        >>> read_dtype = config['MINIMAL INFO']['data_type']
        >>> dtype = np.dtype(read_dtype)
        >>> fname = config['MINIMAL INFO']['file_name']
        >>> read_array = np.fromfile(fname, dtype=read_dtype)
        >>> read_shape = eval(config['MINIMAL INFO']['shape'])
        >>> scale = float(config['COMPRESSION']['scale'])
        >>> offset = float(config['COMPRESSION']['offset'])

        Note
        ----

          If compression ``uint8`` or ``unit16`` are used, the scale and offset used to compress the data are saved
          in the ``ini`` file in the same directory as the raw file, in the "COMPRESSION" section .

          The original data can be obtained by: ``original_data = (compressed_data - offset) / scale``

        Note
        ----

          Data is always written in ‘C’ order independent of the order of the original data,
          https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile,


    '''

    def __init__(self, data, file_name, compression=None):

        if not isinstance(data, (DataContainer, ImageData, AcquisitionData)):
            raise Exception('Writer supports only following data types:\n' +
                            'DataContainer - ImageData\n - AcquisitionData')

        self.data_container = data
        file_name = os.path.abspath(file_name)
        self.file_name = os.path.splitext(os.path.basename(file_name))[0]

        self.dir_name = os.path.dirname(file_name)
        log.info("dir_name %s", self.dir_name)
        log.info("file_name %s", self.file_name)

        # Deal with compression
        self.compress = utilities.get_compress(compression)
        self.dtype = utilities.get_compressed_dtype(data, compression)
        self.scale, self.offset = utilities.get_compression_scale_offset(
            data, compression)
        self.compression = compression

    def write(self):
        '''Write data to disk'''
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)

        fname = os.path.join(self.dir_name, self.file_name + '.raw')

        # write to disk
        header = \
            compress_and_save(self.data_container.as_array(), self.compress, self.scale, self.offset, self.dtype, fname)

        shape = header[0]
        fortran_order = header[1]
        read_dtype = header[2]

        # save information about the file we just saved
        config = configparser.ConfigParser()
        config['MINIMAL INFO'] = {
            'file_name': os.path.basename(fname),
            'data_type': read_dtype,
            'shape': shape,
            # Data is always written in ‘C’ order, independent of the order of d.
            'is_fortran': fortran_order
        }

        if self.compress:
            config['COMPRESSION'] = {
                'scale': self.scale,
                'offset': self.offset,
            }
        log.info("Saving to %s", self.file_name)
        log.info(str(config))
        # write the configuration to an ini file
        with open(os.path.join(self.dir_name, self.file_name + '.ini'),
                  'w') as configfile:
            config.write(configfile)
