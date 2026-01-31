#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
# Evan Kiely (Warwick Manufacturing Group, University of Warwick)

from cil.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
import os, re
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import warnings

pilAvailable = True
try:
    from PIL import Image
except:
    pilAvailable = False
import functools
import glob
import re
import numpy as np
from cil.io import utilities
import json

import logging

log = logging.getLogger(__name__)

def save_scale_offset(fname, scale, offset):
    '''Save scale and offset to file

    Parameters
    ----------
    fname : string
    scale : float
    offset : float
    '''
    dirname = os.path.dirname(fname)
    txt = os.path.join(dirname, 'scaleoffset.json')
    d = {'scale': scale, 'offset': offset}
    utilities.save_dict_to_file(txt, d)

class TIFFWriter(object):
    '''Write a DataSet to disk as a TIFF file or stack of TIFF files


        Parameters
        ----------
        data : DataContainer, AcquisitionData or ImageData
            This represents the data to save to TIFF file(s)
        file_name : string
            This defines the file name prefix, i.e. the file name without the extension.
        counter_offset : int, default 0.
            counter_offset indicates at which number the ordinal index should start.
            For instance, if you have to save 10 files the index would by default go from 0 to 9.
            By counter_offset you can offset the index: from `counter_offset` to `9+counter_offset`
        compression : str, default None. Accepted values None, 'uint8', 'uint16'
            The lossy compression to apply. The default None will not compress data.
            'uint8' or 'unit16' will compress to unsigned int 8 and 16 bit respectively.


        Note
        ----

          If compression ``uint8`` or ``unit16`` are used, the scale and offset used to compress the data are saved
          in a file called ``scaleoffset.json`` in the same directory as the TIFF file(s).

          The original data can be obtained by: ``original_data = (compressed_data - offset) / scale``

        Note
        ----

          In the case of 3D or 4D data this writer will save the data as a stack of multiple TIFF files,
          not as a single multi-page TIFF file.
        '''


    def __init__(self, data=None, file_name=None, counter_offset=0, compression=None):

        self.data_container = data
        self.file_name = file_name
        self.counter_offset = counter_offset
        if ((data is not None) and (file_name is not None)):
            self.set_up(data = data, file_name = file_name,
                        counter_offset=counter_offset,
                        compression=compression)

    def set_up(self,
               data = None,
               file_name = None,
               counter_offset = 0,
               compression=None):

        self.data_container = data
        file_name = os.path.abspath(file_name)
        self.file_name = os.path.splitext(
            os.path.basename(
                file_name
                )
            )[0]

        self.dir_name = os.path.dirname(file_name)
        log.info("dir_name %s", self.dir_name)
        log.info("file_name %s", self.file_name)
        self.counter_offset = counter_offset

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
        '''Write data to disk'''
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
                Image.fromarray(
                    utilities.compress_data(self.data_container.as_array() , self.scale, self.offset, self.dtype)
                    ).save(f, 'tiff')
        elif ndim == 3:
            for sliceno in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = self.data_container.dimension_labels[0]
                fname = "{}_idx_{:04d}.tiff".format(
                    os.path.join(self.dir_name, self.file_name),
                    sliceno + self.counter_offset)
                with open(fname, 'wb') as f:
                    Image.fromarray(
                            utilities.compress_data(self.data_container.as_array()[sliceno] , self.scale, self.offset, self.dtype)
                        ).save(f, 'tiff')
        elif ndim == 4:
            # find how many decimal places self.data_container.shape[0] and shape[1] have
            zero_padding = self._zero_padding(self.data_container.shape[0])
            zero_padding += '_' + self._zero_padding(self.data_container.shape[1])
            format_string = "{}_{}x{}x{}x{}_"+"{}.tiff".format(zero_padding)

            for sliceno1 in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = [ self.data_container.dimension_labels[0] ]
                for sliceno2 in range(self.data_container.shape[1]):
                    fname = format_string.format(os.path.join(self.dir_name, self.file_name),
                        self.data_container.shape[0], self.data_container.shape[1], self.data_container.shape[2],
                        self.data_container.shape[3] , sliceno1, sliceno2)
                    with open(fname, 'wb') as f:
                        Image.fromarray(
                            utilities.compress_data(self.data_container.as_array()[sliceno1][sliceno2] , self.scale, self.offset, self.dtype)
                        ).save(f, 'tiff')
        else:
            raise ValueError('Cannot handle more than 4 dimensions')
        if self.compress:
            save_scale_offset(fname, self.scale, self.offset)

    def _zero_padding(self, number):
        i = 0
        while 10**i < number:
            i+=1
        i+=1
        zero_padding_string = '{:0'+str(i)+'d}'
        return zero_padding_string


class TIFFStackReader(object):

    '''
        Basic TIFF reader which loops through all tiff files in a specific
        folder and loads them in alphabetical order

        Parameters
        ----------

        file_name : str, abspath to folder, list
            Path to folder with tiff files, list of paths of tiffs, or single tiff file

        roi : dictionary, default `None`
            dictionary with roi to load:
            ``{'axis_0': (start, end, step),
            'axis_1': (start, end, step),
            'axis_2': (start, end, step)}``
            roi is specified for axes before transpose.

        transpose : bool, default False
            Whether to transpose loaded images

        mode : str, {'bin', 'slice'}, default 'bin'.
            Defines the 'step' in the roi parameter:

            In bin mode, 'step' number of pixels
            are binned together, values of resulting binned pixels are calculated as average.

            In 'slice' mode 'step' defines standard numpy slicing.

            Note: in general output array size in bin mode != output array size in slice mode

        file_prefix : str, default None
            Leading string for the tiff files to be read. Used only when the file_name 
            is a path to a folder, if None all files in the folder are loaded. 
        
        dtype : numpy type, string, default np.float32
            Requested type of the read image. If set to None it defaults to the type of the saved file.


        Notes:
        ------
        roi behaviour:
            Files are stacked along ``axis_0``, in alphabetical order.

            ``axis_1`` and ``axis_2`` correspond
            to row and column dimensions, respectively.

            To skip files or to change number of files to load,
            adjust ``axis_0``. For instance, ``'axis_0': (100, 300)``
            will skip first 100 files and will load 200 files.

            ``'axis_0': -1`` is a shortcut to load all elements along axis 0.

            ``start`` and ``end`` can be specified as ``None`` which is equivalent
            to ``start = 0`` and ``end = load everything to the end``, respectively.

            Start and end also can be negative.

            roi is specified for axes before transpose.


        Example:
        --------
        You can rescale the read data as `rescaled_data = (read_data - offset)/scale` with the following code:

        >>> reader = TIFFStackReader(file_name = '/path/to/folder')
        >>> rescaled_data = reader.read_rescaled(scale, offset)


        Alternatively, if TIFFWriter has been used to save data with lossy compression, then you can rescale the
        read data to approximately the original data with the following code:

        >>> writer = TIFFWriter(file_name = '/path/to/folder', data=original_data, compression='uint8')
        >>> writer.write()
        >>> reader = TIFFStackReader(file_name = '/path/to/folder')
        >>> about_original_data = reader.read_rescaled()
    '''

    def __init__(self, file_name=None, roi=None, transpose=False, mode='bin', file_prefix = None, dtype=np.float32):    
            self.file_name = file_name
            
            if self.file_name is not None:
                self.set_up(file_name = self.file_name,
                            roi = roi,
                            transpose = transpose,
                            file_prefix = file_prefix,
                            mode = mode, dtype=dtype)
                
    def set_up(self, 
            file_name = None,
            roi = None,
            transpose = False,
            mode = 'bin', 
            file_prefix = None,
            dtype = np.float32):
        '''
        Set up method for the TIFFStackReader class

        Parameters
        ----------

        file_name : str, abspath to folder, list
            Path to folder with tiff files, list of paths of tiffs, or single tiff file

        roi : dictionary, default `None`
            dictionary with roi to load
            ``{'axis_0': (start, end, step), 'axis_1': (start, end, step), 'axis_2': (start, end, step)}``
            Files are stacked along axis_0. axis_1 and axis_2 correspond
            to row and column dimensions, respectively.
            Files are stacked in alphabetic order.
            To skip files or to change number of files to load,
            adjust axis_0. For instance, 'axis_0': (100, 300)
            will skip first 100 files and will load 200 files.
            'axis_0': -1 is a shortcut to load all elements along axis.
            Start and end can be specified as None which is equivalent
            to start = 0 and end = load everything to the end, respectively.
            Start and end also can be negative.
            Notes: roi is specified for axes before transpose.

        transpose : bool, default False
            Whether to transpose loaded images

        mode : str, default 'bin'. Accepted values 'bin', 'slice'
            Referring to the 'step' defined in the roi parameter, in bin mode, 'step' number of pixels
            are binned together, values of resulting binned pixels are calculated as average.
            In 'slice' mode 'step' defines standard numpy slicing.
            Note: in general output array size in bin mode != output array size in slice mode

        file_prefix : str, default None
            Leading string for the tiff files to be read. Used only when the file_name 
            is a path to a folder, if None all files in the folder are loaded. 

        dtype : numpy type, string, default np.float32
            Requested type of the read image. If set to None it defaults to the type of the saved file.

        '''
        self.roi = roi
        self.transpose = transpose
        self.mode = mode
        self.dtype = dtype

        if file_name == None:
            raise ValueError('file_name to tiff files is required. Can be a tiff, a list of tiffs or a directory containing tiffs')

        if self.roi is None:
            self.roi = {'axis_0': -1, 'axis_1': -1, 'axis_2': -1}

        # check that PIL library is installed
        if (pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files.")

        # check labels
        for key in self.roi.keys():
            if key not in ['axis_0', 'axis_1', 'axis_2']:
                raise Exception("Wrong label. axis_0, axis_1 and axis_2 are expected")

        if self.mode not in ['bin', 'slice']:
            raise ValueError("Wrong mode, bin or slice is expected.")

        self._roi = self.roi.copy()

        if 'axis_0' not in self._roi.keys():
            self._roi['axis_0'] = -1

        if 'axis_1' not in self._roi.keys():
            self._roi['axis_1'] = -1

        if 'axis_2' not in self._roi.keys():
            self._roi['axis_2'] = -1


        if isinstance(file_name, list):
            self._tiff_files = file_name
            if file_prefix is not None:
                warnings.warn(f"file_prefix: {file_prefix} is not used with a list of tiffs", stacklevel=2)
        
        elif os.path.isfile(file_name):
            self._tiff_files = [file_name]
            if file_prefix is not None:
                warnings.warn(f"file_prefix: {file_prefix} is not used with a single tiff", stacklevel=2)
        
        elif os.path.isdir(file_name):
            if file_prefix == None:
                file_prefix = ''

            self._tiff_files = glob.glob(os.path.join(glob.escape(file_name),file_prefix + "*.tif"))
            
            if not self._tiff_files:
                self._tiff_files = glob.glob(os.path.join(glob.escape(file_name),file_prefix + "*.tiff"))

            if not self._tiff_files:
                if file_prefix == '':
                    raise Exception("No tiff files were found in the directory \n{}".format(file_name))
                else:
                    raise Exception("No tiff files with prefix {} were found in the directory \n{}".format(file_prefix, file_name))
                
        else:
            raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))


        for fn in self._tiff_files:
            if '.tif' in fn:
                if not(os.path.exists(fn)):
                    raise Exception('File \n {}\n does not exist.'.format(fn))
            else:
                raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))


        self._tiff_files.sort(key=self.__natural_keys)

    def _get_file_type(self, img):
        mode = img.mode
        if mode == '1':
            dtype = np.bool_
        elif mode == 'L':
            dtype = np.uint8
        elif mode == 'F':
            dtype = np.float32
        elif mode == 'I':
            dtype = np.int32
        elif mode in ['I;16']:
            dtype = np.uint16
        else:
            raise ValueError("Unsupported type {}. Expected any of 1 L I I;16 F.".format(mode))
        return dtype

    def read(self):

        '''
        Reads images and return numpy array
        '''
        # load first image to find out dimensions and type
        filename = os.path.abspath(self._tiff_files[0])

        with Image.open(filename) as img:
            if self.dtype is None:
                self.dtype = self._get_file_type(img)
            tmp = np.asarray(img, dtype = self.dtype)

        array_shape_0 = (len(self._tiff_files), tmp.shape[0], tmp.shape[1])

        roi_par = [[0, array_shape_0[0], 1], [0, array_shape_0[1], 1], [0, array_shape_0[2], 1]]

        for key in self._roi.keys():
            if key == 'axis_0':
                idx = 0
            elif key == 'axis_1':
                idx = 1
            elif key == 'axis_2':
                idx = 2
            if self._roi[key] != -1:
                for i in range(2):
                    if self._roi[key][i] != None:
                        if self._roi[key][i] >= 0:
                            roi_par[idx][i] = self._roi[key][i]
                        else:
                            roi_par[idx][i] = roi_par[idx][1]+self._roi[key][i]
                if len(self._roi[key]) > 2:
                    if self._roi[key][2] != None:
                        if self._roi[key][2] > 0:
                            roi_par[idx][2] = self._roi[key][2]
                        else:
                            raise Exception("Negative step is not allowed")

        if self.mode == 'bin':
            # calculate number of pixels
            n_rows = (roi_par[1][1] - roi_par[1][0]) // roi_par[1][2]
            n_cols = (roi_par[2][1] - roi_par[2][0]) // roi_par[2][2]
            num_to_read = (roi_par[0][1] - roi_par[0][0]) // roi_par[0][2]

            if not self.transpose:
                im = np.zeros((num_to_read, n_rows, n_cols), dtype=self.dtype)
            else:
                im = np.zeros((num_to_read, n_cols, n_rows), dtype=self.dtype)

            for i in range(0,num_to_read):

                raw = np.zeros((array_shape_0[1], array_shape_0[2]), dtype=self.dtype)
                for j in range(roi_par[0][2]):

                    index = int(roi_par[0][0] + i * roi_par[0][2] + j)
                    filename = os.path.abspath(self._tiff_files[index])

                    arr = Image.open(filename)
                    raw += np.asarray(arr, dtype = self.dtype)


                shape = (n_rows, roi_par[1][2],
                         n_cols, roi_par[2][2])
                tmp = raw[roi_par[1][0]:(roi_par[1][0] + (((roi_par[1][1] - roi_par[1][0]) // roi_par[1][2]) * roi_par[1][2])), \
                          roi_par[2][0]:(roi_par[2][0] + (((roi_par[2][1] - roi_par[2][0]) // roi_par[2][2]) * roi_par[2][2]))].reshape(shape).mean(-1).mean(1)

                if self.transpose:
                    im[i, :, :] = np.transpose(tmp)
                else:
                    im[i, :, :] = tmp

        else: # slice mode
            # calculate number of pixels
            n_rows = int(np.ceil((roi_par[1][1] - roi_par[1][0]) / roi_par[1][2]))
            n_cols = int(np.ceil((roi_par[2][1] - roi_par[2][0]) / roi_par[2][2]))
            num_to_read = int(np.ceil((roi_par[0][1] - roi_par[0][0]) / roi_par[0][2]))

            if not self.transpose:
                im = np.zeros((num_to_read, n_rows, n_cols), dtype=self.dtype)
            else:
                im = np.zeros((num_to_read, n_cols, n_rows), dtype=self.dtype)

            for i in range(roi_par[0][0], roi_par[0][1], roi_par[0][2]):

                filename = os.path.abspath(self._tiff_files[i])
                #try:
                raw = np.asarray(Image.open(filename), dtype = self.dtype)
                #except:
                #    print('Error reading\n {}\n file.'.format(filename))
                #    raise

                tmp = raw[(slice(roi_par[1][0], roi_par[1][1], roi_par[1][2]),
                           slice(roi_par[2][0], roi_par[2][1], roi_par[2][2]))]

                if self.transpose:
                    im[(i - roi_par[0][0]) // roi_par[0][2], :, :] = np.transpose(tmp)
                else:
                    im[(i - roi_par[0][0]) // roi_par[0][2], :, :] = tmp

        return np.squeeze(im)

    def __atoi(self, text):
        return int(text) if text.isdigit() else text

    def __natural_keys(self, text):
        '''
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.__atoi(c) for c in re.split(r'(\d+)', text) ]

    def _read_as(self, geometry):
        '''reads the TIFF stack as an ImageData with the provided geometry'''
        data = self.read()
        if len(geometry.shape) == 4:
            gsize = np.prod(geometry.shape)
            dsize = np.prod(data.shape)
            if gsize != dsize:
                added_dims = len(geometry.shape) - len(data.shape)
                if data.shape[0] != functools.reduce(lambda x,y: x*y, geometry.shape[:1+added_dims], 1):
                    raise ValueError("Cannot reshape read data {} to the requested shape {}.\n"\
                        .format(data.shape, geometry.shape) +
                                    "Geometry requests first dimension of data to be {} but it is {}"\
                            .format(geometry.shape[0]*geometry.shape[1], data.shape[0] ))
                raise ValueError('data {} and requested {} shapes are not compatible: data size does not match! Expected {}, got {}'\
                    .format(data.shape, geometry.shape, dsize, gsize))
            if len(data.shape) != 3:
                raise ValueError("Data should have 3 dimensions, got {}".format(len(data.shape)))


            reshaped = np.reshape(data, geometry.shape)
            return self._return_appropriate_data(reshaped, geometry)

        if data.shape != geometry.shape:
            raise ValueError('Requested {} shape is incompatible with data. Expected {}, got {}'\
                .format(geometry.__class__.__name__, data.shape, geometry.shape))

        return self._return_appropriate_data(data, geometry)

    def _return_appropriate_data(self, data, geometry):
        if isinstance (geometry, ImageGeometry):
            return ImageData(data, deep_copy=True, geometry=geometry.copy())
        elif isinstance (geometry, AcquisitionGeometry):
            return AcquisitionData(data, deep_copy=True, geometry=geometry.copy())
        else:
            raise TypeError("Unsupported Geometry type. Expected ImageGeometry or AcquisitionGeometry, got {}"\
                .format(type(geometry)))

    def read_as_ImageData(self, image_geometry):
        '''reads the TIFF stack as an ImageData with the provided geometry

        Notice that the data will be reshaped to what requested in the geometry but there is
        no warranty that the data will be read in the right order!
        In facts you can reshape a (2,3,4) array as (3,4,2), however we do not check if the reshape
        leads to sensible data.
        '''
        return self._read_as(image_geometry)
    def read_as_AcquisitionData(self, acquisition_geometry):
        '''reads the TIFF stack as an AcquisitionData with the provided geometry

        Notice that the data will be reshaped to what requested in the geometry but there is
        no warranty that the data will be read in the right order!
        In facts you can reshape a (2,3,4) array as (3,4,2), however we do not check if the reshape
        leads to sensible data.
        '''
        return self._read_as(acquisition_geometry)

    def read_scale_offset(self):
        '''Reads the scale and offset from a json file in the same folder as the tiff stack

        This is a courtesy method that will work only if the tiff stack is saved with the TIFFWriter

        Returns:
        --------

        tuple: (scale, offset)
        '''
        # load first image to find out dimensions and type
        path = os.path.dirname(self._tiff_files[0])
        with open(os.path.join(path, "scaleoffset.json"), 'r') as f:
            d = json.load(f)

        return (d['scale'], d['offset'])

    def read_rescaled(self, scale=None, offset=None):
        '''Reads the TIFF stack and rescales it with the provided scale and offset, or with the ones in the json file if not provided

        This is a courtesy method that will work only if the tiff stack is saved with the TIFFWriter

        Parameters:
        -----------

        scale: float, default None
            scale to apply to the data. If None, the scale will be read from the json file saved by TIFFWriter.
        offset: float, default None
            offset to apply to the data. If None, the offset will be read from the json file saved by TIFFWriter.

        Returns:
        --------

        numpy.ndarray in float32
        '''
        data = self.read()
        if scale is None or offset is None:
            scale, offset = self.read_scale_offset()
        if self.dtype != np.float32:
            data = data.astype(np.float32)
        data -= offset
        data /= scale
        return data
