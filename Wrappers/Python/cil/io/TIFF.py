# -*- coding: utf-8 -*-
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

from cil.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
import os, re
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry

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
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
        folder and loads them in alphabetic order
        
        Parameters
        ----------
            
        file_name : str, abspath to folder, list
            Path to folder with tiff files, list of paths of tiffs, or single tiff file
                   
        transpose : bool, default False
            Whether to transpose loaded images
                    
        dtype : numpy type, string, default np.float32
            Requested type of the read image. If set to None it defaults to the type of the saved file.

        deprecated_kwargs
        -----------------
        
        roi : dictionary, default `None`, deprecated
            dictionary with roi to load: 
            ``{'axis_0': (start, end, step), 
               'axis_1': (start, end, step), 
               'axis_2': (start, end, step)}``
            roi is specified for axes before transpose. Use `set_image_roi()` and `set_panel_roi()` instead.            

        mode : str, {'bin', 'slice'}, default 'bin', deprecated
            Use `set_image_roi()` to set the 'bin'/'slicing' behaviour.  

            Defines the 'step' in the roi parameter:
            
            In bin mode, 'step' number of pixels 
            are binned together, values of resulting binned pixels are calculated as average.

            In 'slice' mode 'step' defines standard numpy slicing.

            Note: in general output array size in bin mode != output array size in slice mode
        

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

        >>> writer = TIFFWriter(file_name = '/path/to/folder', compression='uint8')
        >>> writer.write(original_data)
        >>> reader = TIFFStackReader(file_name = '/path/to/folder')
        >>> about_original_data = reader.read_rescaled()
    '''

    def _deprecated_kwargs(self, deprecated_kwargs):
        # handle deprecated behaviour for backward compatibility
        mode = 'bin'
        if deprecated_kwargs.get('mode', False):
            logging.warning("Input argument `mode` has been deprecated. Please define binning/slicing with method 'set_image_roi()' instead")
            mode = deprecated_kwargs.pop('mode')

        if deprecated_kwargs.get('roi', False):
            logging.warning("Input argument `roi` has been deprecated. Please use methods 'set_image_roi()' and 'set_frame_indices()' instead")
            roi = deprecated_kwargs.pop('roi')
            self.set_frame_indices(roi.get('axis_0')) 
            self.set_image_roi(roi.get('axis_1'), roi.get('axis_2'), mode) 

        if deprecated_kwargs.pop('transpose', None) is not None:
            logging.warning("Input argument `transpose` has been deprecated. Please define your geometry accordingly")

        if deprecated_kwargs:
            logging.warning("Additional keyword arguments passed but not used: {}".format(deprecated_kwargs))


    def __init__(self, file_name=None, dtype=np.float32, **deprecated_kwargs):    
        
        # check that PIL library is installed
        if (pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files.")

        if file_name is not None:
            self.set_up(file_name = file_name,
                        dtype=dtype,
                        **deprecated_kwargs)


    def set_up(self, 
               file_name = None,
               dtype = np.float32,
               **deprecated_kwargs):
        

        self._deprecated_kwargs(deprecated_kwargs)
     
        self.set_file_name(file_name)
        self.dtype = dtype

    @property
    def file_name(self):
        return self._file_name
    

    def set_file_name(self, file_name):

        if file_name == None:
            raise ValueError('file_name to tiff files is required. Can be a tiff, a list of tiffs or a directory containing tiffs')

        if isinstance(file_name, list):
            self._tiff_files = file_name
        elif os.path.isfile(file_name):
            self._tiff_files = [file_name]
        elif os.path.isdir(file_name): 
            self._tiff_files = glob.glob(os.path.join(glob.escape(file_name),"*.tif"))
            
            if not self._tiff_files:
                self._tiff_files = glob.glob(os.path.join(glob.escape(file_name),"*.tiff"))

            if not self._tiff_files:
                raise Exception("No tiff files were found in the directory \n{}".format(file_name))

        else:
            raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))


        for i, fn in enumerate(self._tiff_files):
            if '.tif' in fn:
                if not(os.path.exists(fn)):
                    raise Exception('File \n {}\n does not exist.'.format(fn))
                self._tiff_files[i] = os.path.abspath(fn)
            else:
                raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))

        self._tiff_files.sort(key=self.__natural_keys)

        #reconfigure with new path
        self._configure()


    def _configure(self):
        """
        initialises the shape and dtype based on the TIFFs at the path
        """
    
        # load first image to find out dimensions and type
        with Image.open(self._tiff_files[0]) as img:
            self._imag_dtype = self._get_file_type(img)
            w, h = img.size

        self._image_shape = [h,w]
        self._num_images = len(self._tiff_files)
        self._roi_full = ((0,self._num_images,1), (0,h,1), (0,w,1))
        self.reset_roi()


    def reset_roi(self):
        """
        members that store the requested downsizing
        """
        self._roi_crop = [(0,self._image_shape[0]), (0,self._image_shape[1])]
        self._shape_downsample = [*self._image_shape]
        self._method_downsample = 'slice'
        self._frame_indices = np.arange(*(0,self._num_images,1))


    def set_image_roi(self, roi_height=None, roi_width=None,  mode='slice'):
        """
        roi as int, slice,
        mode 'slice', 'bin'
        """

        if mode not in ['bin', 'slice']:
            raise ValueError("Wrong mode, bin or slice is expected.")

        for ind, roi in enumerate([roi_height, roi_width]):

            if roi == -1 or roi is None:
                roi_slice = slice(None)
            elif isinstance(roi,tuple):
                roi_slice = slice(*roi)
            elif isinstance(roi, int):
                roi_slice = slice(int(roi),int(roi)+1,1)
            elif isinstance(roi ,slice):
                roi_slice = roi
            else:
                raise ValueError("roi not understood")
                    
            axis_range = range(*self._roi_full[ind+1])[roi_slice]

            if mode == 'slice':                
                axis_length = int(np.ceil((axis_range.stop - axis_range.start) / axis_range.step))
            else:
                axis_length = (axis_range.stop - axis_range.start) // axis_range.step
            
            self._roi_crop[ind] = (axis_range.start, axis_range.start + axis_length * axis_range.step)
            self._shape_downsample[ind] = axis_length

        self._method_downsample = mode


    def set_frame_indices(self, indices=None):
        """
        Method to configure the angular indices to be returned

        angle_indices: takes an integer for a single frame, a tuple of (start, stop, step), 
        or a list of frame indices.

        'slice' mode only 
        """      

        if isinstance(indices, (list, np.ndarray)):
            try:
                indices = np.arange(*self._roi_full[0]).take(indices)
            except IndexError:
                raise ValueError("Index out of range")   

        else:
            if indices == -1 or indices is None:
                index_slice = slice(None)
            elif isinstance(indices, int):
                index_slice = slice(int(indices),int(indices)+1)
            elif isinstance(indices,tuple):
                index_slice = slice(*indices)
            elif isinstance(indices ,slice):
                index_slice = indices

            indices = np.arange(*self._roi_full[0])[index_slice]

            if indices.size < 1:
                raise ValueError("No frames selected. Please select at least 1 frame")
            
        self._frame_indices = indices


    def __getitem__(self, val):
        """
        pass slice object and return data chunk
        """

        if val == None:
            val = (slice(None),slice(None),slice(None))

        tmp_roi_crop = self._roi_crop.copy()
        tmp_shape_downsample = self._shape_downsample.copy()
        tmp_method_downsample = self._method_downsample
        tmp_frame_indices = self._frame_indices.copy()

        if self._num_images > 1:
            self.set_frame_indices(val[0])
            self.set_image_roi(val[1],val[2],'slice')
            array = self.read()
        else:
            self.set_frame_indices(0)
            self.set_image_roi(val[0],val[1],'slice')
            array = self.read()

        self._roi_crop = tmp_roi_crop
        self._shape_downsample = tmp_shape_downsample
        self._method_downsample = tmp_method_downsample
        self._frame_indices = tmp_frame_indices

        return array


    def read(self):
        
        '''
        Reads images and return numpy array
        '''
        if self.dtype is None:
            self.dtype = self._imag_dtype


        # single frame crop size, left, top, right, bottom
        crop_box = (self._roi_crop[1][0],self._roi_crop[0][0],self._roi_crop[1][1],self._roi_crop[0][1])

        # create empty data container for downsized array
        array = np.empty((len(self._frame_indices),*self._shape_downsample), dtype=self.dtype)

        count = 0
        for i in self._frame_indices:
                
            #read roi from single projection
            with Image.open(self._tiff_files[i], mode='r', formats=(['tiff'])) as img:
                img = img.crop(crop_box)

                if self._method_downsample == 'slice':
                    img = img.resize((self._shape_downsample[1],self._shape_downsample[0]), Image.NEAREST)
                else:
                    img = img.resize((self._shape_downsample[1],self._shape_downsample[0]), Image.BILINEAR)

                frame_out = np.asarray(img, dtype=self.dtype)

                array[count] = frame_out
            count+=1
        return np.squeeze(array)
        

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
        '''reads the data as an ImageData or AcquisitionData with the provided geometry'''

        data = self.read()
        

        try:
            data.shape = geometry.shape
        except AssertionError:
            raise ValueError('data {} and requested {} shapes are not compatible'\
                    .format(data.shape, geometry.shape))

        if data.shape != geometry.shape:
            raise ValueError('Requested {} shape is incompatible with data. Expected {}, got {}'\
                .format(geometry.__class__.__name__, data.shape, geometry.shape))

        if self.dimension_labels != geometry.dimension_labels:
            raise ValueError('Requested geometry is ordered differently to dataset. Expected {}, got {}'\
                .format(self.dimension_labels, geometry.shape))            

        return self._return_appropriate_data(data, geometry.dimension_labels)
    

    def _return_appropriate_data(self, data, geometry):
        if isinstance (geometry, ImageGeometry):
            return ImageData(data, deep=False, geometry=geometry.copy(), suppress_warning=True)
        elif isinstance (geometry, AcquisitionGeometry):
            return AcquisitionData(data, deep=False, geometry=geometry.copy(), suppress_warning=True)
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
