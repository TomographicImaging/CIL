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
from cil.io import *
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
import os, re
import sys
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import datetime
pilAvailable = True
try:    
    from PIL import Image
except:
    pilAvailable = False
import functools
import glob
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TIFFWriter(object):
    '''Write a DataSet to disk as a TIFF file or stack'''
    
    def __init__(self,
                 **kwargs):
        '''
        :param data: the data to save to TIFF file(s)
        :type data: DataContainer, AcquisitionData or ImageData
        :param file_name: string defining the file name prefix
        :type file_name: string
        :param counter_offset: int indicating at which number the index starts. 
          For instance, if you have to save 10 files the index would by default go from 0 to 9.
          By counter_offset you can offset the index: from `counter_offset` to `9+counter_offset`
        :type counter_offset: int, default 0
        '''
        
        self.data_container = kwargs.get('data', None)
        self.file_name = kwargs.get('file_name', None)
        counter_offset = kwargs.get('counter_offset', 0)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data = self.data_container,
                        file_name = self.file_name, 
                        counter_offset=counter_offset)
        
    def set_up(self,
               data = None,
               file_name = None,
               counter_offset = -1):
        
        self.data_container = data
        file_name = os.path.abspath(file_name)
        self.file_name = os.path.splitext(
            os.path.basename(
                file_name
                )
            )[0]
        logger.info("saving to file_name", self.file_name)
        self.dir_name = os.path.dirname(file_name)
        logger.info("dir_name" , self.dir_name, self.dir_name is None)
        self.counter_offset = counter_offset
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
    
    def write(self):
        if not os.path.isdir(self.dir_name):
            logger.info('creating directory', self.dir_name)
            os.mkdir(self.dir_name)

        ndim = len(self.data_container.shape)
        if ndim == 2:
            # save single slice
            logger.info('2D dataset')
            if self.counter_offset >= 0:
                fname = "{}_idx_{:04d}.tiff".format(os.path.join(self.dir_name, self.file_name), self.counter_offset)
            else:
                fname = "{}.tiff".format(os.path.join(self.dir_name, self.file_name))
            with open(fname, 'wb') as f:
                Image.fromarray(self.data_container.as_array()).save(f, 'tiff')
        elif ndim == 3:
            logger.info('3D dataset')
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
            logger.info('4D dataset')
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
                        Image.fromarray(self.data_container.as_array()[sliceno1][sliceno2]).save(f, 'tiff')
        else:
            raise ValueError('Cannot handle more than 4 dimensions')
    
    def _zero_padding(self, number):
        i = 0
        while 10**i < number:
            i+=1
        i+=1 
        zero_padding_string = '{:0'+str(i)+'d}'
        return zero_padding_string

class TiffReader(object):
    def __init__(self, path, dtype=np.float32):
        '''Basic TIFF reader which loops through all tiff files in a specific 
        folder and load them in alphabetic order
        
        Parameters
        ----------
            
        :param path: path to folder with tiff files, list of paths of tiffs, or single tiff file
        :type path: str, abspath to folder, list'''
        self.path_to_tiffs = path
        if isinstance(path, list):
            self._tiff_files = path[:]
        elif os.path.isfile(path):
            self._tiff_files = [path]
        elif os.path.isdir(path): 
            self._tiff_files = glob.glob(os.path.join(path,"*.tif"))
            if not self._tiff_files:
                    self._tiff_files = glob.glob(os.path.join(path,"*.tiff"))

        if not self._tiff_files:
            raise Exception("No tiff files were found in the directory \n{}".format(file_name))
        self.dtype = dtype
    
    def read(self):
        # read one and figure out how much memory we need to read this dataset
        num_tiff = len(self._tiff_files)
        with Image.open(self._tiff_files[0]) as im:
            data = np.asarray(im, dtype=self.dtype)
        # if only one tiff return immediately after cast
        if num_tiff ==  1:
            return data
        shape = data.shape
        # allocate the memory
        data = np.empty((num_tiff, *shape), dtype=self.dtype)
        # copy the content of each file in data
        for i, el in self._tiff_files:
            with Image.open(el) as im:
                data[i] = np.asarray(im, dtype=self.dtype)[:]
        return data




class TIFFStackReader(object):
    
    def __init__(self, 
                 **kwargs):
        ''' 
        Basic TIFF reader which loops through all tiff files in a specific 
        folder and load them in alphabetic order
        
        Parameters
        ----------
            
        :param file_name: path to folder with tiff files, list of paths of tiffs, or single tiff file
        :type file_name: str, abspath to folder, list
            
        :param roi: dictionary with roi to load 
                {'axis_0': (start, end, step), 
                 'axis_1': (start, end, step), 
                 'axis_2': (start, end, step)}
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
        :type roi: dictionary, default None
            
        :param transpose: transpose loaded images
        :type transpose: bool, default False
            
        :param mode: str, 'bin' (default) or 'slice'. In bin mode, 'step' number
                of pixels is binned together, values of resulting binned
                pixels are calculated as average. 
                In 'slice' mode 'step' defines standard numpy slicing.
                Note: in general output array size in bin mode != output array size
                in slice mode
        :type mode: str, default 'bin'
        
        Returns
        -------
            
            numpy array with stack of images
            
        '''
        
        self.file_name = kwargs.get('file_name', None)
        self.roi = kwargs.get('roi', {'axis_0': -1, 'axis_1': -1, 'axis_2': -1})
        self.transpose = kwargs.get('transpose', False)
        self.mode = kwargs.get('mode', 'bin')
        
        if self.file_name is not None:
            self.set_up(file_name = self.file_name,
                        roi = self.roi,
                        transpose = self.transpose,
                        mode = self.mode)
            
    def set_up(self, 
               file_name = None,
               roi = {'axis_0': -1, 'axis_1': -1, 'axis_2': -1},
               transpose = False,
               mode = 'bin'):
        
        self.roi = roi
        self.transpose = transpose
        self.mode = mode
        
        if file_name == None:
            raise ValueError('file_name to tiff files is required. Can be a tiff, a list of tiffs or a directory containing tiffs')
            
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
        elif os.path.isfile(file_name):
            self._tiff_files = [file_name]
        elif os.path.isdir(file_name): 
            self._tiff_files = glob.glob(os.path.join(file_name,"*.tif"))
            
            if not self._tiff_files:
                self._tiff_files = glob.glob(os.path.join(file_name,"*.tiff"))

            if not self._tiff_files:
                raise Exception("No tiff files were found in the directory \n{}".format(file_name))

        else:
            raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))

        
        for fn in self._tiff_files:
            if '.tif' in fn:
                if not(os.path.exists(fn)):
                    raise Exception('File \n {}\n does not exist.'.format(fn))
            else:
                raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))

        
        self._tiff_files.sort(key=self.__natural_keys)
               
                
    def read(self):
        
        '''
        Reads images and return numpy array
        '''
        # load first image to find out dimensions
        filename = os.path.abspath(self._tiff_files[0])
        
        tmp = np.asarray(Image.open(filename), dtype = np.float32)
        
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
                im = np.zeros((num_to_read, n_rows, n_cols), dtype=np.float32)
            else:
                im = np.zeros((num_to_read, n_cols, n_rows), dtype=np.float32)
            
            for i in range(0,num_to_read):

                raw = np.zeros((array_shape_0[1], array_shape_0[2]), dtype=np.float32)
                for j in range(roi_par[0][2]):
                
                    index = int(roi_par[0][0] + i * roi_par[0][2] + j)
                    filename = os.path.abspath(self._tiff_files[index])

                    try:
                        raw += np.asarray(Image.open(filename), dtype = np.float32)
                    except:
                        print('Error reading\n {}\n file.'.format(filename))
                        raise
                        
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
            n_rows = np.int(np.ceil((roi_par[1][1] - roi_par[1][0]) / roi_par[1][2]))
            n_cols = np.int(np.ceil((roi_par[2][1] - roi_par[2][0]) / roi_par[2][2]))
            num_to_read = np.int(np.ceil((roi_par[0][1] - roi_par[0][0]) / roi_par[0][2]))
            
            if not self.transpose:
                im = np.zeros((num_to_read, n_rows, n_cols), dtype=np.float32)
            else:
                im = np.zeros((num_to_read, n_cols, n_rows), dtype=np.float32)
                        
            for i in range(roi_par[0][0], roi_par[0][1], roi_par[0][2]):
                
                filename = os.path.abspath(self._tiff_files[i])
                #try:
                raw = np.asarray(Image.open(filename), dtype = np.float32)
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
            gsize = functools.reduce(lambda x,y: x*y, geometry.shape, 1)
            dsize = functools.reduce(lambda x,y: x*y, data.shape, 1)
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
    


class EdoAndGemmaTIFFStackReader(TIFFStackReader):
    def read(self):
        reader = TiffReader(path = self._tiff_files)
        return reader.read()

class BinTIFFStackReader(TIFFStackReader):

    def set_Binner(self, binner):
        self.binner = binner
    def set_geometry(self, geometry):
        pass
    def read(self):
        pass