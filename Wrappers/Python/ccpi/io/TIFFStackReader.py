# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2019 UKRI-STFC
#   Copyright 2019 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import os
import re


pilAvailable = True
try:    
    from PIL import Image
except:
    pilAvailable = False


class TIFFStackReader(object):
    
    def __init__(self, 
                 **kwargs):
        ''' 
        Basic TIFF redaer which loops through all riff files in a specific 
        folder and load them in alphabetic order
        
        Parameters
        ----------
            
        path: str containing path to folder with tiff files
            
        roi: dictionary with roi to load 
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
            
        transpose: bool, transpose loaded images, default False
            
        mode: str, 'bin' (default) or 'slice'. In bin mode, 'step' number
                of pixels is binned together, values of resulting binned
                pixels are calculated as average. 
                In 'slice' mode 'step' defines standard numpy slicing.
                Note: in general 
                output array size in bin mode != output array size in slice mode
        
        Returns
        -------
            
            numpy array with stack of images
            
        '''
        
        self.path = kwargs.get('path', None)
        self.roi = kwargs.get('roi', {'axis_0': -1, 'axis_1': -1, 'axis_2': -1})
        self.transpose = kwargs.get('transpose', False)
        self.mode = kwargs.get('mode', 'bin')
        
        if self.path is not None:
            self.set_up(path = self.path,
                        roi = self.roi,
                        transpose = self.transpose,
                        mode = self.mode)
            
    def set_up(self, 
               path = None,
               roi = {'axis_0': -1, 'axis_1': -1, 'axis_2': -1},
               transpose = False,
               mode = 'bin'):
        
        self.path = path
        self.roi = roi
        self.transpose = transpose
        self.mode = mode
        
        if self.path == None:
            raise ValueError('Path to tiff files is required.')
            
        # check if path exists
        if not(os.path.exists(self.path)):
            raise Exception('Path \n {}\n does not exist.'.format(self.path))  
                
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
        
        self._tiff_files = [f for f in os.listdir(self.path) if (os.path.isfile(os.path.join(self.path, f)) and '.tif' in f.lower())]

        if not self._tiff_files:
            raise Exception("No tiff files were found in the directory \n{}".format(self.path))
        
        self._tiff_files.sort(key=self.__natural_keys)
               
                
    def read(self):
        
        '''
        Reads images and return numpy array
        '''
        # load first image to find out dimensions
        filename = os.path.join(self.path, self._tiff_files[0])
        
        try:
            tmp = numpy.asarray(Image.open(filename), dtype = numpy.float32)
        except:
            print('Error reading\n {}\n file.'.format(filename))
            raise
        
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
                im = numpy.zeros((num_to_read, n_rows, n_cols), dtype=numpy.float32)
            else:
                im = numpy.zeros((num_to_read, n_cols, n_rows), dtype=numpy.float32)
            
            for i in range(0,num_to_read):

                raw = numpy.zeros((array_shape_0[1], array_shape_0[2]), dtype=numpy.float32)
                for j in range(roi_par[0][2]):
                
                    filename = os.path.join(self.path, self._tiff_files[roi_par[0][0] + i * roi_par[0][2] + j])

                    try:
                        raw += numpy.asarray(Image.open(filename), dtype = numpy.float32)
                    except:
                        print('Error reading\n {}\n file.'.format(filename))
                        raise
                        
                shape = (n_rows, roi_par[1][2], 
                         n_cols, roi_par[2][2])
                tmp = raw[roi_par[1][0]:(roi_par[1][0] + (((roi_par[1][1] - roi_par[1][0]) // roi_par[1][2]) * roi_par[1][2])), \
                          roi_par[2][0]:(roi_par[2][0] + (((roi_par[2][1] - roi_par[2][0]) // roi_par[2][2]) * roi_par[2][2]))].reshape(shape).mean(-1).mean(1)
                
                if self.transpose:
                    im[i, :, :] = numpy.transpose(tmp)
                else:
                    im[i, :, :] = tmp
                    
        else: # slice mode
            # calculate number of pixels
            n_rows = numpy.int(numpy.ceil((roi_par[1][1] - roi_par[1][0]) / roi_par[1][2]))
            n_cols = numpy.int(numpy.ceil((roi_par[2][1] - roi_par[2][0]) / roi_par[2][2]))
            num_to_read = numpy.int(numpy.ceil((roi_par[0][1] - roi_par[0][0]) / roi_par[0][2]))
            
            if not self.transpose:
                im = numpy.zeros((num_to_read, n_rows, n_cols), dtype=numpy.float32)
            else:
                im = numpy.zeros((num_to_read, n_cols, n_rows), dtype=numpy.float32)
                        
            for i in range(roi_par[0][0], roi_par[0][1], roi_par[0][2]):
                
                filename = os.path.join(self.path, self._tiff_files[i])
                try:
                    raw = numpy.asarray(Image.open(filename), dtype = numpy.float32)
                except:
                    print('Error reading\n {}\n file.'.format(filename))
                    raise
                
                tmp = raw[(slice(roi_par[1][0], roi_par[1][1], roi_par[1][2]), 
                           slice(roi_par[2][0], roi_par[2][1], roi_par[2][2]))]
                
                if self.transpose:
                    im[(i - roi_par[0][0]) // roi_par[0][2], :, :] = numpy.transpose(tmp)
                else:
                    im[(i - roi_par[0][0]) // roi_par[0][2], :, :] = tmp
        
        return numpy.squeeze(im)
    
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
    
    


'''
import matplotlib.pyplot as plt
from ccpi.io import TIFFStackReader

path = '/media/newhd/shared/Data/SophiaBeads/SophiaBeads_256_averaged/'

reader = TIFFStackReader()
reader.set_up(path = path,
              n_images = 100,
              binning = {'axis_0': 5, 'axis_1': 6},
              roi = {'axis_0': (100,900), 'axis_1': (200,700)},
              skip = 100)

data = reader.load_images()

plt.imshow(data[1, :, :])
plt.show()
'''