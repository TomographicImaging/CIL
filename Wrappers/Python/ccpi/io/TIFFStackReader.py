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


pilAvailable = True
try:    
    from PIL import Image
except:
    pilAvailable = False


class TIFFStackReader(object):
    
    def __init__(self, 
                 **kwargs):
        '''
        Constructor
        
        Input:
            
            path            path to tiff files
            
            n_images        number of images to read, -1 (default) read all
            
            skip            number of images to skip, default 0
            
            roi             region-of-interest to load. 
                            If roi = {'axis_0': -1, 'axis_1': -1} (default), 
                            full images will be loaded. Otherwise roi is 
                            given by {'axis_0': (row0, row1), 'axis_1': (column0, column1)}
                            where row0, column0 are coordinates of top left corner and 
                            row1, column1 are coordinates of bottom right corner.
                            
            binning         number of pixels to bin (combine) along corresponsing axis. 
                            If binning = {'axis_0': 1, 'axis_1': 1} (default),
                            images in original resolution are loaded. 
            
            transpose       Transpose loaded images, default False
            
            Notes:
            roi and benning are specified for axes before transpose.
            
                    
        '''
        
        self.path = kwargs.get('path', None)
        self.n_images = kwargs.get('n_images', -1)
        self.skip = kwargs.get('skip', 0)
        self.roi = kwargs.get('roi', {'axis_0': -1, 'axis_1': -1})
        self.binning = kwargs.get('binning', {'axis_0': 1, 'axis_1': 1})
        self.transpose = kwargs.get('transpose', False)
        
        if self.path is not None:
            self.set_up(path = self.path,
                        n_images = self.n_images,
                        skip = self.skip,
                        roi = self.roi,
                        binning = self.binning,
                        transpose = self.normalize)
            
    def set_up(self, 
               path = None, 
               n_images = -1,
               skip = 0,
               roi = {'axis_0': -1, 'axis_1': -1}, 
               binning = {'axis_0': 1, 'axis_1': 1},
               transpose = False):
        
        self.path = path
        self.n_images = n_images
        self.skip = skip
        self.roi = roi
        self.binning = binning
        self.transpose = transpose
        
        if self.path == None:
            raise Exception('Path to tiff files is required.')
        
        if self.n_images != -1:
             if not (isinstance(self.n_images, int)):
                 raise Exception("Number of images to read must be integer.")
                 
        if self.skip != -1:
             if not (isinstance(self.skip, int)):
                 raise Exception("Number of images to skip must be integer.")
            
        # check if path exists
        if not(os.path.exists(self.path)):
            raise Exception('Path \n {}\n does not exist.'.format(self.path))  
                
        # check that PIL library is installed
        if (pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files.")
        
        # check labels
        for key in self.binning.keys():
            if key not in ['axis_0', 'axis_1']:
                raise Exception("Wrong label. axis_0 and axis_1 are expected")
        
        for key in self.roi.keys():
            if key not in ['axis_0', 'axis_1']:
                raise Exception("Wrong label. axis_0 and axis_1 are expected")
        
        self._binning = self.binning.copy()
        self._roi = self.roi.copy()
        
        if 'axis_1' not in self._binning.keys():
            self._binning['axis_1'] = 1
        
        if 'axis_0' not in self._binning.keys():
            self._binning['axis_0'] = 1
        
        if 'axis_1' not in self._roi.keys():
            self._roi['axis_1'] = -1
        
        if 'axis_0' not in self._roi.keys():
            self._roi['axis_0'] = -1
                
        # check if inputs for roi and binning are integer
        if not (isinstance(self._binning['axis_0'], int) and \
                isinstance(self._binning['axis_1'], int)):
            raise Exception("Integers are expected for binning")
        
        if self._roi['axis_0'] != -1:
            if not (isinstance(self._roi['axis_0'][0], int) and \
                    isinstance(self._roi['axis_0'][1], int)):
                raise Exception("Integers are expected for roi")
        
        if self._roi['axis_1'] != -1:
            if not (isinstance(self._roi['axis_1'][0], int) and \
                    isinstance(self._roi['axis_1'][1], int)):
                raise Exception("Integers are expected for roi")
        
        self._tiff_files = [f for f in os.listdir(self.path) if (os.path.isfile(os.path.join(self.path, f)) and '.tif' in f.lower())]

        if not self._tiff_files:
            raise Exception("No tiff files were found in the directory \n{}".format(self.path))
        
        self._tiff_files.sort()
        
        if self.skip > len(self._tiff_files):
            raise Exception("Number of files to skip is larger than number of tiff files in {}".format(self.path))        
        
                
    def load_images(self):
        
        '''
        Load images and return numpy array
        '''
        # load first image to find out dimensions
        filename = os.path.join(self.path, self._tiff_files[0])
        
        try:
            tmp = numpy.asarray(Image.open(filename), dtype = numpy.float32)
        except:
            print('Error reading\n {}\n file.'.format(filename))
            raise
        
        n_rows_raw, n_cols_raw = tmp.shape

        roi_par = []
        if self._roi['axis_0'] == -1:
            roi_par.append((0, n_rows_raw))
        else:
            roi_par.append(self._roi['axis_0'])
        if self._roi['axis_1'] == -1:
            roi_par.append((0, n_cols_raw))
        else:
            roi_par.append(self._roi['axis_1'])
                
        # calculate number of pixels
        if (self._binning['axis_0'] == 1 and self._binning['axis_1'] == 1):
            n_rows = roi_par[0][1] - roi_par[0][0]
            n_cols = roi_par[1][1] - roi_par[1][0]
        else:
            n_rows = (roi_par[0][1] - roi_par[0][0]) // self._binning['axis_0']
            n_cols = (roi_par[1][1] - roi_par[1][0]) // self._binning['axis_1']
        
        if (self.n_images == -1 or self.n_images > (len(self._tiff_files) - self.skip)):
            num_to_read = len(self._tiff_files) - self.skip
        else:
            num_to_read = self.n_images
        
        if not self.transpose:
            im = numpy.zeros((num_to_read, n_rows, n_cols), dtype=numpy.float32)
        else:
            im = numpy.zeros((num_to_read, n_cols, n_rows), dtype=numpy.float32)
            
        for i in range(num_to_read):
            
            filename = filename = os.path.join(self.path, self._tiff_files[i + self.skip])
            
            try:
                raw = numpy.asarray(Image.open(filename), dtype = numpy.float32)
            except:
                print('Error reading\n {}\n file.'.format(filename))
                raise
                
            if (self._binning['axis_0'] == 1 and self._binning['axis_1'] == 1):
                tmp = raw[roi_par[0][0]:roi_par[0][1], roi_par[1][0]:roi_par[1][1]]
            else:
                shape = (n_rows, self._binning['axis_0'], 
                         n_cols, self._binning['axis_1'])
                tmp = raw[roi_par[0][0]:(roi_par[0][0] + (((roi_par[0][1] - roi_par[0][0]) // self._binning['axis_0']) * self._binning['axis_0'])), \
                          roi_par[1][0]:(roi_par[1][0] + (((roi_par[1][1] - roi_par[1][0]) // self._binning['axis_1']) * self._binning['axis_1']))].reshape(shape).mean(-1).mean(1)
            
            if self.transpose:
                im[i, :, :] = numpy.transpose(tmp)
            else:
                im[i, :, :] = tmp
        
        return im


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