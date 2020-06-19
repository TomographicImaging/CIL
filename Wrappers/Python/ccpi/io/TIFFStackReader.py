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
        self.transpose = kwargs.get('transpose', False)
        
        if self.path is not None:
            self.set_up(path = self.path,
                        roi = self.roi,
                        transpose = self.normalize)
            
    def set_up(self, 
               path = None,
               roi = {'axis_0': -1, 'axis_1': -1},
               transpose = False):
        
        self.path = path
        self.roi = roi
        self.transpose = transpose
        
        if self.path == None:
            raise Exception('Path to tiff files is required.')
            
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
        
        self._tiff_files.sort()
               
                
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
                
        # calculate number of pixels
        n_rows = (roi_par[1][1] - roi_par[1][0]) // roi_par[1][2]
        n_cols = (roi_par[2][1] - roi_par[2][0]) // roi_par[2][2]
        num_to_read = (roi_par[0][1] - roi_par[0][0]) // roi_par[0][2]
        
        if not self.transpose:
            im = numpy.zeros((num_to_read, n_rows, n_cols), dtype=numpy.float32)
        else:
            im = numpy.zeros((num_to_read, n_cols, n_rows), dtype=numpy.float32)
        if roi_par[0][2] > 1:
            roi_par[0][1] -= roi_par[0][2]
        for i in range(roi_par[0][0], roi_par[0][1], roi_par[0][2]):
            raw = numpy.zeros((array_shape_0[1], array_shape_0[2]), dtype=numpy.float32)
            for j in range(roi_par[0][2]):
            
                filename = os.path.join(self.path, self._tiff_files[i + j])
            
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
                im[i // roi_par[0][2], :, :] = numpy.transpose(tmp)
            else:
                im[i // roi_par[0][2], :, :] = tmp
        
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