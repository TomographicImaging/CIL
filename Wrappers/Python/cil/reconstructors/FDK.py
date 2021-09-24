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
#   limitations under the License.
from cil.framework import AcquisitionGeometry
from cil.reconstructors import Reconstructor
from scipy.fft import fftfreq
from cil.plugins.tigre import ProjectionOperator

import numpy as np
import ctypes, platform
from ctypes import util

if platform.system() == 'Linux':
    dll = 'libcilacc.so'
elif platform.system() == 'Windows':
    dll_file = 'cilacc.dll'
    dll = util.find_library(dll_file)
elif platform.system() == 'Darwin':
    dll = 'libcilacc.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

cilacc = ctypes.cdll.LoadLibrary(dll)

c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)

cilacc.filter_projections.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the first array 
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the filter 
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the filter 
                                  ctypes.c_int16,
                                  ctypes.c_long,
                                  ctypes.c_long,
                                  ctypes.c_long]    

cilacc.filter_projections_reorder.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the first array 
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the filter 
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the filter 
                                  ctypes.c_int16,
                                  ctypes.c_long,
                                  ctypes.c_long,
                                  ctypes.c_long]   

class FDK(Reconstructor):

    @property
    def filter(self):
        return self.__filter

    @filter.setter
    def filter(self, val):
        self.set_filter(val)

    @property
    def filter_inplace(self):
        return self.__filter_inplace

    @filter_inplace.setter
    def filter_inplace(self, val):
        if type(val) is bool:
            self.__filter_inplace= val

    @property
    def fft_order(self):
        return self.__fft_order

    @fft_order.setter
    def fft_order(self, val):
        self.set_fft_order(val)

    def __init__ (self,input):

        #additional check
        if 'channel' in input.dimension_labels:
            raise ValueError("Input data cannot be multi-channel")

        #call parent initialiser
        super(FDK, self).__init__(input)

        #define defaults
        self.__filter = 'ram-lak' 
        self.set_fft_order(0)
        self.__filter_inplace = False


    def set_fft_order(self, order):
        """
        fft width 2^order, can only specify values over the minimum
        """
        min_order = 0
        while 2**min_order < self.input.geometry.pixel_num_h * 2:
            min_order+=1

        min_order = max(8, min_order)
        fft_order = max(int(order), int(min_order))

        if type(self.filter)==np.ndarray and fft_order != self.fft_order:
            print("Filter length changed - resetting filter array to ram-lak")
            self.__filter=='ram-lak'
        
        self.__fft_order =fft_order

    def set_filter(self, filter):
        """
        set fft filter to 'ram-lak' or custum numpy array
        """
        if filter in ['ram-lak']:
            self.__filter == filter
        elif type(self.filter)==np.ndarray:
            try:
                filter_array = np.asarray(filter,dtype=np.float32).reshape(2**self.order) 
                self.__filter = 'custom'
                self.__filter_array = filter_array
            except ValueError:
                raise ValueError("Custom filter not compatible with input")
        else:
            raise ValueError("Filter not recognised")
            
    def get_filter_array(self):

        if self.filter == 'ram-lak':
            filter_length = 2**self.fft_order
            freq = fftfreq(filter_length)
            filter = np.asarray( [ np.abs(2*el) for el in freq ] ,dtype=np.float32)
        elif self.filter == 'custom':
            filter = self.__filter_array
        return filter

    def __calculate_weights_fdk(self):
        ag = self.input.geometry
        xv = np.arange(-(ag.pixel_num_h -1)/2,(ag.pixel_num_h -1)/2 + 1,dtype=np.float32) * ag.pixel_size_h
        yv = np.arange(-(ag.pixel_num_v -1)/2,(ag.pixel_num_v -1)/2 + 1,dtype=np.float32) * ag.pixel_size_v
        (yy, xx) = np.meshgrid(xv, yv)

        principal_ray_length = ag.dist_source_center + ag.dist_center_detector
        scaling =  ag.magnification * (2 * np.pi/ len(ag.angles)) / ( 4 * ag.pixel_size_h ) 
        weights = scaling * principal_ray_length / np.sqrt((principal_ray_length ** 2 + xx ** 2 + yy ** 2))
        return weights

    def __calculate_weights_fbp(self):
        ag = self.input.geometry
        weights = np.ones(ag.pixel_num_h,ag.pixel_num_v)
        return weights

    def __pre_filtering(self,acquistion_data):

        nda = acquistion_data.as_array()

        filter=self.get_filter_array()

        if AcquisitionGeometry.CONE:
            weights = self.__calculate_weights_fdk()
        else:
            weights = self.__calculate_weights_fbp()

        #call ext function
        data_ptr = nda.ctypes.data_as(c_float_p)
        filter_ptr = filter.ctypes.data_as(c_float_p)
        weights_ptr = weights.ctypes.data_as(c_float_p)

        ag = self.input.geometry
        if ag.dimension_labels == ('angle','vertical','horizontal'):   
            cilacc.filter_projections(data_ptr, filter_ptr, weights_ptr, self.fft_order, *self.input.shape)
        elif ag.dimension_labels == ('angle','horizontal'): 
            cilacc.filter_projections_reorder(data_ptr, filter_ptr, weights_ptr, self.fft_order, 1, *self.input.shape) 
        else:
            raise ValueError ("Gemma says no")

        acquistion_data.fill(nda)

    def run(self, out=None):

        if self.filter_inplace is False:
            proj_filtered = self.input.copy()
        else:
            proj_filtered = self.input

        self.__pre_filtering(proj_filtered)
            
        operator = ProjectionOperator(self.image_geometry,self.input.geometry,adjoint_method='FDK')
        
        if out == None:
            return operator.adjoint(proj_filtered)
        else:
            operator.adjoint(proj_filtered, out = out)

