# -*- coding: utf-8 -*-
#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

from cil.framework import ImageData, image_labels, data_order
import tomophantom
from tomophantom import TomoP2D, TomoP3D
import os
import numpy as np

import ctypes, platform
from ctypes import util
# check for the extension
if platform.system() == 'Linux':
    dll = 'libctomophantom.so'
elif platform.system() == 'Windows':
    dll_file = 'ctomophantom.dll'
    dll = util.find_library(dll_file)
elif platform.system() == 'Darwin':
    dll = 'libctomophantom.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

libtomophantom = ctypes.cdll.LoadLibrary(dll)



path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

def is_model_temporal(num_model, num_dims=2):
    '''Returns whether a model in the TomoPhantom library is temporal
    
    This will go to check the installed library files from TomoPhantom
    https://github.com/dkazanc/TomoPhantom/tree/master/PhantomLibrary/models

    :param num_model: model number
    :type num_model: int
    :param num_dims: dimensionality of the phantom, 2D or 3D
    :type num_dims: int, default 2
    '''
    return get_model_num_channels(num_model, num_dims) > 1

def get_model_num_channels(num_model, num_dims=2):
    '''Returns number of temporal steps (channels) the model has
    
    This will go to check the installed library files from TomoPhantom
    https://github.com/dkazanc/TomoPhantom/tree/master/PhantomLibrary/models

    https://github.com/dkazanc/TomoPhantom/blob/v1.4.9/Core/utils.c#L27
    https://github.com/dkazanc/TomoPhantom/blob/v1.4.9/Core/utils.c#L269

    :param num_model: model number
    :type num_model: int
    :param num_dims: dimensionality of the phantom, 2D or 3D
    :type num_dims: int, default 2
    '''
    
    return check_model_params(num_model, num_dims=num_dims)[3]

def check_model_params(num_model, num_dims=2):
    '''Returns params_switch array from the C TomoPhantom library in function checkParams2D or checkParams3D
    
    This will go to check the installed library files from TomoPhantom
    https://github.com/dkazanc/TomoPhantom/tree/master/PhantomLibrary/models

    https://github.com/dkazanc/TomoPhantom/blob/v1.4.9/Core/utils.c#L27
    https://github.com/dkazanc/TomoPhantom/blob/v1.4.9/Core/utils.c#L269

    :param num_model: model number
    :type num_model: int
    :param num_dims: dimensionality of the phantom, 2D or 3D
    :type num_dims: int, default 2
    '''
    if num_dims == 2:
        libtomophantom.checkParams2D.argtypes = [ctypes.POINTER(ctypes.c_int),  # pointer to the params array 
                                  ctypes.c_int,                                   # model number selector (int)
                                  ctypes.c_char_p]                  # string to the library file
        params = np.zeros([10], dtype=np.int32)
        params_p = params.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        lib2d_p = str(path_library2D).encode('utf-8')
        libtomophantom.checkParams2D(params_p, num_model, lib2d_p)

        return params
        
    elif num_dims == 3:
        libtomophantom.checkParams3D.argtypes = [ctypes.POINTER(ctypes.c_int),  # pointer to the params array 
                                  ctypes.c_int,                                   # model number selector (int)
                                  ctypes.c_char_p]                  # string to the library file
        params = np.zeros([11], dtype=np.int32)
        params_p = params.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        lib2d_p = str(path_library3D).encode('utf-8')
        libtomophantom.checkParams3D(params_p, num_model, lib2d_p)
        
        return params

    else:
        raise ValueError('Unsupported dimensionality. Expected 2 or 3, got {}'.format(dims))

def get_ImageData(num_model, geometry):
    '''Returns an ImageData relative to geometry with the model num_model from tomophantom
    
    :param num_model: model number
    :type num_model: int
    :param geometry: geometrical info that describes the phantom
    :type geometry: ImageGeometry
    Example usage:
    
    .. code-block:: python
      
      ndim = 2
      N=128
      angles = np.linspace(0, 360, 50, True, dtype=np.float32)
      offset = 0.4
      channels = 3
        
      if ndim == 2:
          ag = AcquisitionGeometry.create_Cone2D((offset,-100), (offset,100))
          ag.set_panel(N)
            
      else:
          ag = AcquisitionGeometry.create_Cone3D((offset,-100, 0), (offset,100,0))
          ag.set_panel((N,N-2))
        
      ag.set_channels(channels)
      ag.set_angles(angles, angle_unit=acquisition_labels["DEGREE"])
        
        
      ig = ag.get_ImageGeometry()
      num_model = 1
      phantom = TomoPhantom.get_ImageData(num_model=num_model, geometry=ig)

    
    '''
    ig = geometry.copy()
    ig.set_labels(data_order["TOMOPHANTOM_IG_LABELS"])
    num_dims = len(ig.dimension_labels)
    
    if image_labels["CHANNEL"] in ig.dimension_labels:
        if not is_model_temporal(num_model):
            raise ValueError('Selected model {} is not a temporal model, please change your selection'.format(num_model))
        if num_dims == 4:
            # 3D+time for tomophantom
            # output dimensions channel and then spatial, 
            # e.g. [ 'channel', 'vertical', 'horizontal_y', 'horizontal_x' ]
            num_model = num_model
            shape = tuple(ig.shape[1:])
            phantom_arr = TomoP3D.ModelTemporal(num_model, shape, path_library3D)
        elif num_dims == 3:
            # 2D+time for tomophantom
            # output dimensions channel and then spatial, 
            # e.g. [ 'channel', 'horizontal_y', 'horizontal_x' ]
            N = ig.shape[1]
            num_model = num_model
            phantom_arr = TomoP2D.ModelTemporal(num_model, ig.shape[1], path_library2D)
        else:
            raise ValueError('Wrong ImageGeometry')
        if ig.channels != phantom_arr.shape[0]:
            raise ValueError('The required model {} has {} channels. The ImageGeometry you passed has {}. Please update your ImageGeometry.'\
                .format(num_model, ig.channels, phantom_arr.shape[0]))
    else:
        if num_dims == 3:
            # 3D
            num_model = num_model
            phantom_arr = TomoP3D.Model(num_model, ig.shape, path_library3D)
        elif num_dims == 2:
            # 2D
            if ig.shape[0] != ig.shape[1]:
                raise ValueError('Can only handle square ImageData, got shape'.format(ig.shape))
            N = ig.shape[0]
            num_model = num_model
            phantom_arr = TomoP2D.Model(num_model, N, path_library2D)
        else:
            raise ValueError('Wrong ImageGeometry')

    
    im_data = ImageData(phantom_arr, geometry=ig, suppress_warning=True)
    im_data.reorder(list(geometry.dimension_labels))
    return im_data
