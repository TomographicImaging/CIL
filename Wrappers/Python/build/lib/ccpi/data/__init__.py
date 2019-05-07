# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from ccpi.framework import ImageData
import numpy
from PIL import Image
import os
import os.path 

data_dir = os.path.abspath(os.path.dirname(__file__))

          
def camera(**kwargs):

    tmp = Image.open(os.path.join(data_dir, 'camera.png'))
    
    size = kwargs.get('size',(512, 512))
    
    data = numpy.array(tmp.resize(size))
        
    data = data/data.max()
    
    return ImageData(data) 


def boat(**kwargs):

    tmp = Image.open(os.path.join(data_dir, 'boat.tiff'))
    
    size = kwargs.get('size',(512, 512))
    
    data = numpy.array(tmp.resize(size))
        
    data = data/data.max()
    
    return ImageData(data)  


def peppers(**kwargs):

    tmp = Image.open(os.path.join(data_dir, 'peppers.tiff'))
    
    size = kwargs.get('size',(512, 512))
    
    data = numpy.array(tmp.resize(size))
        
    data = data/data.max()
    
    return ImageData(data)    
    
