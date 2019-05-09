# -*- coding: utf-8 -*-
from ccpi.framework import ImageData
import numpy
from PIL import Image
import os
import os.path 

data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../data/')
)

class TestData(object):
    BOAT = 'boat.tiff'
    CAMERA = 'camera.png'
    PEPPERS = 'peppers.tiff'
    
    def __init__(self, **kwargs):
        self.data_dir = kwargs.get('data_dir', data_dir)
        
    def load(self, which, size=(512,512), scale=(0,1), **kwargs):
        if which not in [TestData.BOAT, TestData.CAMERA, TestData.PEPPERS]:
            raise ValueError('Unknown TestData {}.'.format(which))
        tmp = Image.open(os.path.join(self.data_dir, which))
        
        data = numpy.array(tmp.resize(size))
        
        if scale is not None:
            dmax = data.max()
            dmin = data.min()
        
            data = data -dmin / (dmax - dmin)
            
            if scale != (0,1):
                #data = (data-dmin)/(dmax-dmin) * (scale[1]-scale[0]) +scale[0])
                data *= (scale[1]-scale[0])
                data += scale[0]
        
        return ImageData(data) 
        

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
    
