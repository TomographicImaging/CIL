# -*- coding: utf-8 -*-
from ccpi.framework import ImageData, ImageGeometry
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
    RESOLUTION_CHART = 'resolution_chart.tiff'
    SIMPLE_PHANTOM_2D = 'simple_jakobs_phantom'
    SHAPES =  'shapes.png'
    
    def __init__(self, **kwargs):
        self.data_dir = kwargs.get('data_dir', data_dir)
        
    def load(self, which, size=(512,512), scale=(0,1), **kwargs):
        if which not in [TestData.BOAT, TestData.CAMERA, 
                         TestData.PEPPERS, TestData.RESOLUTION_CHART,
                         TestData.SIMPLE_PHANTOM_2D, TestData.SHAPES]:
            raise ValueError('Unknown TestData {}.'.format(which))
        if which == TestData.SIMPLE_PHANTOM_2D:
            N = size[0]
            M = size[1]
            sdata = numpy.zeros((N,M))
            sdata[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
            sdata[round(M/8):round(7*M/8),round(3*M/8):round(5*M/8)] = 1
            ig = ImageGeometry(voxel_num_x = N, voxel_num_y = M, dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y])
            data = ig.allocate()
            data.fill(sdata)
            
        elif which == TestData.SHAPES:
            
            tmp = numpy.array(Image.open(os.path.join(self.data_dir, which)).convert('L'))
            N = 200
            M = 300   
            ig = ImageGeometry(voxel_num_x = N, voxel_num_y = M, dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y])
            data = ig.allocate()
            data.fill(tmp/numpy.max(tmp))
            
        else:
            tmp = Image.open(os.path.join(self.data_dir, which))
            print (tmp)
            bands = tmp.getbands()
            if len(bands) > 1:
                ig = ImageGeometry(voxel_num_x=size[0], voxel_num_y=size[1], channels=len(bands), 
                dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL])
                data = ig.allocate()
            else:
                ig = ImageGeometry(voxel_num_x = size[0], voxel_num_y = size[1], dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y])
                data = ig.allocate()
            data.fill(numpy.array(tmp.resize((size[1],size[0]))))
            if scale is not None:
                dmax = data.as_array().max()
                dmin = data.as_array().min()
                # scale 0,1
                data = (data -dmin) / (dmax - dmin)
                if scale != (0,1):
                    #data = (data-dmin)/(dmax-dmin) * (scale[1]-scale[0]) +scale[0])
                    data *= (scale[1]-scale[0])
                    data += scale[0]
        print ("data.geometry", data.geometry)
        return data

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
    
    def shapes(**kwargs):
    
        tmp = Image.open(os.path.join(data_dir, 'shapes.png')).convert('LA')
        
        size = kwargs.get('size',(300, 200))
        
        data = numpy.array(tmp.resize(size))
            
        data = data/data.max()
        
        return ImageData(data)     
    
