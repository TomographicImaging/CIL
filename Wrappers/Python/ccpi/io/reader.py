# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev, Edoardo Pasca and Srikanth Nagella

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

'''
This is a reader module with classes for loading 3D datasets. 

@author: Mr. Srikanth Nagella
'''
from ccpi.framework import AcquisitionGeometry
from ccpi.framework import AcquisitionData
import numpy as np
import os

h5pyAvailable = True
try:
    from h5py import File as NexusFile
except:
    h5pyAvailable = False

pilAvailable = True
try:    
    from PIL import Image
except:
    pilAvailable = False
     
class NexusReader(object):
    '''
    Reader class for loading Nexus files. 
    '''

    def __init__(self, nexusFilename=None):
        '''
        This takes in input as filename and loads the data dataset.
        '''
        self.flat = None
        self.dark = None
        self.angles = None
        self.geometry = None
        self.filename = nexusFilename
        
    def load(self, dimensions=None, image_key_id=0):  
        '''
        This is generic loading function of flat field, dark field and projection data.
        '''      
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return        
        try:
            with NexusFile(self.filename,'r') as file:                
                image_keys = np.array(file['entry1/tomo_entry/instrument/detector/image_key'])                
                projections = None
                if dimensions == None:
                    projections = np.array(file['entry1/tomo_entry/data/data'])
                    result = projections[image_keys==image_key_id]
                    return result
                else:
                    #When dimensions are specified they need to be mapped to image_keys
                    index_array = np.where(image_keys==image_key_id)
                    projection_indexes = index_array[0][dimensions[0]]
                    new_dimensions = list(dimensions)
                    new_dimensions[0]= projection_indexes
                    new_dimensions = tuple(new_dimensions)              
                    result = np.array(file['entry1/tomo_entry/data/data'][new_dimensions])
                    return result
        except:
            print("Error reading nexus file")
            raise
        
    def load_projection(self, dimensions=None):
        '''
        Loads the projection data from the nexus file.
        returns: numpy array with projection data
        '''
        return self.load(dimensions, 0)
    
    def load_flat(self, dimensions=None):
        '''
        Loads the flat field data from the nexus file.
        returns: numpy array with flat field data
        '''        
        return self.load(dimensions, 1)
    
    def load_dark(self, dimensions=None):
        '''
        Loads the Dark field data from the nexus file.
        returns: numpy array with dark field data
        '''        
        return self.load(dimensions, 2)
    
    def get_projection_angles(self):
        '''
        This function returns the projection angles
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return        
        try:
            with NexusFile(self.filename,'r') as file:                
                angles = np.array(file['entry1/tomo_entry/data/rotation_angle'],np.float32)
                image_keys = np.array(file['entry1/tomo_entry/instrument/detector/image_key'])                
                return angles[image_keys==0]
        except:
            print("Error reading nexus file")
            raise        

    
    def get_sinogram_dimensions(self):
        '''
        Return the dimensions of the dataset
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:
            with NexusFile(self.filename,'r') as file:                
                projections = file['entry1/tomo_entry/data/data']
                image_keys = np.array(file['entry1/tomo_entry/instrument/detector/image_key'])
                dims = list(projections.shape)
                dims[0] = dims[1]
                dims[1] = np.sum(image_keys==0)
                return tuple(dims)
        except:
            print("Error reading nexus file")
            raise                
        
    def get_projection_dimensions(self):
        '''
        Return the dimensions of the dataset
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:
            with NexusFile(self.filename,'r') as file:                
                projections = file['entry1/tomo_entry/data/data']
                image_keys = np.array(file['entry1/tomo_entry/instrument/detector/image_key'])
                dims = list(projections.shape)
                dims[0] = np.sum(image_keys==0)
                return tuple(dims)
        except:
            print("Error reading nexus file")
            raise  
        
    def get_acquisition_data(self, dimensions=None):
        '''
        This method load the acquisition data and given dimension and returns an AcquisitionData Object
        '''
        data = self.load_projection(dimensions)
        dims = self.get_projection_dimensions()
        geometry = AcquisitionGeometry('parallel', '3D', 
                                       self.get_projection_angles(),
                                       pixel_num_h          = dims[2],
                                       pixel_size_h         = 1 ,
                                       pixel_num_v          = dims[1],
                                       pixel_size_v         = 1,
                                       dist_source_center   = None, 
                                       dist_center_detector = None, 
                                       channels             = 1)
        return AcquisitionData(data, geometry=geometry, 
                               dimension_labels=['angle','vertical','horizontal'])   
    
          
class XTEKReader(object):
    '''
    Reader class for loading XTEK files
    '''
    
    def __init__(self, xtekConfigFilename=None):
        '''
        This takes in the xtek config filename and loads the dataset and the
        required geometry parameters
        '''       
        self.projections = None
        self.geometry = {}
        self.filename = xtekConfigFilename
        self.load()
        
    def load(self):
        pixel_num_h = 0
        pixel_num_v = 0        
        xpixel_size = 0        
        ypixel_size = 0
        source_x = 0
        detector_x = 0
        with open(self.filename) as f:
            content = f.readlines()                    
        content = [x.strip() for x in content]
        for line in content:
            if line.startswith("SrcToObject"):
                source_x = float(line.split('=')[1])
            elif line.startswith("SrcToDetector"):
                detector_x = float(line.split('=')[1])
            elif line.startswith("DetectorPixelsY"):
                pixel_num_v = int(line.split('=')[1])
                #self.num_of_vertical_pixels = self.calc_v_alighment(self.num_of_vertical_pixels, self.pixels_per_voxel)
            elif line.startswith("DetectorPixelsX"):
                pixel_num_h = int(line.split('=')[1])
            elif line.startswith("DetectorPixelSizeX"):
                xpixel_size = float(line.split('=')[1])
            elif line.startswith("DetectorPixelSizeY"):
                ypixel_size = float(line.split('=')[1])   
            elif line.startswith("Projections"):
                self.num_projections = int(line.split('=')[1])
            elif line.startswith("InitialAngle"):
                self.initial_angle = float(line.split('=')[1])
            elif line.startswith("Name"):
                self.experiment_name = line.split('=')[1]
            elif line.startswith("Scattering"):
                self.scattering = float(line.split('=')[1])
            elif line.startswith("WhiteLevel"):
                self.white_level = float(line.split('=')[1])                
            elif line.startswith("MaskRadius"):
                self.mask_radius = float(line.split('=')[1])
                
        #Read Angles
        angles = self.read_angles()    
        self.geometry = AcquisitionGeometry('cone', '3D', angles, pixel_num_h, xpixel_size, pixel_num_v, ypixel_size, -1 * source_x, 
                 detector_x - source_x, 
                 )
        
    def read_angles(self):
        """
        Read the angles file .ang or _ctdata.txt file and returns the angles
        as an numpy array. 
        """ 
        input_path = os.path.dirname(self.filename)
        angles_ctdata_file = os.path.join(input_path, '_ctdata.txt')
        angles_named_file = os.path.join(input_path, self.experiment_name+'.ang')
        angles = np.zeros(self.num_projections,dtype='f')
        #look for _ctdata.txt
        if os.path.exists(angles_ctdata_file):
            #read txt file with angles
            with open(angles_ctdata_file) as f:
                content = f.readlines()
            #skip firt three lines
            #read the middle value of 3 values in each line as angles in degrees
            index = 0
            for line in content[3:]:
                self.angles[index]=float(line.split(' ')[1])
                index+=1
            angles = np.deg2rad(self.angles+self.initial_angle);
        elif os.path.exists(angles_named_file):
            #read the angles file which is text with first line as header
            with open(angles_named_file) as f:
                content = f.readlines()
            #skip first line
            index = 0
            for line in content[1:]:
                angles[index] = float(line.split(':')[1])
                index+=1
            angles = np.flipud(angles+self.initial_angle) #angles are in the reverse order
        else:
            raise RuntimeError("Can't find angles file")
        return angles  
    
    def load_projection(self, dimensions=None):
        '''
        This method reads the projection images from the directory and returns a numpy array
        '''  
        if not pilAvailable:
            raise('Image library pillow is not installed')
        if dimensions != None:
            raise('Extracting subset of data is not implemented')
        input_path = os.path.dirname(self.filename)
        pixels = np.zeros((self.num_projections, self.geometry.pixel_num_h, self.geometry.pixel_num_v), dtype='float32')
        for i in range(1, self.num_projections+1):
            im = Image.open(os.path.join(input_path,self.experiment_name+"_%04d"%i+".tif"))
            pixels[i-1,:,:] = np.fliplr(np.transpose(np.array(im))) ##Not sure this is the correct way to populate the image
            
        #normalising the data
        #TODO: Move this to a processor
        pixels = pixels - (self.white_level*self.scattering)/100.0
        pixels[pixels < 0.0] = 0.000001 # all negative values to approximately 0 as the std log of zero and non negative number is not defined
        return pixels
    
    def get_acquisition_data(self, dimensions=None):
        '''
        This method load the acquisition data and given dimension and returns an AcquisitionData Object
        '''
        data = self.load_projection(dimensions)
        return AcquisitionData(data, geometry=self.geometry)
    
