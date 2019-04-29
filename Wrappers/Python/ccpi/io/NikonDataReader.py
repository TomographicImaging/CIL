#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:25 2019

@author: evelina
"""

from ccpi.framework import AcquisitionData, AcquisitionGeometry
import numpy
import matplotlib.pyplot as plt
import os

pilAvailable = True
try:    
    from PIL import Image
except:
    pilAvailable = False
    
        
class NIKONDataReader(object):
    
    def __init__(self, 
                 **kwargs):
        '''
        Constructor
        
        Input:
            
            xtek_file       full path to .xtexct file
            
            roi             region-of-interest to load. If roi = -1 (default), 
                            full projections will be loaded. Otherwise roi is 
                            given by [row0, column0, row1, column1], where 
                            row0, column0 are coordinates of top left corner and 
                            row1, column1 are coordinates of bottom right corner.
                            
            binning         number of pixels to bin (combine) along 0 (column) 
                            and 1 (row) dimension. If binning = [1, 1] (default),
                            projections in original resolution are loaded. Note, 
                            if binning[0] != binning[1], then loaded projections
                            will have anisotropic pixels, which are currently not 
                            supported by the Framework
            
            normalize       normalize loaded projections by detector 
                            white level (I_0). Default value is False, 
                            i.e. no normalization.
                    
        '''
        
        self.xtek_file = kwargs.get('xtek_file', None)
        self.roi = kwargs.get('roi', -1)
        self.binning = kwargs.get('binning', [1, 1])
        self.normalize = kwargs.get('normalize', False)
        
        if self.xtek_file is not None:
            self.set_up(xtek_file = self.xtek_file,
                        roi = self.roi,
                        binning = self.binning)
            
    def set_up(self, 
               xtek_file, 
               roi = -1, 
               binning = [1, 1],
               normalize = False):
        
        self.xtek_file = xtek_file
        self.roi = roi
        self.binning = binning
        self.normalize = normalize
        
        if self.xtek_file == None:
            raise Exception('Path to xtek file is required.')
        
        # check if xtek file exists
        if not(os.path.isfile(self.xtek_file)):
            raise Exception('File {} does not exist'.format(self.xtek_file))  
        
        # check ROI
        if (self.roi != -1): 
            if not ((isinstance(self.roi, list)) or 
                    (len(self.roi) == 4) or 
                    (self.roi[0] < self.roi[2]) or 
                    (self.roi[1] < self.roi[3])):
                raise Exception('Not valid ROI. ROI must be defined as [row0, column0, row1, column1] \
                                such that ((row0 < row1) and (column0 < column1))')
        
        # check binning parameters
        if (not(isinstance(self.binning, list)) or 
            (len(self.binning) != 2)):
            raise Exception('Not valid binning parameters. \
                            Binning must be defined as [int, int]')
        
        # check that PIL library is installed
        if (pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files")
                
        # parse xtek file
        with open(self.xtek_file, 'r') as f:
            content = f.readlines()    
                
        content = [x.strip() for x in content]
        
        for line in content:
            # filename of TIFF files
            if line.startswith("Name"):
                self._experiment_name = line.split('=')[1]
            # number of projections
            elif line.startswith("Projections"):
                num_projections = int(line.split('=')[1])
            # white level - used for normalization
            elif line.startswith("WhiteLevel"):
                self._white_level = float(line.split('=')[1])
            # number of pixels along Y axis
            elif line.startswith("DetectorPixelsY"):
                pixel_num_v_0 = int(line.split('=')[1])
            # number of pixels along X axis
            elif line.startswith("DetectorPixelsX"):
                pixel_num_h_0 = int(line.split('=')[1])
            # pixel size along X axis
            elif line.startswith("DetectorPixelSizeX"):
                pixel_size_h_0 = float(line.split('=')[1])
            # pixel size along Y axis
            elif line.startswith("DetectorPixelSizeY"):
                pixel_size_v_0 = float(line.split('=')[1])
            # source to center of rotation distance
            elif line.startswith("SrcToObject"):
                source_x = float(line.split('=')[1])
            # source to detector distance
            elif line.startswith("SrcToDetector"):
                detector_x = float(line.split('=')[1])
            # initial angular position of a rotation stage
            elif line.startswith("InitialAngle"):
                initial_angle = float(line.split('=')[1])
            # angular increment (in degrees)
            elif line.startswith("AngularStep"):
                angular_step = float(line.split('=')[1])
    
        # calculate number of pixels and pixel size
        if ((self.binning == [1, 1]) and (self.roi == -1)):
            pixel_num_v = pixel_num_v_0
            pixel_num_h = pixel_num_h_0
            pixel_size_v = pixel_size_v_0
            pixel_size_h = pixel_size_h_0
            
        elif ((self.binning == [1, 1]) and (self.roi != -1)):
            pixel_num_v = self.roi[2] - self.roi[0]
            pixel_num_h = self.roi[3] - self.roi[1]
            pixel_size_v = pixel_size_v_0
            pixel_size_h = pixel_size_h_0
            
        elif ((self.binning > [1, 1]) and (self.roi == -1)):
            pixel_num_v = pixel_num_v_0 // self.binning[0]
            pixel_num_h = pixel_num_h_0 // self.binning[1]
            pixel_size_v = pixel_size_v_0 * self.binning[0]
            pixel_size_h = pixel_size_h_0 * self.binning[1]
            
        elif ((self.binning > [1, 1]) and (self.roi != -1)):
            pixel_num_v = (self.roi[2] - self.roi[0]) // self.binning[0]
            pixel_num_h = (self.roi[3] - self.roi[1]) // self.binning[1]
            pixel_size_v = pixel_size_v_0 * self.binning[0]
            pixel_size_h = pixel_size_h_0 * self.binning[1]
        
        '''
        Parse the angles file .ang or _ctdata.txt file and returns the angles
        as an numpy array. 
        '''
        input_path = os.path.dirname(self.xtek_file)
        angles_ctdata_file = os.path.join(input_path, '_ctdata.txt')
        angles_named_file = os.path.join(input_path, self._experiment_name+'.ang')
        angles = numpy.zeros(num_projections, dtype = 'float')
        
        # look for _ctdata.txt
        if os.path.exists(angles_ctdata_file):
            # read txt file with angles
            with open(angles_ctdata_file) as f:
                content = f.readlines()
            # skip firt three lines
            # read the middle value of 3 values in each line as angles in degrees
            index = 0
            for line in content[3:]:
                angles[index] = float(line.split(' ')[1])
                index += 1
            angles = angles + initial_angle
        
        # look for ang file
        elif os.path.exists(angles_named_file):
            # read the angles file which is text with first line as header
            with open(angles_named_file) as f:
                content = f.readlines()
            # skip first line
            index = 0
            for line in content[1:]:
                angles[index] = float(line.split(':')[1])
                index += 1
            angles = numpy.flipud(angles + initial_angle) # angles are in the reverse order
            
        else:   # calculate angles based on xtek file
            angles = initial_angle + angular_step * range(num_projections)
        
        # fill in metadata
        self._ag = AcquisitionGeometry(geom_type = 'cone', 
                                       dimension = '3D', 
                                       angles = angles, 
                                       pixel_num_h = pixel_num_h, 
                                       pixel_size_h = pixel_size_h, 
                                       pixel_num_v = pixel_num_v, 
                                       pixel_size_v = pixel_size_v, 
                                       dist_source_center = source_x, 
                                       dist_center_detector = detector_x - source_x, 
                                       channels = 1,
                                       angle_unit = 'degree')

    def get_acquisition_geometry(self):
        '''
        Return AcquisitionGeometry object
        '''
        
        return(self._ag)
        
        
    def load_projections(self):
        '''
        Load projections in AcquisitionData container
        '''
        
        # get path to projections
        path_projection = os.path.dirname(self.xtek_file)
        
        # get number of projections
        num_projections = numpy.shape(self._ag.angles)[0]
        
        # allocate array to store projections    
        data = numpy.zeros((num_projections, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
        
        for i in range(num_projections):
            
            file = (path_projection + '/' + self._experiment_name + '_{:04d}.tif').format(i + 1)
            tmp = numpy.asarray(Image.open(file), dtype = float)
            
            if ((self.binning == [1, 1]) and (self.roi == -1)):
                data[i, :, :] = tmp
                
            elif ((self.binning == [1, 1]) and (self.roi != -1)):
                data[i, :, :] = tmp[self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]]
                
            elif ((self.binning > [1, 1]) and (self.roi == -1)):
                shape = (self._ag.pixel_num_v, self.binning[0], 
                         self._ag.pixel_num_h, self.binning[1])
                data[i, :, :] = tmp[:(self._ag.pixel_num_v * self.binning[0]), \
                                    :(self._ag.pixel_num_h * self.binning[1])].reshape(shape).mean(-1).mean(1)
                        
            elif ((self.binning > [1, 1]) and (self.roi != -1)):
                shape = (self._ag.pixel_num_v, self.binning[0], 
                         self._ag.pixel_num_h, self.binning[1])
                data[i, :, :] = tmp[self.roi[0]:(self.roi[0] + (((self.roi[2] - self.roi[0]) // self.binning[0]) * self.binning[0])), \
                                    self.roi[1]:(self.roi[1] + (((self.roi[3] - self.roi[1]) // self.binning[1]) * self.binning[1]))].reshape(shape).mean(-1).mean(1)
        
        if (self.normalize):
            data /= self._white_level
            data[data > 1] = 1
            
        return AcquisitionData(array = data, 
                               geometry = self._ag,
                               dimension_labels = ['angle', 'vertical', 'horizontal'])


'''
# usage example

xtek_file = '/home/evelina/nikon_data/SophiaBeads_256_averaged.xtekct'
reader = NIKONDataReader()
reader.set_up(xtek_file = xtek_file,
              binning = [3, 1],
              roi = [200, 500, 1500, 2000],
              normalize = True)

data = reader.load_projections()
ag = reader.get_acquisition_geometry()
print(ag)

plt.imshow(data.as_array()[1, :, :])
plt.show()
'''
