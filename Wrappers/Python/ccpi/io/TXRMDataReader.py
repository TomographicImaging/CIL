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

from ccpi.framework import AcquisitionData, AcquisitionGeometry
import numpy
import os

import dxchange
import olefile
    
        
class TXRMDataReader(object):
    
    def __init__(self, 
                 **kwargs):
        '''
        Constructor
        
        Input:
            
            txrm_file       full path to .txrm file
                    
        '''
        
        self.txrm_file = kwargs.get('txrm_file', None)
        
        if self.txrm_file is not None:
            self.set_up(txrm_file = self.txrm_file)
            
    def set_up(self, 
               txrm_file = None):
        
        self.txrm_file = txrm_file
        
        if self.txrm_file == None:
            raise Exception('Path to txrm file is required.')
        
        # check if xtek file exists
        if not(os.path.isfile(self.txrm_file)):
            raise Exception('File\n {}\n does not exist.'.format(self.txrm_file))  
                
        '''        
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
                
        if self.roi == -1:
            self._roi_par = [(0, pixel_num_v_0), \
                              (0, pixel_num_h_0)]
        else:
            self._roi_par = self.roi.copy()
            if self._roi_par[0] == -1:
                self._roi_par[0] = (0, pixel_num_v_0)
            if self._roi_par[1] == -1:
                self._roi_par[1] = (0, pixel_num_h_0)
                
        # calculate number of pixels and pixel size
        if (self.binning == [1, 1]):
            pixel_num_v = self._roi_par[0][1] - self._roi_par[0][0]
            pixel_num_h = self._roi_par[1][1] - self._roi_par[1][0]
            pixel_size_v = pixel_size_v_0
            pixel_size_h = pixel_size_h_0
        else:
            pixel_num_v = (self._roi_par[0][1] - self._roi_par[0][0]) // self.binning[0]
            pixel_num_h = (self._roi_par[1][1] - self._roi_par[1][0]) // self.binning[1]
            pixel_size_v = pixel_size_v_0 * self.binning[0]
            pixel_size_h = pixel_size_h_0 * self.binning[1]
        '''
        
        '''
        Parse the angles file .ang or _ctdata.txt file and returns the angles
        as an numpy array. 
        '''
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
        '''

    #def get_geometry(self):
    #    
    #    '''
    #    Return AcquisitionGeometry object
    #    '''
    #    
    #    return self._ag
        
    def load_projections(self):
        
        '''
        Load projections and return AcquisitionData container
        '''
        
        # Load projections and most metadata
        data, metadata = dxchange.read_txrm(self.txrm_file)
        number_of_images = data.shape[0]
        
        # Read source to center and detector to center distances
        ole = olefile.OleFileIO(self.txrm_file)
        StoCdist = dxchange.reader._read_ole_arr(ole, \
                'ImageInfo/StoRADistance', "<{0}f".format(number_of_images))
        DtoCdist = dxchange.reader._read_ole_arr(ole, \
                'ImageInfo/DtoRADistance', "<{0}f".format(number_of_images))
        ole.close()
        StoCdist = numpy.abs(StoCdist[0])
        DtoCdist = numpy.abs(DtoCdist[0])
        
        # normalise data by flatfield
        data = data / metadata['reference']
        
        # circularly shift data by rounded x and y shifts
        for k in range(number_of_images):
            data[k,:,:] = numpy.roll(data[k,:,:], \
                (int(metadata['x-shifts'][k]),int(metadata['y-shifts'][k])), \
                axis=(1,0))
        
        # Pixelsize loaded in metadata is really the voxel size in um.
        # We can compute the effective detector pixel size as the geometric
        # magnification times the voxel size.
        d_pixel_size = ((StoCdist+DtoCdist)/StoCdist)*metadata['pixel_size']
        
        self._ag = AcquisitionGeometry(geom_type = 'cone', 
                                       dimension = '3D', 
                                       angles = numpy.degrees(metadata['thetas']), 
                                       pixel_num_h = metadata['image_width'], 
                                       pixel_size_h = d_pixel_size, 
                                       pixel_num_v = metadata['image_height'], 
                                       pixel_size_v = d_pixel_size, 
                                       dist_source_center =  StoCdist, 
                                       dist_center_detector = DtoCdist, 
                                       channels = 1,
                                       angle_unit = 'degree')
        
        
        #if (self.normalize):
        #    data /= self._white_level
        #    data[data > 1] = 1

        return AcquisitionData(array = data, 
                               deep_copy = False,
                               geometry = self._ag,
                               dimension_labels = ['angle', \
                                                   'vertical', \
                                                   'horizontal'])
                
if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry, ImageData
    from ccpi.astra.operators import AstraProjectorSimple
    from ccpi.optimisation.algorithms import FISTA, CGLS
    
    filename = "/media/newhd/shared/Data/zeiss/walnut/valnut/valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm"
    reader = TXRMDataReader()
    reader.set_up(txrm_file=filename)
    
    data = reader.load_projections()
    
    # Extract AcquisitionGeometry for central slice for 2D fanbeam reconstruction
    ag2d = AcquisitionGeometry('cone',
                          '2D',
                          angles=data.geometry.angles,
                          pixel_num_h=data.geometry.pixel_num_h,
                          pixel_size_h=data.geometry.pixel_size_h,
                          dist_source_center=data.geometry.dist_source_center, 
                          dist_center_detector=data.geometry.dist_center_detector)
    
    # Set up AcquisitionData object for central slice 2D fanbeam
    data2d = AcquisitionData(data.subset(vertical=512),geometry=ag2d)
    
    # Choose the number of voxels to reconstruct onto as number of detector pixels
    N = data.geometry.pixel_num_h
    
    # Geometric magnification
    mag = (numpy.abs(data.geometry.dist_center_detector) + \
           numpy.abs(data.geometry.dist_source_center)) / \
           numpy.abs(data.geometry.dist_source_center)
           
    # Voxel size is detector pixel size divided by mag
    voxel_size_h = data.geometry.pixel_size_h / mag
    
    # Construct the appropriate ImageGeometry
    ig2d = ImageGeometry(voxel_num_x=N,
                         voxel_num_y=N,
                         voxel_size_x=voxel_size_h, 
                         voxel_size_y=voxel_size_h)
    
    # Set up the Projector (AcquisitionModel) using ASTRA on GPU
    Aop = AstraProjectorSimple(ig2d, ag2d,"gpu")
    
    # Set initial guess for CGLS reconstruction
    x_init = ImageData(geometry=ig2d)
    
    # Set tolerance and number of iterations for reconstruction algorithms.
    opt = {'tol': 1e-4, 'iter': 50}
    
    # First a CGLS reconstruction can be done:
    CGLS_alg = CGLS()
    CGLS_alg.set_up(x_init, Aop, data2d)
    CGLS_alg.max_iteration = 2000
    CGLS_alg.run(opt['iter'])


'''
# usage example
xtek_file = '/home/evelina/nikon_data/SophiaBeads_256_averaged.xtekct'
reader = NikonDataReader()
reader.set_up(xtek_file = xtek_file,
              binning = [1, 1],
              roi = -1,
              normalize = True,
              flip = True)

data = reader.load_projections()
print(data)
ag = reader.get_geometry()
print(ag)

plt.imshow(data.as_array()[1, :, :])
plt.show()
'''
