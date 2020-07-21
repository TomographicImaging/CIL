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
        
        # check if txrm file exists
        if not(os.path.isfile(self.txrm_file)):
            raise Exception('File\n {}\n does not exist.'.format(self.txrm_file))  

    def get_geometry(self):
        
        '''
        Return AcquisitionGeometry object
        '''
        
        return self._ag
        
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
                                       pixel_size_h = d_pixel_size/1000, 
                                       pixel_num_v = metadata['image_height'], 
                                       pixel_size_v = d_pixel_size/1000, 
                                       dist_source_center =  StoCdist, 
                                       dist_center_detector = DtoCdist, 
                                       channels = 1,
                                       angle_unit = 'degree')

        return AcquisitionData(array = data, 
                               deep_copy = False,
                               geometry = self._ag,
                               dimension_labels = ['angle', \
                                                   'vertical', \
                                                   'horizontal'])
                
if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    from ccpi.astra.processors import FBP
    import matplotlib.pyplot as plt
    
    filename = "/media/newhd/shared/Data/zeiss/walnut/valnut/valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm"
    reader = TXRMDataReader()
    reader.set_up(txrm_file=filename)
    
    data = reader.load_projections()
    
    print('done loading')
    
    plt.figure()
    plt.imshow(data.subset(angle=0).as_array())
    plt.colorbar()
    plt.gray()
    plt.savefig('walnut_proj0.png',dpi=300)
    
    
    plt.figure()
    plt.imshow(data.subset(angle=800).as_array())
    plt.colorbar()
    plt.gray()
    plt.savefig('walnut_proj800.png',dpi=300)
    
    # Extract AcquisitionGeometry for central slice for 2D fanbeam reconstruction
    ag2d = AcquisitionGeometry('cone',
                          '2D',
                          angles=-numpy.pi/180*data.geometry.angles,
                          pixel_num_h=data.geometry.pixel_num_h,
                          pixel_size_h=data.geometry.pixel_size_h,
                          dist_source_center=data.geometry.dist_source_center, 
                          dist_center_detector=data.geometry.dist_center_detector)
    
    # Set up AcquisitionData object for central slice 2D fanbeam
    data2d = AcquisitionData(data.subset(vertical=512),geometry=ag2d)
    
    plt.figure()
    plt.imshow(data2d.as_array())
    plt.colorbar()
    plt.gray()
    
    data2d.log(out=data2d)
    data2d *= -1
    
    plt.figure()
    plt.imshow(data2d.as_array())
    plt.colorbar()
    plt.gray()
    
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
    
    print('done set up astra op')
    
    fbpalg = FBP(ig2d,ag2d)

    fbpalg.set_input(data2d)
    
    recfbp = fbpalg.get_output()
    
    plt.figure()
    plt.imshow(recfbp.as_array())
    plt.gray()
    plt.colorbar()
    
    