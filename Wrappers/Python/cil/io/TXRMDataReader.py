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

from cil.framework import AcquisitionData, AcquisitionGeometry
import numpy as np
import os
import olefile
    
        
class TXRMDataReader(object):
    
    def __init__(self, 
                 **kwargs):
        '''
        Constructor
        
        Input:
            
            txrm_file       full path to .txrm file
                    
        '''
        
        self.txrm_file = kwargs.get('file_name', None)
        
        if self.txrm_file is not None:
            self.set_up(file_name = self.txrm_file)
        self._metadata = None
            
    def set_up(self, 
               file_name = None,
               angle_unit = AcquisitionGeometry.DEGREE):
        
        self.txrm_file = os.path.abspath(file_name)
        
        if self.txrm_file == None:
            raise ValueError('Path to txrm file is required.')
        
        # check if txrm file exists
        if not(os.path.isfile(self.txrm_file)):
            raise FileNotFoundError('{}'.format(self.txrm_file))  
        possible_units = [AcquisitionGeometry.DEGREE, AcquisitionGeometry.RADIAN]
        if angle_unit in possible_units:
            self.angle_unit = angle_unit
        else:
            raise ValueError('angle_unit should be one of {}'.format(possible_units))

    def get_geometry(self):
        '''
        Return AcquisitionGeometry object
        '''
        
        return self._ag
        
    def read(self):
        '''
        Reads projections and return AcquisitionData container
        '''
        # the import will raise an ImportError if dxchange is not installed
        import dxchange
        # Load projections and most metadata
        data, metadata = dxchange.read_txrm(self.txrm_file)
        for k,v in metadata.items():
            print (k,v)
        number_of_images = data.shape[0]
        
        # Read source to center and detector to center distances
        with olefile.OleFileIO(self.txrm_file) as ole:
            StoRADistance = dxchange.reader._read_ole_arr(ole, \
                    'ImageInfo/StoRADistance', "<{0}f".format(number_of_images))
            DtoRADistance = dxchange.reader._read_ole_arr(ole, \
                    'ImageInfo/DtoRADistance', "<{0}f".format(number_of_images))
            
        dist_source_center   = np.abs( StoRADistance[0] )
        dist_center_detector = np.abs( DtoRADistance[0] )
        
        # normalise data by flatfield
        data = data / metadata['reference']
        
        # circularly shift data by rounded x and y shifts
        for k in range(number_of_images):
            data[k,:,:] = np.roll(data[k,:,:], \
                (int(metadata['x-shifts'][k]),int(metadata['y-shifts'][k])), \
                axis=(1,0))
        
        # Pixelsize loaded in metadata is really the voxel size in um.
        # We can compute the effective detector pixel size as the geometric
        # magnification times the voxel size.
        d_pixel_size = ((dist_source_center+dist_center_detector)/dist_source_center)*metadata['pixel_size']
        
        # convert angles to requested unit measure, Zeiss stores in radians
        # AND 
        # convert direction of rotation from Zeiss to our convention
        if self.angle_unit == AcquisitionGeometry.DEGREE:
            angles = - np.degrees(metadata['thetas'])
        else:
            angles = - np.asarray(metadata['thetas'])

        self._ag = AcquisitionGeometry(geom_type = 'cone', 
                                       dimension = '3D', 
                                       angles = angles, 
                                       pixel_num_h = metadata['image_width'], 
                                       pixel_size_h = d_pixel_size/1000, 
                                       pixel_num_v = metadata['image_height'], 
                                       pixel_size_v = d_pixel_size/1000, 
                                       dist_source_center =  dist_source_center, 
                                       dist_center_detector = dist_center_detector, 
                                       channels = 1,
                                       angle_unit = self.angle_unit,
                                       dimension_labels = ['angle', \
                                                           'vertical', \
                                                           'horizontal'])
        acq_data = self._ag.allocate(None)
        acq_data.fill(data)
        self._metadata = metadata
        return acq_data
    def load_projections(self):
        '''alias of read for backward compatibility'''
        return self.read()
        
    def get_metadata(self):
        '''return the metadata of the loaded file'''
        return self._metadata
if __name__ == '__main__':
    
    from cil.framework import ImageGeometry
    from cil.astra.processors import FBP
    from cil.io import NEXUSDataWriter
    import matplotlib.pyplot as plt
    from cil.framework.dataexample import data_dir

    angle_unit = AcquisitionGeometry.RADIAN

    #filename = "/media/newhd/shared/Data/zeiss/walnut/valnut/valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm"
    filename = os.path.join(data_dir, "valnut_tomo-A.txrm")
    reader = TXRMDataReader()
    reader.set_up(txrm_file=filename, 
                  angle_unit=angle_unit)
    
    data = reader.load_projections()
    
    print('done loading')
    
    # get central slice
    data2d = data.subset(vertical=512)
    # neg log
    data2d.log(out=data2d)
    data2d *= -1

    # Choose the number of voxels to reconstruct onto as number of detector pixels
    N = data.geometry.pixel_num_h
    
    # Geometric magnification
    mag = (np.abs(data.geometry.dist_center_detector) + \
           np.abs(data.geometry.dist_source_center)) / \
           np.abs(data.geometry.dist_source_center)
           
    # Voxel size is detector pixel size divided by mag
    voxel_size_h = data.geometry.pixel_size_h / mag
    
    # Construct the appropriate ImageGeometry
    ig2d = ImageGeometry(voxel_num_x=N,
                         voxel_num_y=N,
                         voxel_size_x=voxel_size_h, 
                         voxel_size_y=voxel_size_h)
    ig3d = ImageGeometry(voxel_num_x=N,
                         voxel_num_y=N,
                         voxel_num_z=N,
                         voxel_size_x=voxel_size_h, 
                         voxel_size_y=voxel_size_h,
                         voxel_size_z=data.geometry.pixel_size_v / mag)
    
    print('done set up astra op')
    
    fbpalg = FBP(ig2d,data2d.geometry)
    fbpalg.set_input(data2d)
    
    recfbp = fbpalg.get_output()
    
    plt.figure()
    plt.imshow(recfbp.as_array())
    plt.gray()
    plt.colorbar()

    writer = NEXUSDataWriter()
    cwd = os.getcwd()
    writer.set_up(data_container = recfbp,
                  file_name = os.path.join(cwd,'walnut_slice512.nxs'))
    writer.write_file()
    
    shuffle = data.subset(dimensions=['vertical','angle','horizontal'])
    shuffle.log(out=shuffle)
    shuffle *= -1

    fbp3d = FBP(ig3d, shuffle.geometry)
    fbp3d.set_input(shuffle)
    vol = fbp3d.get_output()
    
    plt.figure()
    plt.imshow(vol.subset(vertical=512).as_array())
    plt.gray()
    plt.colorbar()
    
    writer.set_up(data_container = fbp3d.get_output(),
                  file_name = os.path.join(cwd,'walnut_3D.nxs'))
    writer.write_file()
    plt.show()
    