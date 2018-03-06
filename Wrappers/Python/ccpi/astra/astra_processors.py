from ccpi.framework import DataSetProcessor, DataSet, VolumeData, SinogramData
import astra


class AstraForwardProjector(DataSetProcessor):
    '''AstraForwardProjector
    
    Forward project VolumeDataSet to SinogramDataSet using ASTRA proj_id.
    
    Input: VolumeDataSet
    Parameter: proj_id
    Output: SinogramDataSet
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_id=None,
                 device='cpu'):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_id'  : proj_id,
                  'device'  : device
                  }
        
        #DataSetProcessor.__init__(self, **kwargs)
        super(AstraForwardProjector, self).__init__(**kwargs)
        
        self.setVolumeGeometry(volume_geometry)
        self.setSinogramGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not stored
        if self.sinogram_geometry.dimension == '2D':
            vol_geom = astra.create_vol_geom(self.volume_geometry.voxel_num_x, 
                                             self.volume_geometry.voxel_num_y, 
                                             self.volume_geometry.getMinX(), 
                                             self.volume_geometry.getMaxX(), 
                                             self.volume_geometry.getMinY(), 
                                             self.volume_geometry.getMaxY())
            
            if self.sinogram_geometry.geom_type == 'parallel':
                proj_geom = astra.create_proj_geom('parallel',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles)
            elif self.sinogram_geometry.geom_type == 'cone':
                proj_geom = astra.create_proj_geom('fanflat',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles,
                                                   self.sinogram_geometry.dist_source_center,
                                                   self.sinogram_geometry.dist_center_detector)
            else:
                NotImplemented
                
        elif self.sinogram_geometry.dimension == '3D':
            vol_geom = astra.create_vol_geom(self.volume_geometry.voxel_num_x, 
                                             self.volume_geometry.voxel_num_y, 
                                             self.volume_geometry.voxel_num_z, 
                                             self.volume_geometry.getMinX(), 
                                             self.volume_geometry.getMaxX(), 
                                             self.volume_geometry.getMinY(), 
                                             self.volume_geometry.getMaxY(), 
                                             self.volume_geometry.getMinZ(), 
                                             self.volume_geometry.getMaxZ())
            
            if self.sinogram_geometry.proj_geom == 'parallel':
                proj_geom = astra.create_proj_geom('parallel3d',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_size_v,
                                                   self.sinogram_geometry.pixel_num_v,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles)
            elif self.sinogram_geometry.geom_type == 'cone':
                proj_geom = astra.create_proj_geom('cone',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_size_v,
                                                   self.sinogram_geometry.pixel_num_v,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles,
                                                   self.sinogram_geometry.dist_source_center,
                                                   self.sinogram_geometry.dist_center_detector)
            else:
                NotImplemented
                
        else:
            NotImplemented
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' is only for parallel (2D) and only one option
            self.proj_id = astra.create_projector('line', proj_geom, vol_geom) 
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', proj_geom, vol_geom) 
        else:
            NotImplemented
    
    def checkInput(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def setProjector(self, proj_id):
        self.proj_id = proj_id
    
    def setVolumeGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
    
    def setSinogramGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def process(self):
        IM = self.getInput()
        sinogram_id, DATA = astra.create_sino(IM.as_array(), self.proj_id)
        astra.data2d.delete(sinogram_id)
        return SinogramData(DATA,geometry=self.sinogram_geometry)

class AstraBackProjector(DataSetProcessor):
    '''AstraBackProjector
    
    Back project SinogramDataSet to VolumeDataSet using ASTRA proj_id.
    
    Input: SinogramDataSet
    Parameter: proj_id
    Output: VolumeDataSet 
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_id=None,
                 device='cpu'):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_id'  : proj_id,
                  'device'  : device
                  }
        
        #DataSetProcessor.__init__(self, **kwargs)
        super(AstraBackProjector, self).__init__(**kwargs)
        
        self.setVolumeGeometry(volume_geometry)
        self.setSinogramGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not stored
        if self.sinogram_geometry.dimension == '2D':
            vol_geom = astra.create_vol_geom(self.volume_geometry.voxel_num_x, 
                                             self.volume_geometry.voxel_num_y, 
                                             self.volume_geometry.getMinX(), 
                                             self.volume_geometry.getMaxX(), 
                                             self.volume_geometry.getMinY(), 
                                             self.volume_geometry.getMaxY())
            
            if self.sinogram_geometry.geom_type == 'parallel':
                proj_geom = astra.create_proj_geom('parallel',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles)
            elif self.sinogram_geometry.geom_type == 'cone':
                proj_geom = astra.create_proj_geom('fanflat',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles,
                                                   self.sinogram_geometry.dist_source_center,
                                                   self.sinogram_geometry.dist_center_detector)
            else:
                NotImplemented
                
        elif self.sinogram_geometry.dimension == '3D':
            vol_geom = astra.create_vol_geom(self.volume_geometry.voxel_num_x, 
                                             self.volume_geometry.voxel_num_y, 
                                             self.volume_geometry.voxel_num_z, 
                                             self.volume_geometry.getMinX(), 
                                             self.volume_geometry.getMaxX(), 
                                             self.volume_geometry.getMinY(), 
                                             self.volume_geometry.getMaxY(), 
                                             self.volume_geometry.getMinZ(), 
                                             self.volume_geometry.getMaxZ())
            
            if self.sinogram_geometry.proj_geom == 'parallel':
                proj_geom = astra.create_proj_geom('parallel3d',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_size_v,
                                                   self.sinogram_geometry.pixel_num_v,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles)
            elif self.sinogram_geometry.geom_type == 'cone':
                proj_geom = astra.create_proj_geom('cone',
                                                   self.sinogram_geometry.pixel_size_h,
                                                   self.sinogram_geometry.pixel_size_v,
                                                   self.sinogram_geometry.pixel_num_v,
                                                   self.sinogram_geometry.pixel_num_h,
                                                   self.sinogram_geometry.angles,
                                                   self.sinogram_geometry.dist_source_center,
                                                   self.sinogram_geometry.dist_center_detector)
            else:
                NotImplemented
                
        else:
            NotImplemented
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' is only for parallel (2D) and only one option
            self.proj_id = astra.create_projector('line', proj_geom, vol_geom) 
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', proj_geom, vol_geom) 
        else:
            NotImplemented
    
    def checkInput(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def setProjector(self, proj_id):
        self.proj_id = proj_id
        
    def setVolumeGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
        
    def setSinogramGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def process(self):
        DATA = self.getInput()
        rec_id, IM = astra.create_backprojection(DATA.as_array(), self.proj_id)
        astra.data2d.delete(rec_id)
        return VolumeData(IM,geometry=self.volume_geometry)