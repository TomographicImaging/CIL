from ccpi.framework import DataSetProcessor, DataSet, VolumeData, SinogramData
from ccpi.astra.astra_utils import convert_geometry_to_astra
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
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' is only for parallel (2D) and only one option
            self.setProjector(astra.create_projector('line', proj_geom, vol_geom) )
        elif device == 'gpu':
            self.setProjector(astra.create_projector('cuda', proj_geom, vol_geom) )
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
        
                # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' is only for parallel (2D) and only one option
            self.setProjector(astra.create_projector('line', proj_geom, vol_geom) )
        elif device == 'gpu':
            self.setProjector(astra.create_projector('cuda', proj_geom, vol_geom) )
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