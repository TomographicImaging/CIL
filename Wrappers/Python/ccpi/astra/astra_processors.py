from ccpi.framework import DataSetProcessor, DataSet, VolumeData, SinogramData
from ccpi.astra.astra_utils import convert_geometry_to_astra
import astra
import numpy


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
            # Note that 'line' only one option
            if self.sinogram_geometry.geom_type == 'parallel':
                self.setProjector(astra.create_projector('line', proj_geom, vol_geom) )
            elif self.sinogram_geometry.geom_type == 'cone':
                self.setProjector(astra.create_projector('line_fanflat', proj_geom, vol_geom) )
            else:
                NotImplemented    
        elif device == 'gpu':
            self.setProjector(astra.create_projector('cuda', proj_geom, vol_geom) )
        else:
            NotImplemented
    
    def checkInput(self, dataset):
        if dataset.number_of_dimensions == 3 or\
           dataset.number_of_dimensions == 2:
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
        DATA = SinogramData(geometry=self.sinogram_geometry)
        #sinogram_id, DATA = astra.create_sino( IM.as_array(), 
        #                           self.proj_id)
        sinogram_id, DATA.array = astra.create_sino(IM.as_array(), 
                                                           self.proj_id)
        astra.data2d.delete(sinogram_id)
        #return SinogramData(array=DATA, geometry=self.sinogram_geometry)
        return DATA

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
            # Note that 'line' only one option
            if self.sinogram_geometry.geom_type == 'parallel':
                self.setProjector(astra.create_projector('line', proj_geom, vol_geom) )
            elif self.sinogram_geometry.geom_type == 'cone':
                self.setProjector(astra.create_projector('line_fanflat', proj_geom, vol_geom) )
            else:
                NotImplemented 
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
        IM = VolumeData(geometry=self.volume_geometry)
        rec_id, IM.array = astra.create_backprojection(DATA.as_array(),
                            self.proj_id)
        astra.data2d.delete(rec_id)
        return IM

class AstraForwardProjectorMC(AstraForwardProjector):
    '''AstraForwardProjector Multi channel
    
    Forward project VolumeDataSet to SinogramDataSet using ASTRA proj_id.
    
    Input: VolumeDataSet
    Parameter: proj_id
    Output: SinogramDataSet
    '''
    def checkInput(self, dataset):
        if dataset.number_of_dimensions == 2 or \
           dataset.number_of_dimensions == 3 or \
           dataset.number_of_dimensions == 4:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    def process(self):
        IM = self.getInput()
        #create the output Sinogram
        DATA = SinogramData(geometry=self.sinogram_geometry)
        
        for k in range(DATA.geometry.channels):
            sinogram_id, DATA.as_array()[k] = astra.create_sino(IM.as_array()[k], 
                                                           self.proj_id)
            astra.data2d.delete(sinogram_id)
        return DATA

class AstraBackProjectorMC(AstraBackProjector):
    '''AstraBackProjector Multi channel
    
    Back project SinogramDataSet to VolumeDataSet using ASTRA proj_id.
    
    Input: SinogramDataSet
    Parameter: proj_id
    Output: VolumeDataSet 
    '''
    def checkInput(self, dataset):
        if dataset.number_of_dimensions == 2 or \
           dataset.number_of_dimensions == 3 or \
           dataset.number_of_dimensions == 4:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    def process(self):
        DATA = self.getInput()
        
        IM = VolumeData(geometry=self.volume_geometry)
        
        for k in range(IM.geometry.channels):
            rec_id, IM.as_array()[k] = astra.create_backprojection(DATA.as_array()[k], 
                                  self.proj_id)
            astra.data2d.delete(rec_id)
        return IM