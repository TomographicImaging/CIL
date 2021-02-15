from cil.framework import DataProcessor, ImageData
from cil.plugins.tigre import CIL2TIGREGeometry
import numpy as np

try:
    from tigre.algorithms import fdk
except ModuleNotFoundError:
    raise ModuleNotFoundError("This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel")

class FBP(DataProcessor):

    '''FBP Filtered Back Projection is a reconstructor for 2D and 3D parallel and cone-beam geometries.
    It is able to back-project circular trajectories with 2 PI anglar range and equally spaced anglular steps.

    This uses the ram-lak filter
    This is provided for simple parallel-beam geometries only (offsets and rotations will be ignored)
   
    Input: Volume Geometry
           Sinogram Geometry
                             
    Example:  fbp = FBP(ig, ag, device)
              fbp.set_input(data)
              reconstruction = fbp.get_ouput()
                           
    Output: ImageData                             

        '''
    
    def __init__(self, volume_geometry, sinogram_geometry): 
        

        tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(volume_geometry,sinogram_geometry)

        super(FBP, self).__init__(  volume_geometry = volume_geometry, sinogram_geometry = sinogram_geometry,\
                                    tigre_geom=tigre_geom, tigre_angles=tigre_angles)


    def check_input(self, dataset):
        
        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))  

        return True

    def process(self, out=None):
        
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(self.get_input().as_array(), axis=1)
            arr_out = fdk(data_temp, self.tigre_geom, self.tigre_angles)
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = fdk(self.get_input().as_array(), self.tigre_geom, self.tigre_angles)

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
            