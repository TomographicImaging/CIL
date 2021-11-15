# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.framework import DataProcessor, ImageData
from cil.framework import DataOrder
from cil.plugins.tigre import CIL2TIGREGeometry

import numpy as np

try:
    from tigre.algorithms import fdk, fbp
except ModuleNotFoundError:
    raise ModuleNotFoundError("This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel")

class FBP(DataProcessor):

    '''FBP Filtered Back Projection is a reconstructor for 2D and 3D parallel and cone-beam geometries.
    It is able to back-project circular trajectories with 2 PI anglar range and equally spaced anglular steps.

    This uses the ram-lak filter
    This is provided for simple and offset parallel-beam geometries only
   
    Input: Volume Geometry
           Sinogram Geometry
                             
    Example:  fbp = FBP(ig, ag, device)
              fbp.set_input(data)
              reconstruction = fbp.get_ouput()
                           
    Output: ImageData                             

        '''
    
    def __init__(self, volume_geometry, sinogram_geometry): 
        
        DataOrder.check_order_for_engine('tigre', volume_geometry)
        DataOrder.check_order_for_engine('tigre', sinogram_geometry) 

        tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(volume_geometry,sinogram_geometry)

        super(FBP, self).__init__(  volume_geometry = volume_geometry, sinogram_geometry = sinogram_geometry,\
                                    tigre_geom=tigre_geom, tigre_angles=tigre_angles)


    def check_input(self, dataset):
        
        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))  

        DataOrder.check_order_for_engine('tigre', dataset.geometry)
        return True

    def process(self, out=None):
        
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(self.get_input().as_array(), axis=1)

            if self.sinogram_geometry.geom_type == 'cone':
                arr_out = fdk(data_temp, self.tigre_geom, self.tigre_angles)
            else:
                arr_out = fbp(data_temp, self.tigre_geom, self.tigre_angles)
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            if self.sinogram_geometry.geom_type == 'cone':
                arr_out = fdk(self.get_input().as_array(), self.tigre_geom, self.tigre_angles)
            else:
                arr_out = fbp(self.get_input().as_array(), self.tigre_geom, self.tigre_angles)

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
            