# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from cil.framework import DataProcessor, ImageData
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D
import astra
import numpy as np


class FDK_Flexible(DataProcessor):

    '''FDK_Flexible Filtered Back Projection is a reconstructor for 2D and 3D cone-beam geometries.
    It is able to back-project circular trajectories with 2 PI anglar range and equally spaced anglular steps.

    This uses the ram-lak filter
    This is a GPU version only
    
    Input: Volume Geometry
           Sinogram Geometry
                             
    Example:  fdk = FDK_Flexible(ig, ag)
              fdk.set_input(data)
              reconstruction = fdk.get_ouput()
                           
    Output: ImageData                             

    
    '''

    def __init__(self, volume_geometry, 
                       sinogram_geometry): 
        
        vol_geom_astra, proj_geom_astra = convert_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry)
 

        super(FDK_Flexible, self).__init__( volume_geometry = volume_geometry,
                                        sinogram_geometry = sinogram_geometry,
                                        vol_geom_astra = vol_geom_astra,
                                        proj_geom_astra = proj_geom_astra)


                          
    def check_input(self, dataset):
        
        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))  

        if self.sinogram_geometry.geom_type != 'cone':
            raise ValueError("Expected input data to be cone beam geometry , got {0}"\
                 .format(self.sinogram_geometry.geom_type))  

        return True
        

    def process(self, out=None):
           
        # Get DATA
        DATA = self.get_input()

        pad = False
        if len(DATA.shape) == 2:
            #for 2D cases
            pad = True
            data_temp = np.expand_dims(DATA.as_array(),axis=0)
        else:
            data_temp = DATA.as_array()


        rec_id = astra.data3d.create('-vol', self.vol_geom_astra)
        sinogram_id = astra.data3d.create('-sino', self.proj_geom_astra, data_temp)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)
        
        astra.algorithm.run(alg_id)       
        arr_out = astra.data3d.get(rec_id)

        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinogram_id)                    
        astra.algorithm.delete(alg_id)

        if pad == True:
            arr_out = np.squeeze(arr_out, axis=0)

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
