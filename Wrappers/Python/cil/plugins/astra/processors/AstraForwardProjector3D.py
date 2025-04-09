#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
from cil.framework import DataProcessor, AcquisitionData
from cil.framework.labels import ImageDimension, AcquisitionDimension
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D
import astra
import astra.experimental
import numpy as np

class AstraForwardProjector3D(DataProcessor):
    """
    AstraForwardProjector3D configures an ASTRA 3D forward projector for GPU.

    Parameters
    ----------

    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    sinogram_geometry : AcquisitionGeometry
        A description of the acquisition data

    """

    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'projector_id'  : None
                  }

        super(AstraForwardProjector3D, self).__init__(**kwargs)

        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)

        vol_geom, proj_geom = convert_geometry_to_astra_vec_3D(self.volume_geometry, self.sinogram_geometry)
        self.projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

    def __del__(self):
        astra.projector3d.delete(self.projector_id)


    def check_input(self, dataset):

        if self.volume_geometry.shape != dataset.geometry.shape:
            raise ValueError("Dataset not compatible with geometry used to create the projector")

        return True
    
    def _set_up(self):
        """
        Configure processor attributes that require the data to setup
        Must set _shape_out
        """
        self._shape_out = self.sinogram_geometry.shape

    def set_ImageGeometry(self, volume_geometry):

        ImageDimension.check_order_for_engine('astra', volume_geometry)

        if len(volume_geometry.dimension_labels) > 3:
            raise ValueError("Supports 2D and 3D data only, got {0}".format(volume_geometry.number_of_dimensions))

        self.volume_geometry = volume_geometry.copy()

    def set_AcquisitionGeometry(self, sinogram_geometry):

        AcquisitionDimension.check_order_for_engine('astra', sinogram_geometry)

        if len(sinogram_geometry.dimension_labels) > 3:
            raise ValueError("Supports 2D and 3D data only, got {0}".format(sinogram_geometry.number_of_dimensions))

        self.sinogram_geometry = sinogram_geometry.copy()


    def process(self, out=None):

        IM = self.get_input()

        #ASTRA expects a 3D array with shape 1, CIL removes dimensions of len 1
        new_shape_ig = [self.volume_geometry.voxel_num_z,self.volume_geometry.voxel_num_y,self.volume_geometry.voxel_num_x]
        new_shape_ig = [x if x>0 else 1 for x in new_shape_ig]

        data_temp = IM.as_array().reshape(new_shape_ig)

        new_shape_ag = [self.sinogram_geometry.pixel_num_v,self.sinogram_geometry.num_projections,self.sinogram_geometry.pixel_num_h]

        if out is None:
            arr_out = np.zeros(new_shape_ag, dtype=np.float32)
        else:
            arr_out = out.as_array().reshape(new_shape_ag)

        astra.experimental.direct_FP3D(self.projector_id, data_temp, arr_out)

        arr_out = np.squeeze(arr_out)

        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self.sinogram_geometry.copy(), suppress_warning=True)
        else:
            out.fill(arr_out)
        return out
