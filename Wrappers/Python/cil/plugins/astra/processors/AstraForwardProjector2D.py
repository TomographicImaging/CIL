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
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_2D
import astra
import numpy as np

class AstraForwardProjector2D(DataProcessor):

    """
    AstraForwardProjector2D configures an ASTRA 2D forward projector for CPU and GPU.

    Parameters
    ----------

    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    sinogram_geometry : AcquisitionGeometry
        A description of the acquisition data

    proj_id : the ASTRA projector ID
        For advances ASTRA users only. The astra_mex_projector ID of the projector, use `astra.astra_create_projector()`

    device : string, default='gpu'
        The device to run on 'gpu' or 'cpu'

    """

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

        #DataProcessor.__init__(self, **kwargs)
        super(AstraForwardProjector2D, self).__init__(**kwargs)

        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)

        vol_geom, proj_geom = convert_geometry_to_astra_vec_2D(self.volume_geometry, self.sinogram_geometry)

        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' only one option
            if self.sinogram_geometry.geom_type == 'parallel':
                self.set_projector(astra.create_projector('line', proj_geom, vol_geom) )
            elif self.sinogram_geometry.geom_type == 'cone':
                self.set_projector(astra.create_projector('line_fanflat', proj_geom, vol_geom) )
            else:
                NotImplemented
        elif device == 'gpu':
            self.set_projector(astra.create_projector('cuda', proj_geom, vol_geom) )
        else:
            NotImplemented

    def check_input(self, dataset):
        if dataset.number_of_dimensions == 1 or\
           dataset.number_of_dimensions == 2:
               return True
        else:
            raise ValueError("Expected input dimensions is 1 or 2, got {0}"\
                             .format(dataset.number_of_dimensions))

    def _set_up(self):
        """
        Configure processor attributes that require the data to setup
        Must set _shape_out
        """
        self._shape_out = self.sinogram_geometry.shape
    
    def set_projector(self, proj_id):
        self.proj_id = proj_id

    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry

    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry

    def process(self, out=None):

        IM = self.get_input()

        #ASTRA expects a 2D array with shape 1, CIL removes dimensions of len 1
        new_shape_ig = [self.volume_geometry.voxel_num_y,self.volume_geometry.voxel_num_x]
        new_shape_ig = [x if x>0 else 1 for x in new_shape_ig]

        IM_data_temp = IM.as_array().reshape(new_shape_ig)

        sinogram_id, arr_out = astra.create_sino(IM_data_temp, self.proj_id)
        astra.data2d.delete(sinogram_id)

        arr_out = np.squeeze(arr_out)
        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self.sinogram_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
