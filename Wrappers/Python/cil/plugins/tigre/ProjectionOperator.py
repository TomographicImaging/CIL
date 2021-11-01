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

from cil.framework import ImageData, AcquisitionData, AcquisitionGeometry
from cil.framework import DataOrder
from cil.optimisation.operators import LinearOperator
from cil.plugins.tigre import CIL2TIGREGeometry
import numpy as np

try:
    from tigre.utilities import Ax, Atb
except ModuleNotFoundError:
    raise ModuleNotFoundError("This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel")

class ProjectionOperator(LinearOperator):
    '''TIGRE Projection Operator'''

    def __init__(self, image_geometry, aquisition_geometry, direct_method='interpolated',adjoint_weights='matched'):
        '''
        This class creates a configured TIGRE ProjectionOperator
        
        Please refer to the TIGRE documentation for futher descriptions
        https://github.com/CERN/TIGRE
        https://iopscience.iop.org/article/10.1088/2057-1976/2/5/055010
                        
        :param image_geometry: A description of the ImageGeometry of your data
        :type image_geometry: ImageGeometry
        :param aquisition_geometry: A description of the AcquisitionGeometry of your data
        :type aquisition_geometry: AcquisitionGeometry
        :param direct_method: The method used by the forward projector, 'Siddon' for ray-voxel intersection, 'interpolated' for interpolated projection
        :type direct_method: str, default 'interpolated'
        :param adjoint_weights: The weighting method used by the cone-beam backward projector, 'matched' for weights to approximatly match the 'interpolated' forward projector, 'FDK' for FDK weights, default 'matched'
        :type adjoint_weights: str    
        '''

        DataOrder.check_order_for_engine('tigre', image_geometry)
        DataOrder.check_order_for_engine('tigre', aquisition_geometry) 

        super(ProjectionOperator,self).__init__(domain_geometry=image_geometry,\
             range_geometry=aquisition_geometry)
             
        self.tigre_geom, self.tigre_angles= CIL2TIGREGeometry.getTIGREGeometry(image_geometry,aquisition_geometry)

        if direct_method not in ['interpolated','Siddon']:
            raise ValueError("direct_method expected 'interpolated' or 'Siddon' got {}".format(direct_method))

        if adjoint_weights not in ['matched','FDK']:
            raise ValueError("adjoint_weights expected 'matched' or 'FDK' got {}".format(adjoint_weights))

        self.method = {'direct':direct_method,'adjoint':adjoint_weights}

        #TIGRE bug workaround, when voxelgrid and panel are aligned ray tracing fails
        if direct_method=='Siddon' and aquisition_geometry.geom_type == AcquisitionGeometry.PARALLEL:
            for i, angle in enumerate(self.tigre_angles):
                if abs(angle/np.pi) % 0.5 < 1e-4:
                    self.tigre_angles[i] += 1e-5


    def direct(self, x, out=None):

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=0)
            arr_out = Ax.Ax(data_temp, self.tigre_geom, self.tigre_angles, projection_type=self.method['direct'])
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = Ax.Ax(x.as_array(), self.tigre_geom, self.tigre_angles, projection_type=self.method['direct'])

        if arr_out.shape[0] == 1:
            arr_out = np.squeeze(arr_out, axis=0)

        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self._range_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def adjoint(self, x, out=None):
        #note Atb.Atb optional parameter name and default changed betwen 2.1 and 2.2

        data = x.as_array()
        if x.dimension_labels[0] != AcquisitionGeometry.ANGLE:
            data = np.expand_dims(data,axis=0)

        if self.tigre_geom.is2D:
            data = np.expand_dims(data,axis=1)
            arr_out = Atb.Atb(data, self.tigre_geom, self.tigre_angles, self.method['adjoint'])
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = Atb.Atb(data, self.tigre_geom, self.tigre_angles, self.method['adjoint'])

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self._domain_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def domain_geometry(self):
        return self._domain_geometry
    
    def range_geometry(self):
        return self._range_geometry
        