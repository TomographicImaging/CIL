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

from cil.framework import ImageData, AcquisitionData
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

    def __init__(self, image_geometry, aquisition_geometry, direct_method='Siddon',adjoint_method='matched'):
    
        DataOrder.check_order_for_engine('tigre', image_geometry)
        DataOrder.check_order_for_engine('tigre', aquisition_geometry) 

        super(ProjectionOperator,self).__init__(domain_geometry=image_geometry,\
             range_geometry=aquisition_geometry)
             
        self.tigre_geom, self.tigre_angles= CIL2TIGREGeometry.getTIGREGeometry(image_geometry,aquisition_geometry)

        self.method = {'direct':direct_method,'adjoint':adjoint_method}
    
    def direct(self, x, out=None):

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=0)
            arr_out = Ax.Ax(data_temp, self.tigre_geom, self.tigre_angles, projection_type=self.method['direct'])
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = Ax.Ax(x.as_array(), self.tigre_geom, self.tigre_angles, projection_type=self.method['direct'])

        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self._range_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def adjoint(self, x, out=None):

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=1)
            arr_out = Atb.Atb(data_temp, self.tigre_geom, self.tigre_angles, krylov=self.method['adjoint'])
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = Atb.Atb(x.as_array(), self.tigre_geom, self.tigre_angles, krylov=self.method['adjoint'])

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self._domain_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def domain_geometry(self):
        return self._domain_geometry
    
    def range_geometry(self):
        return self._range_geometry
        