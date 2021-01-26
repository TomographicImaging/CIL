from cil.framework import ImageData, AcquisitionData
from cil.optimisation.operators import LinearOperator
from cil.plugins.tigre import CIL2TIGREGeometry
from tigre.utilities import Ax, Atb
import numpy as np

# def Ax(img, geo, angles,  krylov="ray-voxel"):
class ProjectionOperator(LinearOperator):
    '''initial TIGRE Projection Operator

    will only work with perfectly aligned data'''

    def __init__(self, image_geometry, aquisition_geometry, direct_method='interpolated',adjoint_method='matched'):
    
        super(ProjectionOperator,self).__init__(domain_geometry=image_geometry,\
             range_geometry=aquisition_geometry)
             
        self.tigre_geom = CIL2TIGREGeometry.getTIGREGeometry(image_geometry,aquisition_geometry)

        self.method = {'direct':direct_method,'adjoint':adjoint_method}
    
    def direct(self, x, out=None):

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=0)
            arr_out = Ax.Ax(data_temp, self.tigre_geom, self.tigre_geom.angles, krylov=self.method['direct'])
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = Ax.Ax(x.as_array(), self.tigre_geom, self.tigre_geom.angles, krylov=self.method['direct'])

        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self._range_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def adjoint(self, x, out=None):

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=1)
            arr_out = Atb.Atb(data_temp, self.tigre_geom, self.tigre_geom.angles, krylov=self.method['adjoint'])
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = Atb.Atb(x.as_array(), self.tigre_geom, self.tigre_geom.angles, krylov=self.method['adjoint'])

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self._domain_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def domain_geometry(self):
        return self._domain_geometry
    
    def range_geometry(self):
        return self._range_geometry 