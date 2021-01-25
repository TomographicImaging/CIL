from cil.optimisation.operators import LinearOperator
from cil.framework.plugins.tigre import CIL2TIGREGeometry
from tigre.utilities import Ax, Atb
import numpy as np

# def Ax(img, geo, angles,  krylov="ray-voxel"):
class ProjectionOperator(LinearOperator):
    '''initial TIGRE Projection Operator

    will only work with perfectly aligned data'''

    def __init__(self, domain_geometry, range_geometry, direct_method='interpolated',adjoint_method='matched'):
    
        super(ProjectionOperator,self).__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)
             
        self.tigre_geom = CIL2TIGREGeometry.getTIGREGeometry(domain_geometry,range_geometry)
        self.method = {'direct':direct_method,'adjoint':adjoint_method}
    
    def direct(self, x, out=None):
        if out is None:
            out = self.range.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=0)
            arr_out = Ax.Ax(data_temp, self.tigre_geom, self.tigre_geom.angles, projection_type='interpolated')
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = Ax.Ax(x.as_array(), self.tigre_geom, self.tigre_geom.angles , projection_type='Siddon')
        out.fill ( arr_out )
        return out
    def adjoint(self, x, out=None):
        if out is None:
            out = self.domain.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=1)
            arr_out = Atb.Atb(data_temp, self.tigre_geom, self.tigre_geom.angles , krylov='matched')
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = Atb.Atb(x.as_array(), self.tigre_geom, self.tigre_geom.angles , krylov='matched')
        out.fill ( arr_out )
        return out
