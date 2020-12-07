import numpy as np
from tigre.utilities.geometry import Geometry
from tigre.utilities import Ax, Atb
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.optimisation.operators import LinearOperator
from tigre.algorithms import fdk

class CIL2TIGREGeometry(object):
    @staticmethod
    def getTIGREGeometry(ig, ag):
        tg = TIGREGeometry(ig, ag)
        tg.check_geo(angles=ag.config.angles.angle_data)
        return tg

class TIGREGeometry(Geometry):

    def __init__(self, ig, ag):

        Geometry.__init__(self)
        
        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = ag.dist_source_center + ag.dist_center_detector
        self.DSO = ag.dist_source_center
       
        # Detector parameters
        # (V,U) number of pixels        (px)
        self.nDetector = np.array(ag.config.panel.num_pixels[::-1])
        # size of each pixel            (mm)
        self.dDetector = np.array(ag.config.panel.pixel_size[::-1])
        self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)
        
        # Image parameters
        if ag.dimension == '2D':
            # number of voxels              (vx)
            self.nVoxel = np.array( [1, ig.voxel_num_x, ig.voxel_num_y] )
            # size of each voxel            (mm)
            self.dVoxel = np.array( [ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_x]  )
            self.is2D = True
        else:
            # number of voxels              (vx)
            self.nVoxel = np.array( [ig.voxel_num_z, ig.voxel_num_x, ig.voxel_num_y] )
            # size of each voxel            (mm)
            self.dVoxel = np.array( [ig.voxel_size_z, ig.voxel_size_y, ig.voxel_size_x]  )
            self.is2D = False
        
        # total size of the image       (mm)
        self.sVoxel = self.nVoxel * self.dVoxel                                         
        
        # Offsets
        self.offOrigin = np.array( [0, 0, 0 ])
        self.offDetector = np.array( [0 , 0, 0 ])                 # Offset of Detector            (mm)
        self.rotDetector = np.array((0, 0, 0))
        # Auxiliary
        self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
        # Mode
        # parallel, cone
        self.mode = ag.config.system.geometry                                  

# def Ax(img, geo, angles,  krylov="ray-voxel"):
class ProjectionOperator(LinearOperator):
    '''initial TIGRE Projection Operator

    will only work with perfectly aligned data'''

    def __init__(self, domain_geometry, range_geometry, direct_method='interpolated',adjoint_method='matched'):
    
        super(TIGREProjectionOperator,self).__init__(domain_geometry=domain_geometry,\
             range_geometry=range_geometry)
        self.tigre_geom = CIL2TIGREGeometry.getTIGREGeometry(domain_geometry,range_geometry)


        # beam goes on y axis in our geometry (for simple geometry)
        #                 o     Y          |   -X,Y
        # Xray --------O-------->----------|
        #              |                   |
        #              |
        #              V  X
        # rotation is around axis Z: out of plane right handed (anti-clockwise)
        # TIGRE seems to have, with right handed reference system
        #                 o                |   (-Y_CIL,+X_CIL)
        # Xray -<------O-------------------|
        #              |                   |
        #              |
        #              V  Y

        # so we need to translate the angles by removing 90 degrees
        
        self.angles = range_geometry.config.angles.angle_data.copy()

        if range_geometry.config.angles.angle_unit == AcquisitionGeometry.DEGREE:
            self.angles *= (np.pi/180.) 
        self.angles -= np.pi/2
        
        self.method = {'direct':direct_method,'adjoint':adjoint_method}
    
    def direct(self, x, out=None):
        if out is None:
            out = self.range.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=0)
            arr_out = Ax.Ax(data_temp, self.tigre_geom, self.angles , krylov=self.method['direct'])
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = Ax.Ax(x.as_array(), self.tigre_geom, self.angles , krylov=self.method['direct'])
        out.fill ( arr_out )
        return out
    def adjoint(self, x, out=None):
        if out is None:
            out = self.domain.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=1)
            arr_out = Atb.Atb(data_temp, self.tigre_geom, self.angles , krylov=self.method['adjoint'])
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = Atb.Atb(x.as_array(), self.tigre_geom, self.angles , krylov=self.method['adjoint'])
        out.fill ( arr_out )
        return out
    def fdk(self, x):
        data_temp = np.expand_dims(x.as_array(),axis=1)
        arr_out = fdk(data_temp, self.tigre_geom, self.angles)
        arr_out = np.squeeze(arr_out, axis=0)
        return arr_out
        