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

        ag_in = ag.copy()
        system = ag_in.config.system
        system.update_reference_frame()

        #rotate and translate system to match tigre definitions
        if ag_in.geom_type == 'cone':
            if ag_in.dimension=='3D':    
                z = system.source.position[2]
                translate = np.asarray([0,0,-z])
                system.source.position += translate
                system.detector.position += translate
                system.rotation_axis.position += translate

                #align source along negative y
                a = system.source.position
                a = a / np.sqrt(a.dot(a))
                b = np.array([0.0, -1.0, 0.0])

                if np.allclose(a,b):
                    axis_rotation = np.eye(3)
                elif np.allclose(a,-b):
                    axis_rotation = np.eye(3) #pi rotation around either axis
                    axis_rotation[0][0] = -1
                    axis_rotation[2][2] = -1
                else:
                    v = np.cross(a, b)
                    s = np.linalg.norm(v)
                    c = np.dot(a, b) 
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) 
                    axis_rotation = np.eye(3) + vx + vx.dot(vx) * (1-c)/(s**2) 

                r = np.matrix(axis_rotation)
                system.source.position = r.dot(system.source.position.reshape(3,1))
                system.detector.position = r.dot(system.detector.position.reshape(3,1))
                system.rotation_axis.position = r.dot(system.rotation_axis.position.reshape(3,1))
                new_x = r.dot(system.detector.direction_x.reshape(3,1)) 
                new_y = r.dot(system.detector.direction_y.reshape(3,1))
                system.detector.set_direction(new_x, new_y)
            else:    
                #align source along negative y
                a = system.source.position
                a = a / np.sqrt(a.dot(a))
                b = np.array([0.0, -1.0])

                if np.allclose(a,b):
                    axis_rotation = np.eye(2)
                elif np.allclose(a,-b):
                    axis_rotation = np.eye(2)
                    axis_rotation[0][0] = -1
                else:
                    theta = np.arctan2(a[0], a[1]) + np.pi
                    axis_rotation = np.eye(2)
                    axis_rotation[0][0] = axis_rotation[1][1] = np.cos(theta)
                    axis_rotation[0][1] = -np.sin(theta)
                    axis_rotation[1][0] = np.sin(theta)   

                r = np.matrix(axis_rotation)
                system.source.position = r.dot(system.source.position.reshape(2,1))
                system.detector.position = r.dot(system.detector.position.reshape(2,1))
                system.rotation_axis.position = r.dot(system.rotation_axis.position.reshape(2,1))
                system.detector.direction_x = r.dot(system.detector.direction_x.reshape(2,1)) 


            #distance source to origin
            DSO = -system.source.position[1]
            DSD = DSO + system.detector.position[1]

        else:
            #can reconstruct simple parallel geometry only
            #Cofr?
            raise NotImplementedError("Currently TIGRE only wrapped for cone-beam tomography")


        self.DSD = DSD
        self.DSO = DSO
       
        # Detector parameters
        # (V,U) number of pixels        (px)
        self.nDetector = np.array(ag_in.config.panel.num_pixels[::-1])
        # size of each pixel            (mm)
        self.dDetector = np.array(ag_in.config.panel.pixel_size[::-1])
        self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)

        # number of voxels              (vx)
        self.nVoxel = np.array( [ig.voxel_num_z, ig.voxel_num_x, ig.voxel_num_y] )
        self.nVoxel[self.nVoxel==0] = 1 #default is 1 not 0
        # size of each voxel            (mm)
        self.dVoxel = np.array( [ig.voxel_size_z, ig.voxel_size_x, ig.voxel_size_y]  )

        if ag_in.dimension == '2D':
            self.is2D = True
            # Offsets Tigre (Z, Y, X) == CIL (X, -Y)
            self.offOrigin = np.array( [0, system.rotation_axis.position[0], -system.rotation_axis.position[1]])
            self.offDetector = np.array( [0, system.detector.position[0] , 0 ]) #y component in DSD
            
            #convert roll, pitch, yaw
            U = [0, system.detector.direction_x[0], -system.detector.direction_x[1]]
            roll = 0
            pitch = 0
            yaw = np.arctan2(-U[2],U[1])

        else:
            self.is2D = False
            # Offsets Tigre (Z, Y, X) == CIL (Z, X, -Y)        
            ind = np.asarray([2, 0, 1])
            flip = np.asarray([1, 1, -1])

            self.offOrigin = np.array( system.rotation_axis.position[ind] * flip )
            self.offDetector = np.array( [system.detector.position[2], system.detector.position[0], 0]) #y component in DSD
            
            #convert roll, pitch, yaw
            U = system.detector.direction_x[ind] * flip
            V = system.detector.direction_y[ind] * flip

            roll = np.arctan2(-V[1], V[0])
            pitch = np.arcsin(V[2])
            yaw = np.arctan2(-U[2],U[1])
 
        self.theta = yaw
        panel_origin = ag_in.config.panel.origin
        if 'right' in panel_origin:
            yaw += np.pi
        if 'top' in panel_origin:
            pitch += np.pi

        self.rotDetector = np.array((roll, pitch, yaw)) 

        # total size of the image       (mm)
        self.sVoxel = self.nVoxel * self.dVoxel                                         

        # Auxiliary
        self.accuracy = 0.5                        # Accuracy of FWD proj          (vx/sample)
        # Mode
        # parallel, cone
        self.mode = ag_in.config.system.geometry                                  

# def Ax(img, geo, angles,  krylov="ray-voxel"):
class ProjectionOperator(LinearOperator):
    '''initial TIGRE Projection Operator

    will only work with perfectly aligned data'''

    def __init__(self, domain_geometry, range_geometry, direct_method='interpolated',adjoint_method='matched'):
    
        super(ProjectionOperator,self).__init__(domain_geometry=domain_geometry,\
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
        
        self.angles = range_geometry.config.angles.angle_data.copy() + range_geometry.config.angles.initial_angle

        if range_geometry.config.angles.angle_unit == AcquisitionGeometry.DEGREE:
            self.angles *= (np.pi/180.) 

        self.angles *= -1 #negate rotation
        self.angles -= np.pi/2 #rotate imagegeometry 90deg
        self.angles -= self.tigre_geom.theta #compensate for image geometry definitions

        self.method = {'direct':direct_method,'adjoint':adjoint_method}
    
    def direct(self, x, out=None):
        if out is None:
            out = self.range.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=0)
            arr_out = Ax.Ax(data_temp, self.tigre_geom, self.angles, projection_type='interpolated')
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = Ax.Ax(x.as_array(), self.tigre_geom, self.angles , projection_type='Siddon')
        out.fill ( arr_out )
        return out
    def adjoint(self, x, out=None):
        if out is None:
            out = self.domain.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=1)
            arr_out = Atb.Atb(data_temp, self.tigre_geom, self.angles , krylov='matched')
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = Atb.Atb(x.as_array(), self.tigre_geom, self.angles , krylov='matched')
        out.fill ( arr_out )
        return out
    def fdk(self, x, out=None):
        if out is None:
            out = self.domain.allocate(None)
        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(x.as_array(),axis=1)
            arr_out = fdk(data_temp, self.tigre_geom, self.angles)
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = fdk(x.as_array(), self.tigre_geom, self.angles)
        out.fill ( arr_out )
        return out