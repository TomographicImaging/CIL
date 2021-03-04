from cil.framework import AcquisitionGeometry, ImageGeometry
import numpy as np

try:
    from tigre.utilities.geometry import Geometry
except ModuleNotFoundError:
    raise ModuleNotFoundError("This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel")

class CIL2TIGREGeometry(object):
    @staticmethod
    def getTIGREGeometry(ig, ag):
        tg = TIGREGeometry(ig, ag)

        #angles
        angles = ag.config.angles.angle_data + ag.config.angles.initial_angle

        if ag.config.angles.angle_unit == AcquisitionGeometry.DEGREE:
            angles *= (np.pi/180.) 

        angles *= -1 #negate rotation
        angles -= np.pi/2 #rotate imagegeometry 90deg
        angles -= tg.theta #compensate for image geometry definitions

        return tg, angles

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
            raise NotImplementedError("TIGRE only wrapped for cone-beam tomography in CIL")


        self.DSD = DSD
        self.DSO = DSO
       
        # Detector parameters
        # (V,U) number of pixels        (px)
        self.nDetector = np.array(ag_in.config.panel.num_pixels[::-1])
        # size of each pixel            (mm)
        self.dDetector = np.array(ag_in.config.panel.pixel_size[::-1])
        self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)

        # number of voxels              (vx)
        self.nVoxel = np.array( [ig.voxel_num_z, ig.voxel_num_y, ig.voxel_num_x] )
        # size of each voxel            (mm)
        self.dVoxel = np.array( [ig.voxel_size_z, ig.voxel_size_y, ig.voxel_size_x]  )


        if ag_in.dimension == '2D':
            #fix IG
            self.nVoxel[0]=1
            self.dVoxel[0]= ag_in.config.panel.pixel_size[1] / ag_in.magnification

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

        