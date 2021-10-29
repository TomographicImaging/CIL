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

        #angles in range 0->2pi
        angles = np.mod(angles, 2*np.pi)

        return tg, angles

class TIGREGeometry(Geometry):

    def __init__(self, ig, ag):

        Geometry.__init__(self)

        ag_in = ag.copy()
        system = ag_in.config.system
        system.align_reference_frame()

        # number of voxels (vx)
        self.nVoxel = np.array( [ig.voxel_num_z, ig.voxel_num_y, ig.voxel_num_x] )
        # size of each voxel (mm)
        self.dVoxel = np.array( [ig.voxel_size_z, ig.voxel_size_y, ig.voxel_size_x]  )

        # Detector parameters
        # (V,U) number of pixels        (px)
        self.nDetector = np.array(ag_in.config.panel.num_pixels[::-1])
        # size of each pixel            (mm)
        self.dDetector = np.array(ag_in.config.panel.pixel_size[::-1])
        self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)

        if ag_in.geom_type == 'cone':  
            self.mode = 'cone'

            self.DSO = -system.source.position[1]       
            self.DSD = self.DSO + system.detector.position[1]
        
        else:
            if ag_in.system_description == 'advanced':
                raise NotImplementedError ("CIL cannot use TIGRE to process parallel geometries with tilted axes")

            self.mode = 'parallel'
            
            lenx = (ig.voxel_num_x * ig.voxel_size_x)
            leny = (ig.voxel_num_y * ig.voxel_size_y)

            #to avoid clipping the ray the detector must be outside the reconstruction volume
            self.DSO = max(lenx,leny)
            self.DSD = self.DSO*2

        if ag_in.dimension == '2D':
            self.is2D = True

            #fix IG to single slice in z
            self.nVoxel[0]=1
            self.dVoxel[0]= ag_in.config.panel.pixel_size[1] / ag_in.magnification

            # Offsets Tigre (Z, Y, X) == CIL (X, -Y)
            self.offOrigin = np.array( [0, system.rotation_axis.position[0], -system.rotation_axis.position[1]])

            if ag_in.geom_type == 'cone':  
                self.offDetector = np.array( [0, system.detector.position[0]-system.source.position[0], 0 ])
            else:
                self.offDetector = np.array( [0, system.detector.position[0], 0 ]) 

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

            if ag_in.geom_type == 'cone':  
                self.offDetector = np.array( [system.detector.position[2]-system.source.position[2], system.detector.position[0]-system.source.position[0], 0])
            else:
                self.offDetector = np.array( [system.detector.position[2], system.detector.position[0], 0])

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
