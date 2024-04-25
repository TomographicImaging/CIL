#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

from cil.framework import acquisition_labels
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

        if ag.config.angles.angle_unit == acquisition_labels["DEGREE"]:
            angles *= (np.pi/180.)

        #convert CIL to TIGRE angles s
        angles = -(angles + np.pi/2 +tg.theta )

        #angles in range -pi->pi
        for i, a in enumerate(angles):
            while a < -np.pi:
                a += 2 * np.pi
            while a >= np.pi:
                a -= 2 * np.pi
            angles[i] = a

        return tg, angles

class TIGREGeometry(Geometry):

    def __init__(self, ig, ag):

        Geometry.__init__(self)

        ag_in = ag.copy()
        system = ag_in.config.system
        system.align_reference_frame('tigre')


        #TIGRE's interpolation fp must have the detector outside the reconstruction volume otherwise the ray is clipped
        #https://github.com/CERN/TIGRE/issues/353
        lenx = (ig.voxel_num_x * ig.voxel_size_x)
        leny = (ig.voxel_num_y * ig.voxel_size_y)
        lenz = (ig.voxel_num_z * ig.voxel_size_z)

        panel_width = max(ag_in.config.panel.num_pixels * ag_in.config.panel.pixel_size)*0.5
        clearance_len =  np.sqrt(lenx**2 + leny**2 + lenz**2)/2 + panel_width

        if ag_in.geom_type == 'cone':

            if system.detector.position[1] < clearance_len:

                src = system.source.position.astype(np.float64)
                vec1 = system.detector.position.astype(np.float64) - src

                mag_new = (clearance_len - src[1]) /-src[1]
                scale = mag_new / ag_in.magnification
                scale=np.ceil(scale)

                system.detector.position = src + vec1 * scale
                ag_in.config.panel.pixel_size[0] *= scale
                ag_in.config.panel.pixel_size[1] *= scale

            self.DSO = -system.source.position[1]
            self.DSD = self.DSO + system.detector.position[1]
            self.mode = 'cone'

        else:
            if ag_in.system_description == 'advanced':
                raise NotImplementedError ("CIL cannot use TIGRE to process parallel geometries with tilted axes")

            self.DSO = clearance_len
            self.DSD = 2*clearance_len
            self.mode = 'parallel'

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


        if ag_in.dimension == '2D':
            self.is2D = True

            #fix IG to single slice in z
            self.nVoxel[0]=1
            self.dVoxel[0]= ag_in.config.panel.pixel_size[1] / ag_in.magnification

            self.offOrigin = np.array( [0, 0, 0] )

            # Offsets Tigre (Z, Y, X) == CIL (X, -Y)
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

            if ag_in.geom_type == 'cone':
                #TIGRE origin is at a point on the rotate axis that is in a perpendicular plane containing the source
                #As rotation axis is aligned with z this is the x-y plane
                #We have aligned the y axis with the source->rotate axis direction
                self.offOrigin = np.array( [-system.source.position[2], 0, 0])
                self.offDetector = np.array( [system.detector.position[2]-system.source.position[2], system.detector.position[0]-system.source.position[0], 0])
            else:
                self.offOrigin = np.array( [0,0,0] )
                self.offDetector = np.array( [system.detector.position[2], system.detector.position[0], 0])

            #shift origin z to match image geometry
            #this is in CIL reference frames as the TIGRE geometry rotates the reconstruction volume to match our definitions
            self.offOrigin[0] += ig.center_z


            #convert roll, pitch, yaw
            U = system.detector.direction_x[ind] * flip
            V = system.detector.direction_y[ind] * flip

            roll = np.arctan2(-V[1], V[0])
            pitch = np.arcsin(V[2])
            yaw = np.arctan2(-U[2],U[1])

        #shift origin to match image geometry
        self.offOrigin[1] += ig.center_y
        self.offOrigin[2] += ig.center_x

        self.theta = yaw
        panel_origin = ag_in.config.panel.origin
        if 'right' in panel_origin and 'top' in panel_origin:
            roll += np.pi
        elif 'right' in panel_origin:
            yaw += np.pi
        elif 'top' in panel_origin:
            pitch += np.pi

        self.rotDetector = np.array((roll, pitch, yaw))

        # total size of the image       (mm)
        self.sVoxel = self.nVoxel * self.dVoxel

        # Auxiliary
        self.accuracy = 0.5                        # Accuracy of FWD proj          (vx/sample)
