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

from cil.framework.labels import AcquisitionType, AngleUnit
import numpy as np
import warnings
from scipy.spatial.transform import Rotation

try:
    from tigre.utilities.geometry import Geometry
except ModuleNotFoundError:
    Geometry = object


def calculate_euler_angles(angles, base):
    """Spin the reference orientation 'base' about z by each scan angle; return ZYZ Euler angles."""
    angles = np.asarray(angles, dtype=float)
    spun = Rotation.from_euler('z', angles[:, None]) * Rotation.from_matrix(base)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        return spun.as_euler('ZYZ').astype(np.float32)


class CIL2TIGREGeometry(object):
    """Convert a CIL image and acquisition geometry to a TIGRE geometry and projection angles.

    Advanced geometries (offset centre of rotation, tilted rotation axis) are handled by rotating
    the volume through per-view Euler angles.
    """

    @staticmethod
    def getTIGREGeometry(ig, ag):
        """Return the TIGRE geometry and projection angles for the CIL geometries (ig, ag)."""
        converter = CIL2TIGREGeometry(ig, ag)
        return converter.tg_geometry, converter.tg_angles

    def __init__(self, ig, ag):
        if Geometry is object:
            raise ModuleNotFoundError(
                "This plugin requires the additional package TIGRE\n"
                "Please install it via conda as tigre from the ccpi channel")

        if ag.geom_type not in ['cone', 'parallel']:
            raise ValueError(f"CIL cannot use TIGRE to process geometries of type {ag.geom_type}.")

        # work on a copy of the CIL geometry aligned to the TIGRE frame
        self._ag = ag.copy()
        self._ag.config.system.align_reference_frame('tigre')
        self._ig = ig

        # z-spin between the TIGRE and CIL frames, undone per view in _convert_angles
        self.theta = 0.0

        # the TIGRE geometry to populate and return
        self.tg_geometry = Geometry()
        self.tg_geometry.accuracy = 0.5   # forward projection accuracy (voxels/sample)

        self._scale_geometry()
        self._set_up_tigre_geometry()

    def _scale_geometry(self):
        """Move the CIL detector clear of the volume so TIGRE's interpolated projector doesn't clip the ray."""
        system = self._ag.config.system
        ig = self._ig
        panel = self._ag.config.panel

        lenx = ig.voxel_num_x * ig.voxel_size_x
        leny = ig.voxel_num_y * ig.voxel_size_y
        lenz = ig.voxel_num_z * ig.voxel_size_z
        panel_width = max(panel.num_pixels * panel.pixel_size) * 0.5
        clearance_len = np.sqrt(lenx**2 + leny**2 + lenz**2)/2 + panel_width

        if self._ag.geom_type == 'cone':
            # push the detector out along the source ray, scaling the pixel size to match so
            # magnification leaves the projection identical
            if np.linalg.norm(system.detector.position) < clearance_len:
                src = system.source.position.astype(np.float64)
                vec1 = system.detector.position.astype(np.float64) - src
                src_dist = np.linalg.norm(system.source.position)
                scale = np.ceil((clearance_len + src_dist) / src_dist / self._ag.magnification)
                system.detector.position = src + vec1 * scale
                panel.pixel_size[0] *= scale
                panel.pixel_size[1] *= scale
        else:
            system.detector.position = system.detector.position + system.ray.direction * clearance_len

    def _set_up_tigre_geometry(self):
        """Populate the TIGRE geometry and angles (tg_geometry, tg_angles) from the CIL geometry."""
        self._set_detector()
        self._set_volume()

        if self._ag.geom_type == 'cone':
            self.tg_angles = self._set_geometry_cone()
        else:
            self.tg_angles = self._set_geometry_parallel()

        self._set_panel_origin()

        self.tg_geometry.is2D = bool(AcquisitionType.DIM2 & self._ag.dimension)

    def _set_detector(self):
        """Set the detector panel pixel counts, pixel size and total size, in TIGRE (V, U) order."""
        panel = self._ag.config.panel
        self.tg_geometry.nDetector = np.array(panel.num_pixels[::-1])
        self.tg_geometry.dDetector = np.array(panel.pixel_size[::-1])
        self.tg_geometry.sDetector = self.tg_geometry.dDetector * self.tg_geometry.nDetector

    def _set_volume(self):
        """Set the reconstruction volume voxel counts and sizes, in TIGRE (Z, Y, X) order."""
        ig = self._ig
        self.tg_geometry.nVoxel = np.array([ig.voxel_num_z, ig.voxel_num_y, ig.voxel_num_x])
        self.tg_geometry.dVoxel = np.array([ig.voxel_size_z, ig.voxel_size_y, ig.voxel_size_x])
        if AcquisitionType.DIM2 & self._ag.dimension:
            # collapse z to a single slice matched to the detector pixel size
            self.tg_geometry.nVoxel[0] = 1
            self.tg_geometry.dVoxel[0] = self._ag.config.panel.pixel_size[1] / self._ag.magnification
        self.tg_geometry.sVoxel = self.tg_geometry.nVoxel * self.tg_geometry.dVoxel

    def _off_origin(self):
        """Volume-centre offset in TIGRE (Z, Y, X)"""
        center_z = 0. if AcquisitionType.DIM2 & self._ag.dimension else self._ig.center_z
        return np.array([center_z, self._ig.center_y, self._ig.center_x])

    def _view_vectors_3d(self, beam):
        """Return acquisition geometry vectors as 3D
        """
        detector = self._ag.config.system.detector
        if AcquisitionType.DIM2 & self._ag.dimension:
            return (np.append(beam, 0.), np.append(detector.position, 0.),
                    np.append(detector.direction_x, 0.), np.array([0., 0., 1.]))
        return beam, detector.position, detector.direction_x, detector.direction_y

    def _cone_view_to_tigre(self, S, D, dx, dy):
        """Map one cone-beam view to its TIGRE parameters.

        Given a source 'S', detector centre 'D' and detector axes 'dx', 'dy' return that view's
        (DSO, DSD, offDetector, rotDetector, B), where B = [e0, h, v] is (the source direction, horizontal, vertical).
        """

        e0 = S / np.linalg.norm(S)
        h = np.cross([0., 0., 1.], e0)
        h_norm = np.linalg.norm(h)
        if h_norm < 1e-8:
            # source parallel to z: pick any perpendicular horizontal
            h = np.cross([0., 1., 0.], e0)
            h_norm = np.linalg.norm(h)
        h /= h_norm
        v = np.cross(e0, h)
        B = np.column_stack([e0, h, v])

        # force the detector normal to face the source (its sign depends on panel handedness)
        n = np.cross(dx, dy)
        n *= np.sign((S - D) @ n)
        RD = B.T @ np.column_stack([n, dx, dy])

        DSO = np.linalg.norm(S)
        DSD = float((S - D) @ e0)
        offDetector = np.array([v @ D, h @ D, 0])
        rotDetector = Rotation.from_matrix(RD).as_euler('xyz')
        return DSO, DSD, offDetector, rotDetector, B

    def _set_geometry_cone(self):
        """Set the cone-beam distances, offsets and detector orientation, and return the angles."""
        S, D, dx, dy = self._view_vectors_3d(self._ag.config.system.source.position)

        self.tg_geometry.mode = 'cone'
        self.tg_geometry.offOrigin = self._off_origin()

        DSO, DSD, offDetector, rotDetector, B = self._cone_view_to_tigre(S, D, dx, dy)
        self.tg_geometry.DSO = DSO
        self.tg_geometry.DSD = DSD
        self.tg_geometry.offDetector = offDetector
        self.tg_geometry.rotDetector = rotDetector

        # theta is the z-spin between the TIGRE and CIL frames: the in-plane azimuth of the
        # principal ray D-S in the TIGRE frame, undone per view in _convert_angles
        w = D - S
        self.theta = -np.arctan2(w[0], w[1])

        # each view spins this reference frame about z, giving per-view ZYZ Euler angles
        euler_base = Rotation.from_euler('z', np.pi/2).as_matrix() @ B
        return calculate_euler_angles(self._convert_angles(), euler_base)

    def _set_geometry_parallel(self):
        """Set the parallel-beam distances, offsets and detector orientation, and return the angles."""
        ray, D, dx, dy = self._view_vectors_3d(self._ag.config.system.ray.direction)

        det_dist = D @ ray
        self.tg_geometry.DSO = det_dist
        self.tg_geometry.DSD = 2*det_dist
        self.tg_geometry.mode = 'parallel'
        self.tg_geometry.offOrigin = self._off_origin()

        # theta is the in-plane azimuth of the ray, undone per view in _convert_angles
        self.theta = -np.arctan2(ray[0], ray[1])

        # detector orientation [n, dx, dy]
        e0 = -ray
        h = np.array([1., 0, 0])
        v = np.cross(e0, h)
        B = np.column_stack([e0, h, v])
        n = np.cross(dx, dy)
        RD = B.T @ np.column_stack([n, dx, dy])
        self.tg_geometry.rotDetector = Rotation.from_matrix(RD).as_euler('xyz')

        if self._ag.system_description == 'advanced':
            # tilted rotation axis: offset from the full detector position, and spin the volume
            # per view via the Euler reference frame
            self.tg_geometry.offDetector = np.array([v @ D, h @ D, 0])
            euler_base = Rotation.from_euler('z', np.pi/2).as_matrix() @ B
            return calculate_euler_angles(self._convert_angles(), euler_base)

        # axis-aligned: the lateral detector shift is the component perpendicular to the ray
        det_pos = D - det_dist * ray
        self.tg_geometry.offDetector = np.array([det_pos[2], det_pos[0], 0])
        return self._convert_angles()

    def _set_panel_origin(self):
        """Rotate the panel around it's centre based on the panel origin to reflect the data direction
        """
        panel_origin = self._ag.config.panel.origin

        roll = pitch = yaw = 0.0
        if 'right' in panel_origin and 'top' in panel_origin:
            roll = np.pi
        elif 'right' in panel_origin:
            yaw = np.pi
        elif 'top' in panel_origin:
            pitch = np.pi

        flip = Rotation.from_euler('xyz', [roll, pitch, yaw])
        base = Rotation.from_euler('xyz', self.tg_geometry.rotDetector)
        self.tg_geometry.rotDetector = (base * flip).as_euler('xyz')

    def _convert_angles(self):
        """Convert the CIL scan angles to TIGRE's angle convention, wrapped to (-pi, pi).
        """
        config = self._ag.config.angles
        angles = config.angle_data + config.initial_angle
        if config.angle_unit == AngleUnit.DEGREE:
            angles *= (np.pi/180.)
        angles += np.pi/2 + self.theta
        angles *= -1
        return (angles + np.pi) % (2*np.pi) - np.pi
