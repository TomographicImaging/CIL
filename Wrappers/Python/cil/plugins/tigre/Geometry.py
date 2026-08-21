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
    """Spin a reference orientation about z by each scan angle and return ZYZ Euler angles.

    Parameters
    ----------
    angles : array_like
        The scan angles, in radians.
    base : numpy.ndarray
        The 3x3 reference orientation each view spins about z.

    Returns
    -------
    numpy.ndarray
        An (N, 3) array of ZYZ Euler angles, one triple per scan angle.
    """
    angles = np.asarray(angles, dtype=float)
    spun = Rotation.from_euler('z', angles[:, None]) * Rotation.from_matrix(base)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        return spun.as_euler('ZYZ').astype(np.float32)


class CIL2TIGREGeometry(object):
    """Convert a CIL image and acquisition geometry into a TIGRE geometry and projection angles.

    TIGRE describes the acquisition with a ``tigre`` ``Geometry`` object (source/detector
    distances, panel, volume and offsets). CIL's advanced geometries (offset centre of rotation,
    tilted rotation axis / laminography, and combinations) are mapped by rotating the object
    through per-view Euler angles rather than tilting the detector. That mapping needs working
    state - the reference orientation each view spins about z (``_euler_base``) and the z-spin
    between the TIGRE and CIL frames (``theta``) - that is not part of the TIGRE geometry.

    This converter keeps that working state to itself and populates a plain TIGRE ``Geometry`` by
    composition, exposing only the finished products: the TIGRE ``tg_geometry`` and the ``angles``.

    Parameters
    ----------
    ig : ImageGeometry
        A description of the volume to reconstruct.
    ag : AcquisitionGeometry
        A description of the acquisition data.
    """

    @staticmethod
    def getTIGREGeometry(ig, ag):
        """Build the TIGRE geometry and projection angles for a CIL dataset.

        Parameters
        ----------
        ig : ImageGeometry
            A description of the volume to reconstruct.
        ag : AcquisitionGeometry
            A description of the acquisition data.

        Returns
        -------
        tigre.utilities.geometry.Geometry
            The TIGRE geometry describing the system.
        numpy.ndarray
            The projection angles: a 1D array of scan angles, or an (N, 3) array of
            ZYZ Euler angles for advanced geometries.
        """
        converter = CIL2TIGREGeometry(ig, ag)
        return converter.tg_geometry, converter.angles

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
        self.is2D = bool(AcquisitionType.DIM2 & self._ag.dimension)

        # working state for the per-view Euler path, not part of the TIGRE geometry
        self.theta = 0.0
        self._euler_base = None

        # the TIGRE geometry to populate and return
        self.tg_geometry = Geometry()
        self.tg_geometry.accuracy = 0.5   # forward projection accuracy (voxels/sample)

        self._scale_geometry()
        self._set_up_tigre_geometry()

        # ProjectionOperator and FBP branch on this to switch between the 2D and 3D projectors
        self.tg_geometry.is2D = self.is2D

        self.angles = self._convert_angles()

    def _scale_geometry(self):
        """Move the CIL detector clear of the volume without changing the projection.

        TIGRE's interpolated forward projector clips the ray if the detector sits inside the
        reconstruction volume, so this mutates the (already TIGRE-aligned) CIL geometry to keep
        the detector clear of it.
        """
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
        """Populate the TIGRE geometry (distances, volume, detector, offsets) from the CIL geometry."""
        self._set_distances()
        self._set_volume()
        self._set_detector()
        if self._ag.geom_type == 'parallel':
            self._set_offsets_parallel()

        self._set_geometry_advanced()
        self._set_panel_origin()

    def _set_distances(self):
        """Set the source-origin (DSO) and source-detector (DSD) distances and the beam mode."""
        system = self._ag.config.system

        if self._ag.geom_type == 'cone':
            self.tg_geometry.DSO = -system.source.position[1]
            self.tg_geometry.DSD = self.tg_geometry.DSO + system.detector.position[1]
            self.tg_geometry.mode = 'cone'
        else:
            det_dist = system.detector.position @ system.ray.direction
            self.tg_geometry.DSO = det_dist
            self.tg_geometry.DSD = 2*det_dist
            self.tg_geometry.mode = 'parallel'

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
        if self.is2D:
            # collapse z to a single slice matched to the detector pixel size
            self.tg_geometry.nVoxel[0] = 1
            self.tg_geometry.dVoxel[0] = self._ag.config.panel.pixel_size[1] / self._ag.magnification
        self.tg_geometry.sVoxel = self.tg_geometry.nVoxel * self.tg_geometry.dVoxel

    def _set_offsets_parallel(self):
        """Set the volume/detector offsets and detector orientation for parallel geometries.

        TIGRE offsets are in (Z, Y, X) order, which maps to CIL (Z, X, -Y). The detector's
        orientation - including reflections from a negated detector_direction_x/_y - is written to
        rotDetector as a single-axis data mirror, matching the convention in _set_panel_origin,
        while theta carries only the in-plane ray azimuth. Splitting them this way stops a
        reflection being double-applied (once as a mirror and again as a spurious scan rotation).
        """
        system = self._ag.config.system
        ig = self._ig

        ray = system.ray.direction
        det_pos = system.detector.position - (system.detector.position @ ray) * ray

        if self.is2D:
            self.tg_geometry.offOrigin = np.array([0, 0, 0])
            self.tg_geometry.offDetector = np.array([0, det_pos[0], 0])

            dx = np.append(system.detector.direction_x, 0.)
            dy = np.array([0., 0., 1.])
            ray = np.append(ray, 0.)
        else:
            self.tg_geometry.offOrigin = np.array([ig.center_z, 0, 0])
            self.tg_geometry.offDetector = np.array([det_pos[2], det_pos[0], 0])

            dx = system.detector.direction_x
            dy = system.detector.direction_y

        self.tg_geometry.offOrigin[1] += ig.center_y
        self.tg_geometry.offOrigin[2] += ig.center_x

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

    def _set_geometry_advanced(self):
        """Set up the per-view Euler-angle path for cone and advanced parallel geometries.
        """
        system = self._ag.config.system

        if self._ag.geom_type == 'cone':
            if self.is2D:
                # promote the in-plane geometry to 3D; the vertical detector axis is out-of-plane
                S = np.append(system.source.position, 0.)
                D = np.append(system.detector.position, 0.)
                dx = np.append(system.detector.direction_x, 0.)
                dy = np.array([0., 0., 1.])
            else:
                S = system.source.position
                D = system.detector.position
                dx = system.detector.direction_x
                dy = system.detector.direction_y

            # theta is the z-spin between the TIGRE and CIL frames: the in-plane azimuth of the
            # principal ray D-S in the TIGRE frame, undone per view in _convert_angles
            w = D - S
            self.theta = -np.arctan2(w[0], w[1])

            # force the detector normal to face the source (its sign depends on panel handedness)
            n = np.cross(dx, dy)
            n *= np.sign((S - D) @ n)

            # canonical detector frame: e0 the source direction, h horizontal, v vertical.
            # align('tigre') puts the source in-plane on -y (Sx == 0) so h = normalize(z x e0) is x.
            e0 = S / np.linalg.norm(S)
            h = np.array([1., 0, 0])
            v = np.cross(e0, h)
            B = np.column_stack([e0, h, v])

            self.tg_geometry.DSO = np.linalg.norm(S)
            self.tg_geometry.DSD = float((S - D) @ e0)

            center_z = 0. if self.is2D else self._ig.center_z
            self.tg_geometry.offOrigin = np.array([center_z, self._ig.center_y, self._ig.center_x])
            self.tg_geometry.offDetector = np.array([v @ D, h @ D, 0])

            # static panel tilt relative to the canonical frame
            RD = B.T @ np.column_stack([n, dx, dy])
            self.tg_geometry.rotDetector = Rotation.from_matrix(RD).as_euler('xyz')
            self._euler_base = Rotation.from_euler('z', np.pi/2).as_matrix() @ B

        elif self._ag.system_description == 'advanced':

            D = system.detector.position
            dx = system.detector.direction_x
            dy = system.detector.direction_y
            ray = system.ray.direction

            self.theta = -np.arctan2(ray[0], ray[1])
            e0 = -ray
            h = np.array([1., 0, 0])
            v = np.cross(e0, h)
            B = np.column_stack([e0, h, v])

            self.tg_geometry.offOrigin = np.array([self._ig.center_z, self._ig.center_y, self._ig.center_x])
            self.tg_geometry.offDetector = np.array([v @ D, h @ D, 0])

            # detector orientation relative to the canonical frame
            n = np.cross(dx, dy)
            RD = B.T @ np.column_stack([n, dx, dy])
            self.tg_geometry.rotDetector = Rotation.from_matrix(RD).as_euler('xyz')
            self._euler_base = Rotation.from_euler('z', np.pi/2).as_matrix() @ B

    def _set_panel_origin(self):
        """Rotate the panel around it's centre based on the panel origin and data direction
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
        """Convert the CIL scan angles to TIGRE projection angles.

        Returns
        -------
        numpy.ndarray
            The scan angles wrapped to [-pi, pi) as a 1D array; or, for advanced geometries,
            an (N, 3) array of ZYZ Euler angles carrying the axis tilt.
        """
        config = self._ag.config.angles
        angles = config.angle_data + config.initial_angle
        if config.angle_unit == AngleUnit.DEGREE:
            angles *= (np.pi/180.)
        angles += np.pi/2 + self.theta
        angles *= -1
        angles = (angles + np.pi) % (2*np.pi) - np.pi

        if self._euler_base is not None:
            return calculate_euler_angles(angles, self._euler_base)
        return angles
