from cil.optimisation.operators import LinearOperator
from cil.framework import AcquisitionData, ImageData

import itk
from itk import RTK as rtk
import numpy as np

ImageType = itk.Image[itk.F, 3]

def build_rtk_geometry_from_cil_ag(ag):
    """
    Build RTK ThreeDCircularProjectionGeometry from a CIL cone-beam AcquisitionGeometry.
    """
    # CIL geometry atts
    src = np.asarray(ag.config.system.source.position, dtype=float)      
    det = np.asarray(ag.config.system.detector.position, dtype=float)    
    ang_rad = np.asarray(ag.angles, dtype=float)

    sid = float(np.linalg.norm(src - np.array([0.0, 0.0, 0.0])))
    sdd = float(np.linalg.norm(det - src))

    geom = rtk.ThreeDCircularProjectionGeometry.New()
    for a in np.rad2deg(ang_rad):  
        geom.AddProjection(sid, sdd, float(a))

    return geom, sid, sdd

def cil_acquisitiondata_to_rtk_projection_image(data3D, apply_minus_log=True, eps=1e-6):
    """
    Convert CIL AcquisitionData (angle, vertical, horizontal) -> ITK image (u, v, angle) 
    """
    ag = data3D.geometry

    ##TODO need to flip check?
    projs = np.asarray(data3D.array[::-1], dtype=np.float32)

    if apply_minus_log:
        projs = -np.log(np.clip(projs, eps, None)).astype(np.float32, copy=False)


    proj_itk = itk.image_from_array(projs)

    du, dv = map(float, ag.config.panel.pixel_size)
    det_u, det_v = map(int, ag.config.panel.num_pixels)
    n_proj = int(projs.shape[0])

    proj_itk.SetSpacing([du, dv, 1.0])

    proj_itk.SetOrigin([
        -(det_u - 1) * du / 2.0,
        -(det_v - 1) * dv / 2.0,
        0.0
    ])

    return proj_itk, projs




def make_constant_image_source(size, spacing, origin, constant=0.0):
    """
    Return an RTK ConstantImageSource producing a 3D image.
    """
    src = rtk.ConstantImageSource[ImageType].New()
    src.SetSize([int(size[0]), int(size[1]), int(size[2])])
    src.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])
    src.SetOrigin([float(origin[0]), float(origin[1]), float(origin[2])])
    src.SetConstant(float(constant))
    return src


def make_constant_image(size, spacing, origin, constant=0.0):
    """
    Convenience wrapper that returns the image output directly.
    """
    src = make_constant_image_source(size, spacing, origin, constant)
    src.Update()
    return src.GetOutput()

def apply_fov_mask(recon_img, projections_img, geometry):
    FOVFilterType = rtk.FieldOfViewImageFilter[ImageType, ImageType]
    fov = FOVFilterType.New()
    fov.SetInput(0, recon_img)
    fov.SetProjectionsStack(projections_img)
    fov.SetGeometry(geometry)
    fov.Update()
    return fov.GetOutput()


class ProjectionOperator(LinearOperator):
    """
    ### FIX ###
    ## TODO more tests with zeng
    ## TODO test with GPU?
    
    RTKProjectionOperator (CPU) for CIL <-> RTK.

    Uses a matched projector pair:
      - "joseph" (recommended default)
      - "zeng" (this iswith mismatch adjoint)


    Parameters
    ----------
    image_geometry : CIL ImageGeometry
        Domain geometry (volume).
    acquisition_geometry : CIL AcquisitionGeometry
        Range geometry (projections).
    projector : str, default="joseph"
        "joseph" or "zeng".
    n_work_units : int or None, default=None
        Per-filter ITK work-units hint (if supported by your wrapped RTK/ITK class).
    rtk_geometry : RTK ThreeDCircularProjectionGeometry, optional
        If provided, use this exact RTK geometry instead of rebuilding from CIL.
        Recommended when you already have a working FDK / adjoint test with RTK.
    projection_template_itk : itk.Image[itk.F,3], optional
        If provided, projection stack metadata (size/spacing/origin) is taken from this.
    flip_u : bool, default=False
        Flip detector horizontal axis when converting CIL AcquisitionData <-> ITK.
    flip_v : bool, default=False
        Flip detector vertical axis when converting CIL AcquisitionData <-> ITK.
    reverse_angles : bool, default=False
        Reverse projection order (angle axis) when converting CIL AcquisitionData <-> ITK.
        Useful if angle ordering convention differs.
    """

    def __init__(
        self,
        image_geometry,
        acquisition_geometry,
        projector="joseph",
        n_work_units=None,
        rtk_geometry=None,
        projection_template_itk=None,
        flip_u=False,
        flip_v=False,
        reverse_angles=False,
    ):
        super().__init__(domain_geometry=image_geometry, range_geometry=acquisition_geometry)

        self.volume_geometry = image_geometry
        self.sinogram_geometry = acquisition_geometry
        self.projector = str(projector).lower()
        self.n_work_units = None if n_work_units is None else int(n_work_units)

        self.flip_u = bool(flip_u)
        self.flip_v = bool(flip_v)
        self.reverse_angles = bool(reverse_angles)

        self.ImageType = itk.Image[itk.F, 3]

        if rtk_geometry is not None:
            self.rtk_geometry = rtk_geometry
        else:
            self.rtk_geometry = self._build_rtk_geometry_from_cil(self.sinogram_geometry)

        if self.projector == "joseph":
            self.FPType = rtk.JosephForwardProjectionImageFilter[self.ImageType, self.ImageType]
            self.BPType = rtk.JosephBackProjectionImageFilter[self.ImageType, self.ImageType]
        elif self.projector == "zeng":
            self.FPType = rtk.ZengForwardProjectionImageFilter[self.ImageType, self.ImageType]
            self.BPType = rtk.ZengBackProjectionImageFilter[self.ImageType, self.ImageType]
        else:
            raise ValueError("Unsupported projector. Use 'joseph' or 'zeng'.")

        self._vol_size, self._vol_spacing, self._vol_origin = self._cil_image_geometry_to_rtk_grid(
            self.volume_geometry
        )

        if projection_template_itk is not None:
            self._proj_size, self._proj_spacing, self._proj_origin = self._itk_proj_metadata(
                projection_template_itk
            )
        else:
            self._proj_size, self._proj_spacing, self._proj_origin = self._cil_acq_geometry_to_rtk_proj_grid(
                self.sinogram_geometry
            )

    def direct(self, x, out=None):
        """        
        Parameters
        ----------
        x : ImageData
            CIL ImageData volume.
        out : AcquisitionData, optional
            Fill this output container in-place.

        Returns
        -------
        AcquisitionData 
        """
        
        if not isinstance(x, ImageData):
            raise TypeError(f"direct expects ImageData, got {type(x)}")

        vol_itk = self._cil_image_to_itk(x)

        proj_template = self._make_constant_image(
            size=self._proj_size,
            spacing=self._proj_spacing,
            origin=self._proj_origin,
            constant=0.0,
        )

        fp = self.FPType.New()
        fp.SetGeometry(self.rtk_geometry)
        self._set_work_units_if_available(fp)

        fp.SetInput(proj_template)
        fp.SetInput(1, vol_itk)
        fp.Update()

        proj_itk = fp.GetOutput()
        return self._itk_to_cil_acquisition(proj_itk, out=out)

    def adjoint(self, x, out=None):
        """

        Parameters
        ----------
        x : AcquisitionData
            CIL AcquisitionData projections.
        out : ImageData, optional
            Fill this output container in-place.

        Returns
        -------
        ImageData 
        """
        if not isinstance(x, AcquisitionData):
            raise TypeError(f"adjoint expects AcquisitionData, got {type(x)}")

        proj_itk = self._cil_acquisition_to_itk(x)

        vol_template = self._make_constant_image(
            size=self._vol_size,
            spacing=self._vol_spacing,
            origin=self._vol_origin,
            constant=0.0,
        )

        bp = self.BPType.New()
        bp.SetGeometry(self.rtk_geometry)
        self._set_work_units_if_available(bp)

        bp.SetInput(vol_template)
        bp.SetInput(1, proj_itk)
        bp.Update()

        vol_itk = bp.GetOutput()
        return self._itk_to_cil_image(vol_itk, out=out)


    def _set_work_units_if_available(self, flt):
        # by default all cpu cores are used
        if self.n_work_units is not None and hasattr(flt, "SetNumberOfWorkUnits"):
            flt.SetNumberOfWorkUnits(int(self.n_work_units))

    def _build_rtk_geometry_from_cil(self, ag):
        """
        Convert CIL circular cone-beam geometry to RTK ThreeDCircularProjectionGeometry.
        """
        src = np.asarray(ag.config.system.source.position, dtype=float)
        det = np.asarray(ag.config.system.detector.position, dtype=float)
        rot = np.asarray(ag.config.system.rotation_axis.position, dtype=float)

        sid = float(np.linalg.norm(src - rot))  
        sdd = float(np.linalg.norm(det - src))   

        angles = np.asarray(ag.angles, dtype=float)
        if angles.size == 0:
            raise ValueError("Acquisition geometry has no angles.")

        angles_deg = np.rad2deg(angles) if np.max(np.abs(angles)) <= (2*np.pi + 1e-2) else angles

        geom = rtk.ThreeDCircularProjectionGeometry.New()
        for a in angles_deg:
            geom.AddProjection(sid, sdd, float(a))
        return geom

    def _cil_image_geometry_to_rtk_grid(self, ig):
        nx = int(ig.voxel_num_x)
        ny = int(ig.voxel_num_y)
        nz = int(ig.voxel_num_z)

        sx = float(ig.voxel_size_x)
        sy = float(ig.voxel_size_y)
        sz = float(ig.voxel_size_z)

        cx = float(getattr(ig, "center_x", 0.0))
        cy = float(getattr(ig, "center_y", 0.0))
        cz = float(getattr(ig, "center_z", 0.0))

        ox = cx - (nx - 1) * sx / 2.0
        oy = cy - (ny - 1) * sy / 2.0
        oz = cz - (nz - 1) * sz / 2.0

        return [nx, ny, nz], [sx, sy, sz], [ox, oy, oz]

    def _cil_acq_geometry_to_rtk_proj_grid(self, ag):
        det_u = int(ag.config.panel.num_pixels[0])  
        det_v = int(ag.config.panel.num_pixels[1])  
        n_proj = int(len(ag.angles))

        du = float(ag.config.panel.pixel_size[0])   
        dv = float(ag.config.panel.pixel_size[1])   

        size = [det_u, det_v, n_proj]
        spacing = [du, dv, 1.0]
        origin = [
            -(det_u - 1) * du / 2.0,
            -(det_v - 1) * dv / 2.0,
            0.0
        ]
        return size, spacing, origin

    def _itk_proj_metadata(self, img_itk):
        region = img_itk.GetLargestPossibleRegion()
        size = region.GetSize()
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        return (
            [int(size[0]), int(size[1]), int(size[2])],
            [float(spacing[0]), float(spacing[1]), float(spacing[2])],
            [float(origin[0]), float(origin[1]), float(origin[2])],
        )

    def _make_constant_image(self, size, spacing, origin, constant=0.0):
        src = rtk.ConstantImageSource[self.ImageType].New()
        src.SetSize([int(size[0]), int(size[1]), int(size[2])])
        src.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])
        src.SetOrigin([float(origin[0]), float(origin[1]), float(origin[2])])
        src.SetConstant(float(constant))
        src.Update()
        return src.GetOutput()


    def _norm_label(self, lbl):
        s = str(lbl)
        if "." in s:
            s = s.split(".")[-1]
        return s.lower()

    def _transpose_to_order(self, arr, current_labels, desired_labels):
        cur = [self._norm_label(l) for l in current_labels]
        des = [self._norm_label(l) for l in desired_labels]
        perm = [cur.index(d) for d in des]
        return np.transpose(arr, axes=perm)

    def _transpose_from_order(self, arr, target_labels, source_labels):
        src = [self._norm_label(l) for l in source_labels]
        tgt = [self._norm_label(l) for l in target_labels]
        perm = [src.index(t) for t in tgt]
        return np.transpose(arr, axes=perm)


    def _apply_acq_convention_to_itk_array(self, arr_avh):
        """
        Convert CIL-aligned projection array [angle,vertical,horizontal] 
        """
        arr = arr_avh
        if self.reverse_angles:
            arr = arr[::-1, :, :]
        if self.flip_v:
            arr = arr[:, ::-1, :]
        if self.flip_u:
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _apply_itk_convention_to_acq_array(self, arr_avh):

        arr = arr_avh
        # need to flip???
        if self.flip_u:
            arr = arr[:, :, ::-1]
        if self.flip_v:
            arr = arr[:, ::-1, :]
        if self.reverse_angles:
            arr = arr[::-1, :, :]
        return np.ascontiguousarray(arr, dtype=np.float32)


    def _cil_image_to_itk(self, x):
        """
        CIL ImageData -> ITK image.

        ITK image_from_array expects NumPy layout [z,y,x] for a 3D image.
        """
        arr = np.asarray(x.array, dtype=np.float32)

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 3D image array (or 4D with singleton channel), got shape {arr.shape}")

        labels = list(getattr(x, "dimension_labels", []))
        if len(labels) == 4 and self._norm_label(labels[0]) == "channel":
            labels = labels[1:]

        desired = ["vertical", "horizontal_y", "horizontal_x"]  
        if len(labels) == 3:
            try:
                arr = self._transpose_to_order(arr, labels, desired)
            except Exception:
                pass  

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        img = itk.image_from_array(arr)
        img.SetSpacing([float(v) for v in self._vol_spacing])
        img.SetOrigin([float(v) for v in self._vol_origin])
        return img

    def _cil_acquisition_to_itk(self, x):
        """
        CIL AcquisitionData -> ITK projection stack.

        RTK expects NumPy layout [angle, vertical, horizontal] before itk.image_from_array(...)

        """
        arr = np.asarray(x.array, dtype=np.float32)

        # Support optional singleton channel dim: [C,A,V,H] -> [A,V,H]
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 3D acquisition array (or 4D with singleton channel), got shape {arr.shape}")

        labels = list(getattr(x, "dimension_labels", []))
        if len(labels) == 4 and self._norm_label(labels[0]) == "channel":
            labels = labels[1:]

        desired = ["angle", "vertical", "horizontal"]  # [A,V,H]
        if len(labels) == 3:
            try:
                arr = self._transpose_to_order(arr, labels, desired)
            except Exception:
                pass  

        arr = self._apply_acq_convention_to_itk_array(arr)

        n_geom = int(len(self.sinogram_geometry.angles))
        if arr.shape[0] != n_geom:
            raise ValueError(
                f"Projection stack angle dimension ({arr.shape[0]}) does not match "
                f"geometry entries ({n_geom})."
            )

        img = itk.image_from_array(arr)  # [A,V,H] -> ITK size [H,V,A]
        img.SetSpacing([float(v) for v in self._proj_spacing])
        img.SetOrigin([float(v) for v in self._proj_origin])
        return img

    def _itk_to_cil_image(self, img_itk, out=None):
        """
        ITK image -> CIL ImageData.
        """
        arr_zyx = np.array(itk.array_view_from_image(img_itk), dtype=np.float32, copy=True)

        target_labels = list(getattr(self.volume_geometry, "dimension_labels", []))
        add_channel = False
        if len(target_labels) == 4 and self._norm_label(target_labels[0]) == "channel":
            add_channel = True
            target_labels_3d = target_labels[1:]
        else:
            target_labels_3d = target_labels

        source_labels = ["vertical", "horizontal_y", "horizontal_x"]  # [z,y,x]

        if len(target_labels_3d) == 3:
            try:
                arr = self._transpose_from_order(arr_zyx, target_labels_3d, source_labels)
                arr = np.ascontiguousarray(arr, dtype=np.float32)
            except Exception:
                arr = arr_zyx
        else:
            arr = arr_zyx

        if add_channel:
            arr = arr[np.newaxis, ...]

        if out is not None:
            out.fill(arr)
            return out

        return ImageData(array=np.array(arr, dtype=np.float32, copy=True),
                         deep_copy=True,
                         geometry=self.volume_geometry.copy())

    def _itk_to_cil_acquisition(self, img_itk, out=None):
        """
        ITK projection stack -> CIL AcquisitionData.
        """
        arr_avh = np.array(itk.array_view_from_image(img_itk), dtype=np.float32, copy=True)  # [A,V,H]
        arr_avh = self._apply_itk_convention_to_acq_array(arr_avh)

        target_labels = list(getattr(self.sinogram_geometry, "dimension_labels", []))
        add_channel = False
        if len(target_labels) == 4 and self._norm_label(target_labels[0]) == "channel":
            add_channel = True
            target_labels_3d = target_labels[1:]
        else:
            target_labels_3d = target_labels

        source_labels = ["angle", "vertical", "horizontal"]

        if len(target_labels_3d) == 3:
            try:
                arr = self._transpose_from_order(arr_avh, target_labels_3d, source_labels)
                arr = np.ascontiguousarray(arr, dtype=np.float32)
            except Exception:
                arr = arr_avh
        else:
            arr = arr_avh

        if add_channel:
            arr = arr[np.newaxis, ...]

        if out is not None:
            out.fill(arr)
            return out

        return AcquisitionData(array=np.array(arr, dtype=np.float32, copy=True),
                               deep_copy=True,
                               geometry=self.sinogram_geometry.copy())


    def adjoint_test(self, x_seed=0, b_seed=1, verbose=True):
        """
        CIL-space adjoint test: <A x, b> vs <x, A^T b>
        x, b are CIL objects 
        """
        rng_x = np.random.default_rng(x_seed)
        rng_b = np.random.default_rng(b_seed)

        # random x
        x_shape = tuple(self.volume_geometry.shape)
        x_arr = rng_x.standard_normal(x_shape).astype(np.float32)
        x = ImageData(array=np.array(x_arr, copy=True), deep_copy=True, geometry=self.volume_geometry.copy())

        # random b
        b_shape = tuple(self.sinogram_geometry.shape)
        b_arr = rng_b.standard_normal(b_shape).astype(np.float32)
        b = AcquisitionData(array=np.array(b_arr, copy=True), deep_copy=True, geometry=self.sinogram_geometry.copy())

        Ax = self.direct(x)
        ATb = self.adjoint(b)

        lhs = float(np.sum(np.asarray(Ax.array, dtype=np.float64) * np.asarray(b.array, dtype=np.float64)))
        rhs = float(np.sum(np.asarray(x.array, dtype=np.float64) * np.asarray(ATb.array, dtype=np.float64)))

        abs_err = abs(lhs - rhs)
        rel_err = abs_err / max(abs(lhs), abs(rhs), 1e-12)

        if verbose:
            print(f"<Ax, b>     = {lhs:.8e}")
            print(f"<x, A^T b>  = {rhs:.8e}")
            print(f"abs error   = {abs_err:.8e}")
            print(f"rel error   = {rel_err:.8e}")

        return {"lhs": lhs, "rhs": rhs, "abs_err": abs_err, "rel_err": rel_err}