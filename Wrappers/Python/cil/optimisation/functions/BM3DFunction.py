import numpy as np
from cil.optimisation.functions import Function
import warnings

try:
    from bm3d import bm3d, BM3DStages
    _HAS_BM3D = True
    ALL_STAGES = BM3DStages.ALL_STAGES
except ImportError:
    bm3d = None
    BM3DStages = None
    ALL_STAGES = None
    _HAS_BM3D = False

    warnings.warn(
        "Optional dependency 'bm3d' is not installed.  Install via `pip install bm3d.",
        RuntimeWarning,
        stacklevel=2,
    )


class BM3DFunction(Function):
    """
    PnP 'regulariser' whose proximal applies BM3D denoising.

    Use in PnP-ISTA/FISTA
    Maybe add damping: (1-gamma) z + gamma * BM3D(z).
    """

    def __init__(self, sigma,  profile="np", stage_arg=ALL_STAGES, 
                 positivity=True):
        
        
        # self.gamma = float(gamma)      # damping in (0,1]
        # if not (0.0 < self.gamma <= 1.0):
            # raise ValueError("gamma must be in (0,1].")
        self.sigma = sigma
        if self.sigma<=0:
            raise ValueError("pos")
        self.profile = profile
        self.stage_arg = stage_arg
        self.positivity = positivity

        super(BM3DFunction, self).__init__(L=None)

    def __call__(self, x):
        # does not exist, return 0 for now, add warning
        return 0.0


    def _denoise(self, znp: np.ndarray) -> np.ndarray:
        z = np.asarray(znp, dtype=np.float32)
        # BM3D expects sigma as noise std (same units as the image)
        return bm3d(z, sigma_psd=self.sigma, profile=self.profile,
                    stage_arg=self.stage_arg).astype(np.float32)

    def proximal(self, x, tau, out=None):

        ## TODO asarray for SIRF?
        z = x.array.astype(np.float32, copy=False)
        den_bm3d_np = self._denoise(z)

        # damping/relaxation (oscillations)
        # u = (1.0 - self.gamma) * z + self.gamma * d

        if self.positivity:
            np.maximum(den_bm3d_np, 0.0, out=den_bm3d_np)

        if out is None:
            out = x * 0.0
        out.fill(den_bm3d_np)
        return out