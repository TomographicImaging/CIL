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

    r"""
    Plug-and-Play (PnP) BM3D prior.

    This class is meant to be used in proximal-gradient
    schemes (PnP-ISTA / PnP-FISTA), where the proximal operator is replaced
    by a BM3D denoiser:
    \[
        \operatorname{prox}_{\tau g}(x) \approx D_\sigma(x),
    \]
    with ``sigma`` interpreted as the assumed noise standard deviation in the image domain.

    Notes
    -----
    * The function value ``g(x)`` is not defined for PnP; therefore
      ``__call__`` returns ``0.0``.
    * Optionally enforces non-negativity by projecting the denoised output
      onto ``\{x \ge 0\}``.

    Parameters
    ----------
    sigma : float
        BM3D noise standard deviation (same units as the image). Must be > 0.

    profile : str, default="np"
        BM3D profile passed to ``bm3d`` (speed/quality trade-off). Available
        profiles are ``('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')

    stage_arg : BM3DStages or np.ndarray, default=BM3DStages.ALL_STAGES
        Controls which BM3D stage(s) are executed, or provides a pilot image:
        - ``BM3DStages.ALL_STAGES``: hard-thresholding + Wiener filtering.
        - ``BM3DStages.HARD_THRESHOLDING``: hard-thresholding only.
        - ``np.ndarray``: a pilot estimate of the noise-free image (used by BM3D).

    positivity : bool, default=True
        If ``True``, clip the denoised image to be non-negative.

    Note
    ----------
    Reference: Dabov,  K. and Foi,  A. and Katkovnik, V. and Egiazarian, K., 2007. Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering. IEEE Transactions on Image Processing. http://dx.doi.org/10.1109/TIP.2007.901238. 

    """
 

    def __init__(self, sigma,  profile="np", stage_arg=ALL_STAGES, 
                 positivity=True):
        
        self.sigma = sigma
        if self.sigma<=0:
            raise ValueError("Need a positive value for sigma")
        self.profile = profile
        self.stage_arg = stage_arg
        self.positivity = positivity
        self._warned_call = False

        super(BM3DFunction, self).__init__(L=None)

    def __call__(self, x):
        if not self._warned_call:
            warnings.warn(
                "BM3DFunction does not define objective value; returning 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_call = True
        return 0.0


    def _denoise(self, znp: np.ndarray) -> np.ndarray:
        z = np.asarray(znp, dtype=np.float32)
        return bm3d(z, sigma_psd=self.sigma, profile=self.profile,
                    stage_arg=self.stage_arg).astype(np.float32)

    def proximal(self, x, tau=1., out=None):

        ## TODO asarray for SIRF?
        z = x.array.astype(np.float32, copy=False)
        den_bm3d_np = self._denoise(z)
        

        ## TODO maybe we need a more general constraint?
        if self.positivity:
            np.maximum(den_bm3d_np, 0.0, out=den_bm3d_np)

        if out is None:
            out = x * 0.0
        out.fill(den_bm3d_np)

        return out