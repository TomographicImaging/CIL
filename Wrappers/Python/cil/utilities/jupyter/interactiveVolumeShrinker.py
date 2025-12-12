import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import (
    IntRangeSlider,
    Button,
    Output,
    Layout,
    VBox,
    HBox,
    interactive_output,
)
from IPython.display import display

from cil.plugins.astra.processors import FBP
from cil.utilities.shrink_volume import VolumeShrinker


class InteractiveVolumeShrinker:
    """
    Jupyter widget for interactively choosing a crop on a 3D volume
    and running VolumeShrinker with those limits.

    Parameters
    ----------
    absorption_data3D : AcquisitionData
        Projections to reconstruct from (full-resolution, ASTRA-ordered).
    preview : ImageData
        3D image used for visualization and slider placement (e.g. Slicer(roi_xy)(pdhg1.solution)).
    roi_xy : dict
        Offsets used to build `preview`, e.g.
        {'horizontal_x': (padsize_x, padend_x),
         'horizontal_y': (padsize_y, padend_y),
         'vertical':     (padsize_v, padend_v)}

    Attributes (after clicking "Apply")
    -----------------------------------
    cropped_ig : ImageGeometry
        Cropped image geometry returned by VolumeShrinker.
    crop_limits : dict
        Cropping limits in full-volume coordinates.
    last_recon : ImageData
        Reconstruction in the cropped geometry using FBP.
    last_fig : matplotlib.figure.Figure
        Last preview figure (three projections).
    last_recon_fig : matplotlib.figure.Figure
        Last reconstruction figure (axial/coronal/sagittal).
    """

    def __init__(self, absorption_data3D, preview, roi_xy, shrinker=None):
        self.absorption_data3D = absorption_data3D
        self.preview = preview
        self.roi_xy = roi_xy
        self.shrinker = shrinker or VolumeShrinker()

        self.cropped_ig = None
        self.crop_limits = None
        self.last_recon = None
        self.last_fig = None
        self.last_recon_fig = None

        self._build_widgets()

    def _build_widgets(self):
        dims = list(self.preview.dimension_labels)
        self.dims = dims
        sizes = {d: self.preview.get_dimension_size(d) for d in dims}
        self.sizes = sizes
        print("Preview sizes:", sizes)

        self.sx = IntRangeSlider(
            description="horizontal_x",
            min=0,
            max=sizes["horizontal_x"] - 1,
            value=[0, sizes["horizontal_x"] - 1],
            step=1,
            continuous_update=True,
            layout=Layout(width="460px"),
        )

        self.sy = IntRangeSlider(
            description="horizontal_y",
            min=0,
            max=sizes["horizontal_y"] - 1,
            value=[0, sizes["horizontal_y"] - 1],
            step=1,
            continuous_update=True,
            layout=Layout(width="460px"),
        )

        self.sv = IntRangeSlider(
            description="vertical",
            min=0,
            max=sizes["vertical"] - 1,
            value=[0, sizes["vertical"] - 1],
            step=1,
            continuous_update=True,
            layout=Layout(width="460px"),
        )

        out = interactive_output(
            self._update_plots,
            {"sx_range": self.sx, "sy_range": self.sy, "sv_range": self.sv},
        )

        self.apply_btn = Button(
            description="Apply crop with VolumeShrinker", button_style="success"
        )
        self.apply_out = Output()
        self.apply_btn.on_click(self._on_apply)

        # Top-level widget container
        self.widget = VBox(
            [
                VBox([out, HBox([self.sx, self.sy, self.sv])]),
                VBox([self.apply_btn, self.apply_out]),
            ]
        )


    @staticmethod
    def _draw_box(xmin, xmax, ymin, ymax):
        plt.plot([xmin, xmax], [ymin, ymin], "--r")
        plt.plot([xmin, xmax], [ymax, ymax], "--r")
        plt.plot([xmin, xmin], [ymin, ymax], "--r")
        plt.plot([xmax, xmax], [ymin, ymax], "--r")

    def _update_plots(self, sx_range, sy_range, sv_range):
        x0, x1 = sx_range
        y0, y1 = sy_range
        v0, v1 = sv_range

        fig = plt.figure(figsize=(13, 4))

        # (1) Max over vertical -> view X–Y
        ax1 = fig.add_subplot(1, 3, 1)
        img = self.preview.max(axis="vertical").array
        ax1.imshow(
            img,
            cmap="gray",
            origin="lower",
            extent=[0, self.sizes["horizontal_x"], 0, self.sizes["horizontal_y"]],
        )
        self._draw_box(x0, x1, y0, y1)
        ax1.set_title("Max over vertical (X–Y)")
        ax1.set_xlabel("horizontal_x")
        ax1.set_ylabel("horizontal_y")

        # (2) Max over horizontal_y -> view X–V
        ax2 = fig.add_subplot(1, 3, 2)
        img = self.preview.max(axis="horizontal_y").array
        ax2.imshow(
            img,
            cmap="gray",
            origin="lower",
            extent=[0, self.sizes["horizontal_x"], 0, self.sizes["vertical"]],
        )
        self._draw_box(x0, x1, v0, v1)
        ax2.set_title("Max over horizontal_y (X–V)")
        ax2.set_xlabel("horizontal_x")
        ax2.set_ylabel("vertical")

        # (3) Max over horizontal_x -> view Y–V
        ax3 = fig.add_subplot(1, 3, 3)
        img = self.preview.max(axis="horizontal_x").array
        ax3.imshow(
            img,
            cmap="gray",
            origin="lower",
            extent=[0, self.sizes["horizontal_y"], 0, self.sizes["vertical"]],
        )
        self._draw_box(y0, y1, v0, v1)
        ax3.set_title("Max over horizontal_x (Y–V)")
        ax3.set_xlabel("horizontal_y")
        ax3.set_ylabel("vertical")

        fig.tight_layout()
        self.last_fig = fig
        plt.show()


    def _on_apply(self, _):
        self.apply_out.clear_output()
        with self.apply_out:
            hx0, hx1 = self.sx.value
            hy0, hy1 = self.sy.value
            vz0, vz1 = self.sv.value

            # offsets from the original crop
            def _start(key):
                val = self.roi_xy.get(key, None)
                if isinstance(val, (tuple, list)) and len(val) >= 1:
                    return val[0]
                return 0

            off_x = _start("horizontal_x")
            off_y = _start("horizontal_y")
            off_v = _start("vertical")

            manual_limits = {
                "horizontal_x": (hx0 + off_x, hx1 + off_x + 1),
                "horizontal_y": (hy0 + off_y, hy1 + off_y + 1),
                "vertical": (vz0 + off_v, vz1 + off_v + 1),
            }
            print("mapped manual_limits →", manual_limits)

            cropped_ig = self.shrinker.run(
                self.absorption_data3D,
                auto=False,
                threshold="Otsu",
                buffer=None,
                mask_radius=None,
                manual_limits=manual_limits,
            )

            self.cropped_ig = cropped_ig
            self.crop_limits = manual_limits.copy()
            print("Cropped ImageGeometry stored in `self.cropped_ig`")
            print("Crop limits stored in `self.crop_limits`")

            # reconstruct in cropped geometry
            fbp = FBP(cropped_ig, self.absorption_data3D.geometry)
            recon = fbp(self.absorption_data3D)
            self.last_recon = recon

            arr = recon.as_array()
            zc, yc, xc = np.array(arr.shape) // 2
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(arr[zc, :, :], cmap="gray")
            axs[0].set_title("Axial")
            axs[0].axis("off")
            axs[1].imshow(arr[:, yc, :], cmap="gray")
            axs[1].set_title("Coronal")
            axs[1].axis("off")
            axs[2].imshow(arr[:, :, xc], cmap="gray")
            axs[2].set_title("Sagittal")
            axs[2].axis("off")
            plt.tight_layout()
            plt.show()

            self.last_recon_fig = fig

            return self.cropped_ig, self.crop_limits

    def show(self):
        display(self.widget)

        sx0, sx1 = self.sx.value
        sy0, sy1 = self.sy.value
        sv0, sv1 = self.sv.value

        self._update_plots(
            (sx0, sx1),
            (sy0, sy1),
            (sv0, sv1)
        )
