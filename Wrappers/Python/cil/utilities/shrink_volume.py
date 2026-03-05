#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label
from skimage.filters import threshold_multiotsu
import importlib

from cil.processors import Binner, Slicer


import logging
log = logging.getLogger(__name__)
class VolumeShrinker(object):
    """
    Shrinks the reconstruction volume from a dataset based on supplied volume 
    limits, thresholding or automatic detection of the region of interest using 
    Otsu thresholding.
    
    Parameters
    ----------
    data: AcquisitionData
        The dataset to create a reduced reconstruction volume from.

    recon_backend : {'tigre', 'astra'}
        The plugin backend to use for the reconstruction

    """

    _supported_backends = ['astra', 'tigre']

    def __init__(self, data, recon_backend='tigre'):
         
        self.data = data
        self.recon_backend = recon_backend
        self.recon_method = self._configure_recon(recon_backend)

    def run(self, limits=None, preview=True, method='manual', **kwargs):
        """
        Parameters
        ----------
        limits : dict, optional
            ImageGeometry limits {'axis_name1':(min, max), 'axis_name2':(min, max)}
            The `key` being the ImageGeometry axis name and `value` a tuple containing 
            the min and max limits.
            Default None, uses the full extent of the data

        preview: bool, optional
            If True, plots the maximum values from a binned reconstruction in each 
            direction to check the ImageGeometry contains the full volume. 
            Default is False.
        
        method : string, optional
            If 'manual', use `limits`
            If 'threshold' crop the reconstruction volume based on a threshold
            value between sample and background. Pass threshold as a kwarg.
            If 'otsu', automatically detect and crop the reconstruction volume.
            Default is 'manual'

        **kwargs:
        threshold: float, optional
            If using 'threshold' method, specify the intensity threshold
            between sample and background in the reconstruction. Default is None.

        buffer: float, optional
            Buffer in pixels around the automatically detected limits. 
            Default is 0, no buffer added.

        mask_radius: float, optional
            Radius of circular mask to apply on the reconstructed volume. This
            impacts the automatic cropping of the reconstruction volume when method
            is 'threshold' or 'otsu' and displaying with preview. Default is None.

        otsu_classes: int, optional
            Number of material classes to use when automatically detecting the 
            reconstruction volume using Otsu thresholding method. Default is 2.

        min_component_size: int, optional
            Minimum size in pixels of connected components to consider when automatically 
            cropping the reconstruction volume. Default is None.

        Returns
        -------
        ig_reduced: ImageGeometry
            The reduced size ImageGeometry

        Example
        -------
        >>> vs = VolumeShrinker(data, recon_backend='astra')
        >>> ig_reduced = vs.run(method='manual', limits={'horizontal_x':(10, 150)})
        
        Example
        -------
        >>> cil_log_level = logging.getLogger()
        >>> cil_log_level.setLevel(logging.DEBUG)
        >>> vs = VolumeShrinker(data, recon_backend='astra')
        >>> ig_reduced = vs.run(method='threshold', threshold=0.9)

        Example
        -------
        >>> cil_log_level = logging.getLogger()
        >>> cil_log_level.setLevel(logging.DEBUG)
        >>> vs = VolumeShrinker(data, recon_backend='astra')
        >>> ig_reduced = vs.run(method='otsu', otsu_classes=3)

        Note
        ----
        For the `otsu` and `threshold` methods, setting logging to 'debug' shows 
        a plot of the threshold on a histogram of the data and the threshold mask 
        on the reconstruction.
        """

        ig = self.data.geometry.get_ImageGeometry()
        if method.lower() == 'manual':
            bounds = {}
            for dim in ig.dimension_labels:
                bounds[dim] = (0, ig.shape[ig.dimension_labels.index(dim)])
            if limits is not None:
                for dim, v in limits.items():
                    if dim in ig.dimension_labels:
                        if v is None:
                            v = (0, ig.shape[ig.dimension_labels.index(dim)])
                        else:
                            if v[0] is None:
                                v = list(v)
                                v[0] = 0
                                v = tuple(v)
                            if v[1] is None:
                                v = list(v)
                                v[1] = ig.shape[ig.dimension_labels.index(dim)]
                                v = tuple(v)
                        bounds[dim] = v
                    else:
                        raise ValueError("dimension {} not recognised, must be one of {}".format(dim, ig.dimension_labels))

        elif method.lower() in ['threshold', 'otsu']:
            mask_radius = kwargs.pop('mask_radius', None)
            recon, binning = self._get_recon(mask_radius=mask_radius)
            threshold = kwargs.pop('threshold', None)
            buffer = kwargs.pop('buffer', 0)
            min_component_size = kwargs.pop('min_component_size', None)
            otsu_classes = kwargs.pop('otsu_classes', 2) 
            bounds = self._reduce_reconstruction_volume(recon, binning, method, 
                                                        threshold, buffer, min_component_size, otsu_classes)
            
        else:
            raise ValueError("Method {method} not recognised, must be one of 'manual', 'threshold' or 'otsu'")

        if preview:
            if method.lower() == 'manual':
                mask_radius = kwargs.pop('mask_radius', None)
                recon, binning = self._get_recon(mask_radius=mask_radius)

            self._plot_with_bounds(recon, bounds, binning)
        
        return self.update_ig(self.data.geometry.get_ImageGeometry(), bounds)

    def update_ig(self, ig_unbinned, bounds):
        """
        Return a new unbinned ImageGeometry with the bounds applied
        """
        if bounds is None:
            ig = ig_unbinned
        else:
            ig = Slicer(roi={'horizontal_x':(bounds['horizontal_x'][0], bounds['horizontal_x'][1],1),
                    'horizontal_y':(bounds['horizontal_y'][0], bounds['horizontal_y'][1], 1),
                    'vertical':(bounds['vertical'][0], bounds['vertical'][1], 1)})(ig_unbinned)
        return ig
    
    def _get_recon(self, mask_radius=None):
        """
        Gets a binned reconstruction from the dataset with an optional mask
        Also returns the binning which has been used
        """
        binning = min(int(np.ceil(self.data.geometry.config.panel.num_pixels[0] / 128)),16)
        angle_binning = np.ceil(self.data.get_dimension_size('angle')/(self.data.get_dimension_size('horizontal')*(np.pi/2)))
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle' : (None, None, angle_binning)
            }
        data_binned = Binner(roi)(self.data)

        ag = data_binned.geometry
        ig = ag.get_ImageGeometry()

        fbp = self.recon_method(ig, ag)
        recon = fbp(data_binned)
        
        if mask_radius is not None:
            recon.apply_circular_mask(mask_radius)

        return recon, binning

    def _plot_with_bounds(self, recon, bounds, binning):
        """
        Plots the bounds on the maximum value in the reconstructed dataset along 
        each direction.
        """

        fig, axs = plt.subplots(nrows=1, ncols=recon.ndim, figsize=(14, 6))

        dims = recon.dimension_labels
        for i, dim in enumerate(dims):
            ax = axs[i]

            other_dims = [d for d in dims if d != dim]
            y_dim, x_dim = other_dims
            x_size = recon.get_dimension_size(x_dim)*binning
            y_size = recon.get_dimension_size(y_dim)*binning

            im = ax.imshow(recon.max(axis=dim).array, origin='lower', cmap='gray',
                    extent=[0, x_size, 0, y_size])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            if bounds is not None:
                x_min, x_max = bounds[x_dim]
                y_min, y_max = bounds[y_dim]

                ax.plot([x_min, x_max], [y_min, y_min], '--r')
                ax.plot([x_min, x_max], [y_max, y_max], '--r')
                ax.plot([x_min, x_min], [y_min, y_max], '--r')
                ax.plot([x_max, x_max], [y_min, y_max], '--r')

            ax.set_xlabel(f"Downsampled  {x_dim}")
            ax.set_ylabel(f"Downsampled  {y_dim}")
            ax.set_title(f"Maximum values in direction: {dim}")
        plt.tight_layout()

    def _reduce_reconstruction_volume(self, recon, binning, method, 
                                      threshold=None, buffer=0, min_component_size=None, otsu_classes=2):
        """
        Automatically finds the boundaries of the sample in a reconstructed
        volume based on a threshold between sample and background. 
        If method=`threshold`, the threshold must be provided.
        If method=`otsu`, the threshold is calculated from an Otsu filter. The
        number of `otsu_classes` can be passed. 
        """
          

        dims = recon.dimension_labels
        all_bounds = {dim: [] for dim in dims}

        for dim in dims:
            arr = recon.max(axis=dim).array
            
            if method.lower() == 'threshold':
                if threshold is not None:
                    if threshold >= arr.max():
                        raise ValueError(f"Threshold {threshold} is greater than maximum value in dimension: {arr.max()}. Try specifying a lower threshold.")
                    elif threshold <= arr.min():
                        raise ValueError(f"Threshold {threshold} is less than minimum value in dimension: {arr.min()}. Try specifying a higher threshold.")
                    mask = arr > threshold

                else:
                    raise ValueError("You must supply a threshold argument if method='threshold'")

            elif method.lower() == 'otsu':
                n_bins = 256
                threshold = threshold_multiotsu(arr[arr>0], classes=otsu_classes, nbins=n_bins)
                threshold = threshold[0]
                mask = arr > threshold
                if not mask.any():
                    raise ValueError("No pixels found within threshold, consider using a different number of otsu_classes or method='threshold' or 'manual'")

            if min_component_size is not None:
                mask = self._threshold_large_components(mask, min_component_size)
                if not mask.any():
                    raise ValueError("No pixels found within threshold, consider reducing min_component_size")

            x_indices = np.where(np.any(mask, axis=0))[0]
            y_indices = np.where(np.any(mask, axis=1))[0]
            x_min, x_max = x_indices[0], x_indices[-1]
            y_min, y_max = y_indices[0], y_indices[-1]

            if log.isEnabledFor(logging.DEBUG):
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))

                axes[0].imshow(arr, cmap=plt.cm.gray)
                axes[0].set_title('Maximum values')

                axes[1].hist(arr.ravel(), bins=100)
                axes[1].set_title('Histogram of maximum values')
                axes[1].axvline(threshold, color='r')

                axes[2].imshow(mask, cmap=plt.cm.gray, extent=[axes[0].get_xlim()[0], axes[0].get_xlim()[1], axes[0].get_ylim()[0], axes[0].get_ylim()[1]])
                axes[2].set_title('Calculated threshold mask')

                axes[2].plot([x_min, x_max], [y_min, y_min], '--r')
                axes[2].plot([x_min, x_max], [y_max, y_max], '--r')
                axes[2].plot([x_min, x_min], [y_min, y_max], '--r')
                axes[2].plot([x_max, x_max], [y_min, y_max], '--r')

                plt.suptitle(dim)
                
                plt.tight_layout()

            axis = recon.get_dimension_axis(dim)
            other_axes = [j for j in range(recon.ndim) if j != axis]


            all_bounds[dims[other_axes[0]]].append((y_min, y_max))
            all_bounds[dims[other_axes[1]]].append((x_min, x_max))

        bounds = {}
        for dim in dims:

            mins = [b[0] for b in all_bounds[dim]]
            maxs = [b[1] for b in all_bounds[dim]]
            dim_min = np.min(mins)*binning
            dim_max = np.max(maxs)*binning

            # add the buffer but limit to original ig size
            dim_min = np.max([0, (dim_min - buffer)]) 
            dim_full = recon.get_dimension_size(dim)*binning
            dim_max = np.min([dim_full, (dim_max + buffer)])
            
            bounds[dim] = (dim_min, dim_max)

            if log.isEnabledFor(logging.DEBUG):
                print(f"{dim}: {bounds[dim][0]} to {bounds[dim][1]}")
            
        return bounds
        
    def _threshold_large_components(self, mask, min_component_size):
        """
        Modify a threshold mask to ignore clusters of pixel values above the threshold 
        if the cluster is below a specified `min_component_size` in pixels. This 
        can be useful for noisy datasets where a few noisy pixels above the threshold 
        may appear in the background and erroneously increase the bounds.
        """
        labeled_mask, _ = label(mask)
        component_sizes = np.bincount(labeled_mask.ravel())

        large_labels = np.where(component_sizes > min_component_size)[0]
        large_labels = large_labels[large_labels != 0]  
        large_components_mask = np.isin(labeled_mask, large_labels)

        return large_components_mask
    
    def _configure_recon(self, backend='tigre'):
        """
        Configures the recon for the right engine.
        """
        if backend not in self._supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self._supported_backends))

        module = importlib.import_module(f'cil.plugins.{backend}')

        return module.FBP
