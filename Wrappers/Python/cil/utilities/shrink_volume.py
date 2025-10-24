import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.filters import threshold_otsu

from cil.processors import Binner, Slicer
from cil.plugins.astra.processors import FBP

import logging
log = logging.getLogger(__name__)
class VolumeShrinker(object):
    """
    Shrinks the reconstruction volume based on a supplied volume size or 
    automatic detection of the region of interest using Otsu thresholding and 
    connected components.   
    """

    def run(self, data, auto=True, threshold='Otsu', buffer=None, manual_limits=None):
        """
        Parameters
        ----------
        auto : bool, optional
            If True, automatically detect and crop the reconstruction volume
            If False, use manual_limits

        threshold: string or float, optional
            If automatically detecting the limits, specify the intensity threshold
            between sample and background. By default use an Otsu filter. 

        buffer: float, optional
            Add a buffer around the automatically detected limits, expressed as 
            a percentage of the axis size.

        manual_limits : dict, optional
            The limits {'axis_name1':(min, max), 'axis_name2':(min, max)}
            The `key` being the axis name to apply the processor to, 
            the `value` holding a tuple containing the min and max limits
            or None, to specify no limit
            Manual limits over-ride automatically detected limits
        """

        binning = min(int(np.ceil(data.geometry.config.panel.num_pixels[0] / 128)),16)
        angle_binning = np.ceil(data.get_dimension_size('angle')/(data.get_dimension_size('horizontal')*(np.pi/2)))
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle' : (None, None, angle_binning)
            }
        data_binned = Binner(roi)(data)

        ag = data_binned.geometry
        ig = ag.get_ImageGeometry()

        fbp = FBP(ig, ag)
        recon = fbp(data_binned)
        recon.apply_circular_mask(0.9)

        if auto:
            bounds = self.reduce_reconstruction_volume(recon, binning, threshold, buffer)
        else:
            bounds = {}
            for dim in recon.dimension_labels:
                bounds[dim] = (0, recon.get_dimension_size(dim)*binning)

        if manual_limits is not None:    
            for dim, v in manual_limits.items():
                if dim in recon.dimension_labels:
                    if v is None:
                        v = (0, recon.get_dimension_size(dim)*binning)
                    elif v[0] is None:
                        v[0] = 0
                    elif v[1] is None:
                        v[1] = recon.get_dimension_size(dim)*binning
                    bounds[dim] = v
                else:
                    raise ValueError("dimension {} not recognised, must be one of {}".format(dim, recon.dimension_labels))

        self.plot_with_bounds(recon, bounds, binning)

        return self.update_ig(data.geometry.get_ImageGeometry(), bounds)

    def update_ig(self, ig_unbinned, bounds):
        ig = Slicer(roi={'horizontal_x':(bounds['horizontal_x'][0], bounds['horizontal_x'][1],1),
                  'horizontal_y':(bounds['horizontal_y'][0], bounds['horizontal_y'][1], 1),
                  'vertical':(bounds['vertical'][0], bounds['vertical'][1], 1)})(ig_unbinned)
        return ig

    def plot_with_bounds(self, recon, bounds, binning):
        fig, axs = plt.subplots(nrows=1, ncols=recon.ndim, figsize=(14, 6))

        dims = recon.dimension_labels
        for i, dim in enumerate(dims):
            ax = axs[i]

            other_dims = [d for d in dims if d != dim]
            y_dim, x_dim = other_dims
            x_size = recon.get_dimension_size(x_dim)*binning
            y_size = recon.get_dimension_size(y_dim)*binning

            ax.imshow(recon.max(axis=dim).array, origin='lower', cmap='gray',
                    extent=[0, x_size, 0, y_size])
            
            x_min, x_max = bounds[x_dim]
            y_min, y_max = bounds[y_dim]

            ax.plot([x_min, x_max], [y_min, y_min], '--r')
            ax.plot([x_min, x_max], [y_max, y_max], '--r')
            ax.plot([x_min, x_min], [y_min, y_max], '--r')
            ax.plot([x_max, x_max], [y_min, y_max], '--r')

            ax.set_xlabel(x_dim)
            ax.set_ylabel(y_dim)
            ax.set_title(f"Maximum values in direction: {dim}")

    def reduce_reconstruction_volume(self, recon, binning, threshold, buffer):
            
        dims = recon.dimension_labels
        all_bounds = {dim: [] for dim in dims}

        for dim in dims:
            arr = recon.max(axis=dim).array
            mask, large_components_mask = self.otsu_large_components(arr, threshold)

            x_indices = np.where(np.any(large_components_mask, axis=0))[0]
            y_indices = np.where(np.any(large_components_mask, axis=1))[0]
            x_min, x_max = x_indices[0], x_indices[-1]
            y_min, y_max = y_indices[0], y_indices[-1]

            axis = recon.get_dimension_axis(dim)
            other_axes = [j for j in range(recon.ndim) if j != axis]

            if buffer is not None:
                y_full = recon.get_dimension_size(dims[other_axes[0]])
                y_min_buffer = np.max([0, (y_min-y_full//buffer)])
                y_max_buffer = np.min([y_full, y_max+(y_full//buffer)])

                x_full = recon.get_dimension_size(dims[other_axes[1]])
                x_min_buffer = np.max([0, (x_min-x_full//buffer)])
                x_max_buffer = np.min([x_full, x_max+(x_full//buffer)])

                all_bounds[dims[other_axes[0]]].append((y_min_buffer, y_max_buffer))
                all_bounds[dims[other_axes[1]]].append((x_min_buffer, x_max_buffer))
            else:
                all_bounds[dims[other_axes[0]]].append((y_min, y_max))
                all_bounds[dims[other_axes[1]]].append((x_min, x_max))

        bounds = {}
        for dim in dims:

            mins = [b[0] for b in all_bounds[dim]]
            maxs = [b[1] for b in all_bounds[dim]]
            dim_min = np.min(mins)*binning
            dim_max = np.max(maxs)*binning
            
            bounds[dim] = (dim_min, dim_max)

            if log.isEnabledFor(logging.DEBUG):
                print(f"{dim}: {bounds[dim][0]} to {bounds[dim][1]}")
            
        return bounds
        
    def otsu_large_components(self, arr, threshold):
        

        if isinstance(threshold, (int, float)):
            thresh = threshold
        elif isinstance(threshold, str) and threshold.lower() == 'otsu':
            thresh = threshold_otsu(arr[arr > 0])
        else:
            raise ValueError(f"Threshold {threshold} not recognised, must be a number or 'Otsu'")
        mask = arr > thresh


        labeled_mask, num_features = label(mask)
        component_sizes = np.bincount(labeled_mask.ravel())
        min_size = 10

        large_labels = np.where(component_sizes > min_size)[0]
        large_labels = large_labels[large_labels != 0]  
        large_components_mask = np.isin(labeled_mask, large_labels)

        if log.isEnabledFor(logging.DEBUG):
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5))

            axes[0].imshow(arr, cmap=plt.cm.gray)
            axes[0].set_title('Original')

            axes[1].hist(arr.ravel(), bins=100)
            axes[1].set_title('Histogram')
            axes[1].axvline(thresh, color='r')

            axes[2].imshow(mask, cmap=plt.cm.gray, extent=[axes[0].get_xlim()[0], axes[0].get_xlim()[1], axes[0].get_ylim()[0], axes[0].get_ylim()[1]])
            axes[2].set_title('Thresholded')

            axes[3].imshow(large_components_mask, cmap=plt.cm.gray, extent=[axes[0].get_xlim()[0], axes[0].get_xlim()[1], axes[0].get_ylim()[0], axes[0].get_ylim()[1]])
            axes[3].set_title('Large components')

            x_indices = np.where(np.any(large_components_mask, axis=0))[0]
            y_indices = np.where(np.any(large_components_mask, axis=1))[0]
            x_min, x_max = x_indices[0], x_indices[-1]
            y_min, y_max = y_indices[0], y_indices[-1]

            axes[3].plot([x_min, x_max], [y_min, y_min], '--r')
            axes[3].plot([x_min, x_max], [y_max, y_max], '--r')
            axes[3].plot([x_min, x_min], [y_min, y_max], '--r')
            axes[3].plot([x_max, x_max], [y_min, y_max], '--r')

            plt.tight_layout()

        return mask, large_components_mask