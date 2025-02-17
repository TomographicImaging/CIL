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

from cil.framework import Processor, AcquisitionData
from cil.framework.labels import AcquisitionDimension, AcquisitionType
import matplotlib.pyplot as plt
import scipy
import numpy as np
import logging
import math
import importlib

log = logging.getLogger(__name__)

class CofR_image_sharpness(Processor):

    """This creates a CentreOfRotationCorrector processor.

    The processor will find the centre offset by maximising the sharpness of a reconstructed slice.

    Can be used on single slice parallel-beam, and centre slice cone beam geometry. For use only with datasets that can be reconstructed with FBP/FDK.

    Parameters
    ----------

    slice_index : int, str, default='centre'
        An integer defining the vertical slice to run the algorithm on. The special case slice 'centre' is the default.

    backend : {'tigre', 'astra'}
        The backend to use for the reconstruction

    tolerance : float, default=0.005
        The tolerance of the fit in pixels, the default is 1/200 of a pixel. This is a stopping criteria, not a statement of accuracy of the algorithm.

    search_range : int
        The range in pixels to search either side of the panel centre. If `None` a quarter of the width of the panel is used.

    initial_binning : int
        The size of the bins for the initial search. If `None` will bin the image to a step corresponding to <128 pixels. The fine search will be on unbinned data.


    Example
    -------
    .. code-block :: python 
        from cil.processors import CentreOfRotationCorrector

        processor = CentreOfRotationCorrector.image_sharpness('centre', 'tigre')
        processor.set_input(data)
        data_centred = processor.get_output()


    Example
    -------
    .. code-block :: python
        from cil.processors import CentreOfRotationCorrector

        processor = CentreOfRotationCorrector.image_sharpness(slice_index=120, 'astra')
        processor.set_input(data)
        processor.get_output(out=data)


    Note
    ----
    For best results data should be 360deg which leads to blurring with incorrect geometry.
    This method is unreliable on half-scan data with 'tuning-fork' style artifacts.

    """
    _supported_backends = ['astra', 'tigre']

    def __init__(self, slice_index='centre', backend='tigre', tolerance=0.005, search_range=None, initial_binning=None):

        FBP = self._configure_FBP(backend)


        kwargs = {
                    'slice_index': slice_index,
                    'FBP': FBP,
                    'backend' : backend,
                    'tolerance': tolerance,
                    'search_range': search_range,
                    'initial_binning': initial_binning
                 }

        super(CofR_image_sharpness, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(data, AcquisitionData):
            raise Exception('Processor supports only AcquisitionData')

        if data.geometry == None:
            raise Exception('Geometry is not defined.')

        if data.geometry.geom_type == 'cone' and self.slice_index != 'centre':
            raise ValueError("Only the centre slice is supported with this algorithm")

        if data.geometry.system_description not in ['simple','offset']:
            raise NotImplementedError("Not implemented for rotated system geometries")

        if data.geometry.channels > 1:
            raise ValueError("Only single channel data is supported with this algorithm")

        if self.slice_index != 'centre':
            try:
                int(self.slice_index)
            except:
                raise ValueError("slice_index expected to be a positive integer or the string 'centre'. Got {0}".format(self.slice_index))

            if self.slice_index < 0 or self.slice_index >= data.get_dimension_size('vertical'):
                raise ValueError('slice_index is out of range. Must be in range 0-{0}. Got {1}'.format(data.get_dimension_size('vertical'), self.slice_index))

        #check order for single slice data
        test_geom = data.geometry.get_slice(vertical='centre') if AcquisitionType.DIM3 & data.geometry.dimension else data.geometry

        if not AcquisitionDimension.check_order_for_engine(self.backend, test_geom):
            raise ValueError("Input data must be reordered for use with selected backend. Use input.reorder{'{0}')".format(self.backend))

        return True


    def _configure_FBP(self, backend='tigre'):
        """
        Configures the processor for the right engine. Checks the data order.
        """
        if backend not in self._supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self._supported_backends))

        #set FBPOperator class from backend
        try:
            module = importlib.import_module(f'cil.plugins.{backend}')
        except ImportError as exc:
            msg = {'tigre': "TIGRE (e.g. `conda install conda-forge::tigre`)",
                   'astra': "ASTRA (e.g. `conda install astra-toolbox::astra-toolbox`)"}.get(backend, backend)
            raise ImportError(f"Please install {msg} or select a different backend") from exc

        return module.FBP


    def gss(self, data, ig, search_range, tolerance, binning):
        '''Golden section search'''
        # intervals c:cr:c where r = φ − 1=0.619... and c = 1 − r = 0.381..., φ
        log.debug("GSS between %f and %f", *search_range)
        phi = (1 + math.sqrt(5))*0.5
        r = phi - 1
        #1/(r+2)
        r2inv = 1/ (r+2)
        #c = 1 - r

        all_data = {}
        #set up
        sample_points = [np.nan]*4
        evaluation = [np.nan]*4

        sample_points[0] = search_range[0]
        sample_points[3] = search_range[1]

        interval = sample_points[-1] - sample_points[0]
        step_c = interval *r2inv
        sample_points[1] = search_range[0] + step_c
        sample_points[2] = search_range[1] - step_c

        for i in range(4):
            evaluation[i] = self.calculate(data, ig, sample_points[i])
            all_data[sample_points[i]] = evaluation[i]

        count = 0
        while(count < 30):
            ind = np.argmin(evaluation)
            if ind == 1:
                del sample_points[-1]
                del evaluation[-1]

                interval = sample_points[-1] - sample_points[0]
                step_c = interval *r2inv
                new_point = sample_points[0] + step_c

            elif ind == 2:
                del sample_points[0]
                del evaluation[0]

                interval = sample_points[-1] - sample_points[0]
                step_c = interval *r2inv
                new_point = sample_points[-1]- step_c

            else:
                raise ValueError("The centre of rotation could not be located to the requested tolerance. Try increasing the search tolerance.")

            if interval < tolerance:
                break

            sample_points.insert(ind, new_point)
            obj = self.calculate(data, ig, new_point)
            evaluation.insert(ind, obj)
            all_data[new_point] = obj

            count +=1

        log.info("evaluated %d points",len(all_data))
        if log.isEnabledFor(logging.DEBUG):
            keys, values = zip(*all_data.items())
            self.plot(keys, values, ig.voxel_size_x/binning)

        z = np.polyfit(sample_points, evaluation, 2)
        min_point = -z[1] / (2*z[0])

        if np.sign(z[0]) == 1 and min_point < sample_points[2] and min_point > sample_points[0]:
            return min_point
        else:
            ind = np.argmin(evaluation)
            return sample_points[ind]

    def calculate(self, data, ig, offset):
        ag_shift = data.geometry.copy()
        ag_shift.config.system.rotation_axis.position = [offset, 0]

        reco = self.FBP(ig, ag_shift)(data)
        return (reco*reco).sum()

    def plot(self, offsets,values, vox_size):
        x=[x / vox_size for x in offsets]
        y=values

        plt.figure()
        plt.scatter(x,y)
        plt.show()

    def get_min(self, offsets, values, ind):
        #calculate quadratic from 3 points around ind  (-1,0,1)
        a = (values[ind+1] + values[ind-1] - 2*values[ind]) * 0.5
        b = a + values[ind] - values[ind-1]
        ind_centre = -b / (2*a)+ind

        ind0 = int(ind_centre)
        w1 = ind_centre - ind0
        return (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]

    def process(self, out=None):
        #get slice
        data = data_full = self.get_input()

        if AcquisitionType.DIM3 & data_full.geometry.dimension:
            data = data.get_slice(vertical=self.slice_index)

        data.geometry.config.system.align_reference_frame('cil')
        width = data.geometry.config.panel.num_pixels[0]

        #initial grid search
        if self.search_range is None:
            self.search_range = width //4

        if self.initial_binning is None:
            self.initial_binning = min(int(np.ceil(width / 128)),16)

        log.debug("Initial search:")
        log.debug("search range is %d", self.search_range)
        log.debug("initial binning is %d", self.initial_binning)

        #filter full projections
        data_filtered = data.copy()
        data_filtered.fill(scipy.ndimage.sobel(data.as_array(), axis=1, mode='reflect', cval=0.0))

        if self.initial_binning > 1:

            #gaussian filter data
            data_temp = data_filtered.copy()
            data_temp.fill(scipy.ndimage.gaussian_filter(data_filtered.as_array(), [0,self.initial_binning//2]))

            #bin data whilst preserving centres
            num_pix_new = np.ceil(width/self.initial_binning)

            new_half_panel = (num_pix_new - 1)/2
            half_panel = (width - 1)/2

            sampling_points = np.mgrid[-self.initial_binning*new_half_panel:self.initial_binning*new_half_panel+1:self.initial_binning]
            initial_coordinates = np.mgrid[-half_panel:half_panel+1:1]

            new_geom = data.geometry.copy()
            new_geom.config.panel.num_pixels[0] = num_pix_new
            new_geom.config.panel.pixel_size[0] *= self.initial_binning
            data_binned = new_geom.allocate()

            for i in range(data.shape[0]):
                data_binned.fill(np.interp(sampling_points, initial_coordinates, data.array[i,:]),angle=i)

            #filter
            data_binned_filtered = data_binned.copy()
            data_binned_filtered.fill(scipy.ndimage.sobel(data_binned.as_array(), axis=1, mode='reflect', cval=0.0))
            data_processed = data_binned_filtered
        else:
            data_processed = data_filtered

        ig = data_processed.geometry.get_ImageGeometry()

        #binned grid search
        vox_rad = np.ceil(self.search_range /self.initial_binning)
        steps = int(4*vox_rad + 1)
        offsets = np.linspace(-vox_rad, vox_rad, steps) * ig.voxel_size_x
        obj_vals = []

        for offset in offsets:
            obj_vals.append(self.calculate(data_processed, ig, offset))

        if log.isEnabledFor(logging.DEBUG):
            self.plot(offsets,obj_vals,ig.voxel_size_x / self.initial_binning)

        ind = np.argmin(obj_vals)
        if ind == 0 or ind == len(obj_vals)-1:
            raise ValueError ("Unable to minimise function within set search_range")
        else:
            centre = self.get_min(offsets, obj_vals, ind)

        if self.initial_binning > 8:
            #binned search continued
            log.debug("binned search starting at %f", centre)
            a = centre - ig.voxel_size_x *2
            b = centre + ig.voxel_size_x *2
            centre = self.gss(data_processed,ig, (a, b), self.tolerance *ig.voxel_size_x, self.initial_binning )

        #fine search
        log.debug("fine search starting at %f", centre)
        data_processed = data_filtered
        ig = data_processed.geometry.get_ImageGeometry()
        a = centre - ig.voxel_size_x *2
        b = centre + ig.voxel_size_x *2
        centre = self.gss(data_processed,ig, (a, b), self.tolerance *ig.voxel_size_x, 1 )

        new_geometry = data_full.geometry.copy()
        new_geometry.config.system.rotation_axis.position[0] = centre

        log.info("Centre of rotation correction found using image_sharpness")
        log.info("backend FBP/FDK {}".format(self.backend))
        log.info("Calculated from slice: %s", str(self.slice_index))
        log.info("Centre of rotation shift = %f pixels", centre/ig.voxel_size_x)
        log.info("Centre of rotation shift = %f units at the object", centre)
        log.info("Return new dataset with centred geometry")

        if out is None:
            return AcquisitionData(array=data_full, deep_copy=True, geometry=new_geometry, supress_warning=True)
        else:
            out.geometry = new_geometry
            return out