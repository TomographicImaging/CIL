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

from abc import ABCMeta
from typing import List, Literal, Optional, Union
from cil.framework import cilacc
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.framework.framework import ImageData, ImageGeometry
from .Reconstructor import Reconstructor
from scipy.fft import fftfreq

import numpy as np
import ctypes
from tqdm import tqdm

c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)

try:
    cilacc.filter_projections_avh
    has_ipp = True
except:
    has_ipp = False

if has_ipp:
    cilacc.filter_projections_avh.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the data array
                                    ctypes.POINTER(ctypes.c_float),  # pointer to the filter array
                                    ctypes.POINTER(ctypes.c_float),  # pointer to the weights array
                                    ctypes.c_int16, #order of the fft
                                    ctypes.c_long, #num_proj
                                    ctypes.c_long, #pix_v
                                    ctypes.c_long] #pix_x

    cilacc.filter_projections_vah.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the data array
                                    ctypes.POINTER(ctypes.c_float),  # pointer to the filter array
                                    ctypes.POINTER(ctypes.c_float),  # pointer to the weights array
                                    ctypes.c_int16, #order of the fft
                                    ctypes.c_long, #pix_v
                                    ctypes.c_long, #num_proj
                                    ctypes.c_long] #pix_x

class GenericFilteredBackProjection(Reconstructor, metaclass=ABCMeta):
    """Abstract Base Class GenericFilteredBackProjection holding common and virtual methods for FBP and FDK.

    Parameters
    ----------
    input
        Input data for reconstruction
    image_geometry
        Image geometry of input data, by default None
    filter
        Reconstruction filter from a string selecting from the list of pre-set filters, or pass a numpy.ndarray with a custom filter.
    backend
        Engine backend used for reconstruction, by default "tigre"

    Raises
    ------
    ImportError
        Raised if IPP libraries cannot be found
    ValueError
        Raised if input data is multi-channel
    """

    def __init__(self, input:AcquisitionData, image_geometry:Optional[ImageGeometry]=None, filter:str="ram-lak", backend:Literal["tigre"]="tigre"):
        #call parent initialiser
        super().__init__(input, image_geometry, backend)

        if has_ipp == False:
            raise ImportError("IPP libraries not found. Cannot use CIL FBP")

        #additional check
        if "channel" in input.dimension_labels:
            raise ValueError("Input data cannot be multi-channel")

        #define defaults
        self._fft_order = self._default_fft_order()
        self.set_filter(filter)
        self.set_filter_inplace(False)
        self._weights = None

    @property
    def filter(self) -> str:
        """Get the current filter.

        Returns
        -------
        str
            Name of the current filter, if 'custom', ten get_filter_array will return the numpy array used for filtering.
        """
        return self._filter

    @property
    def filter_inplace(self) -> bool:
        """If True, data filtering is done in-place, rather than allocating temporary memory.

        Returns
        -------
        bool
            True if inplace filtering is performed, false otherwise.
        """
        return self._filter_inplace

    @property
    def fft_order(self) -> int:
        """The width of the fourier transform N=2^order.

        Returns
        -------
        int
            Width of the fourier transform N=2^order.
        """
        return self._fft_order

    @property
    def preset_filters(self) -> List[str]:
        """Return a list of all filter presets.

        Returns
        -------
        str
            Supported filter presets
        """
        return ["ram-lak", "shepp-logan", "cosine", "hamming", "hann"]

    def set_filter_inplace(self, inplace:bool=False) -> None:
        """Allocate temporary memory for filtered projections.

        False (default) will allocate temporary memory for filtered projections.
        True will filter projections in-place.

        Parameters
        ----------
        inplace: boolean
            Sets the inplace filtering of projections
        """
        if type(inplace) is bool:
            self._filter_inplace= inplace
        else:
            raise TypeError("set_filter_inplace expected a boolean. Got {}".format(type(inplace)))

    def set_fft_order(self, order:Optional[int]=None) -> None:
        """Width of the fourier transform N=2^order.

        Parameters
        ----------
        order
            The width of the fft N=2^order

        Notes
        -----
        If `None` the default used is the power-of-2 greater than 2 * detector width, or 8, whichever is greater
        Higher orders will yield more accurate results but increase computation time.
        """
        min_order = self._default_fft_order()

        if order is None:
            fft_order = min_order
        else:
            try:
                fft_order = int(order)
            except TypeError:
                raise TypeError("fft order expected type `int`. Got{}".format(type(order)))

        if fft_order < min_order:
            raise ValueError("Minimum fft width 2^order is order = {0}. Got{1}".format(min_order,order))

        if fft_order != self.fft_order:
            self._fft_order = fft_order

            if self.filter=="custom":
                print("Filter length changed - please update your custom filter")
            else:
                #create default filter type of new length
                self.set_filter(self._filter)

    def set_filter(self, filter:Union[str, np.ndarray]="ram-lak", cutoff:float=1.0) -> None:
        """Set the filter used by the reconstruction.

        Pre-set filters are constructed in the frequency domain.
        Pre-set filters are: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'

        Parameters
        ----------
        filter
            Pass a string selecting from the list of pre-set filters, or pass a numpy.ndarray with a custom filter.
        cutoff
            The cut-off frequency of the filter between 0 - 1 pi rads/pixel. The filter will be 0 outside the range rect(-frequency_cutoff, frequency_cutoff)

        Notes
        -----
        If passed a numpy array the filter must have length N = 2^self.fft_order

        The indices of the array are interpreted as:

        - [0] The DC frequency component
        - [1:N/2] positive frequencies
        - [N/2:N-1] negative frequencies
        """
        if type(filter)==str and filter in self.preset_filters:
            self._filter = filter
            self._filter_cutoff = cutoff
            self._filter_array = None

        elif type(filter)==np.ndarray:
            try:
                filter_array = np.asarray(filter,dtype=np.float32).reshape(2**self.fft_order)
                self._filter_array = filter_array.copy()
                self._filter = "custom"
            except ValueError:
                raise ValueError("Custom filter not compatible with input.")
        else:
            raise ValueError("Filter not recognised")

    def get_filter_array(self) -> np.ndarray:
        """Return the filter array in the frequency domain.

        Returns
        -------
        numpy.ndarray
            An array containing the filter values

        Notes
        -----
        The filter length N is 2^self.fft_order.

        The indices of the array are interpreted as:

        - [0] The DC frequency component
        - [1:N/2] positive frequencies
        - [N/2:N-1] negative frequencies

        The array can be modified and passed back using set_filter()

        Notes
        -----
        Filter reference in frequency domain:
        Eq. 1.12 - 1.15 T. M. Buzug. Computed Tomography: From Photon Statistics to Modern Cone-Beam CT. Berlin: Springer, 2008.

        Plantagie, L. Algebraic filters for filtered backprojection, 2017
        https://hdl.handle.net/1887/48289
        """
        if self._filter == 'custom':
            return self._filter_array

        filter_length = 2**self.fft_order

        # frequency bins in cycles/pixel
        freq = fftfreq(filter_length)
        # in pi rad/pixel
        freq*=2

        ramp = abs(freq)
        ramp[ramp>self._filter_cutoff]=0

        if self._filter == 'ram-lak':
            filter_array = ramp
        if self._filter == 'shepp-logan':
            filter_array = ramp * np.sinc(freq/2)
        elif self._filter == 'cosine':
            filter_array = ramp * np.cos(freq*np.pi/2)
        elif self._filter == 'hamming':
            filter_array = ramp * (0.54 + 0.46 * np.cos(freq*np.pi))
        elif self._filter == 'hann':
            filter_array = ramp * (0.5 + 0.5 * np.cos(freq*np.pi))

        return np.asarray(filter_array,dtype=np.float32).reshape(2**self.fft_order)

    def reset(self) -> None:
        """Reset all optional configuration parameters to their default values."""
        self.set_filter()
        self.set_fft_order()
        self.set_filter_inplace()
        self.set_image_geometry()
        self._weights = None

    def _default_fft_order(self) -> int:
        min_order = 0

        while 2**min_order < self.acquisition_geometry.pixel_num_h * 2:
            min_order+=1

        min_order = max(8, min_order)
        return min_order

    def _calculate_weights(self, acquisition_geometry:AcquisitionGeometry) -> None:
        raise NotImplementedError()

    def _pre_filtering(self, acquisition_data:AcquisitionData) -> None:
        """Filter and weight the projections inplace.

        Parameters
        ----------
        acquisition_data
            The projections to be filtered

        Notes
        -----
        self.input is not used to allow processing in smaller chunks
        """
        if self._weights is None or self._weights.shape[0] != acquisition_data.geometry.pixel_num_v:
            self._calculate_weights(acquisition_data.geometry)

        if self._weights.shape[1] != acquisition_data.shape[-1]: #horizontal
            raise ValueError("Weights not compatible")

        filter_array = self.get_filter_array()
        if filter_array.size != 2**self.fft_order:
            raise ValueError("Custom filter has length {0} and is not compatible with requested fft_order {1}. Expected filter length 2^{1}"\
                            .format(filter_array.size,self.fft_order))

        #call ext function
        data_ptr = acquisition_data.array.ctypes.data_as(c_float_p)
        filter_ptr = filter_array.ctypes.data_as(c_float_p)
        weights_ptr = self._weights.ctypes.data_as(c_float_p)

        ag = acquisition_data.geometry
        if ag.dimension_labels == ('angle','vertical','horizontal'):
            cilacc.filter_projections_avh(data_ptr, filter_ptr, weights_ptr, self.fft_order, *acquisition_data.shape)
        elif ag.dimension_labels == ('vertical','angle','horizontal'):
            cilacc.filter_projections_vah(data_ptr, filter_ptr, weights_ptr, self.fft_order, *acquisition_data.shape)
        elif ag.dimension_labels == ('angle','horizontal'):
            cilacc.filter_projections_vah(data_ptr, filter_ptr, weights_ptr, self.fft_order, 1, *acquisition_data.shape)
        elif ag.dimension_labels == ('vertical','horizontal'):
            cilacc.filter_projections_avh(data_ptr, filter_ptr, weights_ptr, self.fft_order, 1, *acquisition_data.shape)
        else:
            raise ValueError ("Could not determine correct function call from dimension labels")


class FDK(GenericFilteredBackProjection):
    """Create an FDK reconstructor based on your cone-beam acquisition data using TIGRE as a backend.

    Parameters
    ----------
    input
        The input data to reconstruct. The reconstructor is set-up based on the geometry of the data.

    image_geometry
        A description of the area/volume to reconstruct

    filter
        The filter to be applied. Can be a string from: {'`ram-lak`', '`shepp-logan`', '`cosine`', '`hamming`', '`hann`'}, or a numpy array.

    Example
    -------
    >>> from cil.utilities.dataexample import SIMULATED_CONE_BEAM_DATA
    >>> from cil.recon import FDK
    >>> data = SIMULATED_CONE_BEAM_DATA.get()
    >>> fdk = FDK(data)
    >>> out = fdk.run()

    Notes
    -----
    The reconstructor can be further customised using additional 'set' methods provided.
    """

    supported_backends = ['tigre']

    def __init__(self, input:AcquisitionData, image_geometry:Optional[ImageGeometry]=None, filter:str="ram-lak", backend:Literal["tigre"]="tigre"):
        #call parent initialiser
        super().__init__(input, image_geometry, filter, backend='tigre')

        if  input.geometry.geom_type != AcquisitionGeometry.CONE:
            raise TypeError("This reconstructor is for cone-beam data only.")

    def run(self, out:Optional[ImageData]=None, verbose:int=1) -> Optional[ImageData]:
        """Run the configured FDK recon and returns the reconstruction.

        Parameters
        ----------
        out
           Fills the referenced ImageData with the reconstructed volume and suppresses the return
        verbose
           Controls the verbosity of the reconstructor. 0: No output is logged, 1: Full configuration is logged

        Returns
        -------
        ImageData
            The reconstructed volume. Suppressed if `out` is passed
        """
        if verbose:
            print(self)

        if self.filter_inplace is False:
            proj_filtered = self.input.copy()
        else:
            proj_filtered = self.input

        self._pre_filtering(proj_filtered)
        operator = self._PO_class(self.image_geometry,self.acquisition_geometry,adjoint_weights='FDK')

        if out is None:
            return operator.adjoint(proj_filtered)
        else:
            operator.adjoint(proj_filtered, out = out)

    def _calculate_weights(self, acquisition_geometry:AcquisitionGeometry) -> None:
        ag = acquisition_geometry
        xv = np.arange(-(ag.pixel_num_h -1)/2,(ag.pixel_num_h -1)/2 + 1,dtype=np.float32) * ag.pixel_size_h
        yv = np.arange(-(ag.pixel_num_v -1)/2,(ag.pixel_num_v -1)/2 + 1,dtype=np.float32) * ag.pixel_size_v
        (yy, xx) = np.meshgrid(xv, yv)

        principal_ray_length = ag.dist_source_center + ag.dist_center_detector
        scaling = 0.25 * ag.magnification * (2 * np.pi/ ag.num_projections) / ag.pixel_size_h
        self._weights = scaling * principal_ray_length / np.sqrt((principal_ray_length ** 2 + xx ** 2 + yy ** 2))

    def __str__(self) -> str:
        """Return a string representation of the FDK reconstructor.

        Returns
        -------
        str
            String representation of the FDK reconstructor
        """
        repres = "FDK recon\n"

        repres += self._str_data_size()

        repres += "\nReconstruction Options:\n"
        repres += "\tBackend: {}\n".format(self._backend)
        repres += "\tFilter: {}\n".format(self._filter)
        if self._filter != 'custom':
            repres += "\tFilter cut-off frequency: {}\n".format(self._filter_cutoff)
        repres += "\tFFT order: {}\n".format(self._fft_order)
        repres += "\tFilter_inplace: {}\n".format(self._filter_inplace)

        return repres


class FBP(GenericFilteredBackProjection):
    """Create an FBP reconstructor based on your parallel-beam acquisition data.

    Parameters
    ----------
    input : AcquisitionData
        The input data to reconstruct. The reconstructor is set-up based on the geometry of the data.

    image_geometry : ImageGeometry, default used if None
        A description of the area/volume to reconstruct

    filter : string, numpy.ndarray, default='ram-lak'
        The filter to be applied. Can be a string from: {'`ram-lak`', '`shepp-logan`', '`cosine`', '`hamming`', '`hann`'}, or a numpy array.

    backend : string
        The backend to use, can be 'astra' or 'tigre'. Data must be in the correct order for requested backend.

    Example
    -------
    >>> from cil.utilities.dataexample import SIMULATED_PARALLEL_BEAM_DATA
    >>> from cil.recon import FBP
    >>> data = SIMULATED_PARALLEL_BEAM_DATA.get()
    >>> fbp = FBP(data)
    >>> out = fbp.run()

    Notes
    -----
    The reconstructor can be further customised using additional 'set' methods provided.
    """

    supported_backends = ['tigre', 'astra']

    @property
    def slices_per_chunk(self) -> int:
        """Return the number of slices used to chunk data.

        Returns
        -------
        int
            Number of slices used to chunk data
        """
        return self._slices_per_chunk

    def __init__(self, input:AcquisitionData, image_geometry:Optional[ImageGeometry]=None, filter:str="ram-lak", backend:Literal["tigre"]="tigre"):
        super().__init__(input, image_geometry, filter, backend)
        self.set_split_processing(False)

        if  input.geometry.geom_type != AcquisitionGeometry.PARALLEL:
            raise TypeError("This reconstructor is for parallel-beam data only.")

    def set_split_processing(self, slices_per_chunk:int=0) -> None:
        """Split the processing in to chunks. Default, 0 will process the data in a single call.

        Parameters
        ----------
        out : slices_per_chunk, optional
            Process the data in chunks of n slices. It is recommended to use value of power-of-two.

        Notes
        -----
        This will reduce memory use but may increase computation time.
        It is recommended to tune it too your hardware requirements using 8, 16 or 32 slices.

        This can only be used on simple and offset data-geometries.
        """
        try:
            num_slices = int(slices_per_chunk)
        except:
            num_slices = 0

        if  num_slices >= self.acquisition_geometry.pixel_num_v:
            num_slices = self.acquisition_geometry.pixel_num_v

        self._slices_per_chunk = num_slices

    def run(self, out:Optional[ImageData]=None, verbose:int=1) -> Optional[ImageData]:
        """Run the configured FBP recon and returns the reconstruction.

        Parameters
        ----------
        out
           Fills the referenced ImageData with the reconstructed volume and suppresses the return

        verbose
           Controls the verbosity of the reconstructor. 0: No output is logged, 1: Full configuration is logged

        Returns
        -------
        ImageData
            The reconstructed volume. Suppressed if `out` is passed
        """
        if verbose:
            print(self)

        if self.slices_per_chunk:

            if self.acquisition_geometry.dimension == '2D':
                raise ValueError("Only 3D datasets can be processed in chunks with `set_split_processing`")
            elif self.acquisition_geometry.system_description == 'advanced':
                raise ValueError("Only simple and offset geometries can be processed in chunks with `set_split_processing`")
            elif self.acquisition_geometry.get_ImageGeometry() != self.image_geometry:
                raise ValueError("Only default image geometries can be processed in chunks `set_split_processing`")

            if out is None:
                ret = self.image_geometry.allocate()
            else:
                ret = out

            if self.filter_inplace:
                self._pre_filtering(self.input)

            tot_slices = self.acquisition_geometry.pixel_num_v
            remainder = tot_slices % self.slices_per_chunk
            num_chunks = int(np.ceil(self.image_geometry.shape[0] / self._slices_per_chunk))

            if verbose:
                pbar = tqdm(total=num_chunks)

            #process dataset by requested chunk size
            self._setup_PO_for_chunks(self.slices_per_chunk)
            for i in range(0, tot_slices-remainder, self.slices_per_chunk):

                if 'bottom' in self.acquisition_geometry.config.panel.origin:
                    start = i
                    end = i + self.slices_per_chunk
                else:
                    start = tot_slices -i - self.slices_per_chunk
                    end = tot_slices - i

                ret.array[start:end,:,:] = self._process_chunk(i, self.slices_per_chunk)

                if verbose:
                    pbar.update(1)

            #process excess rows
            if remainder:
                self._setup_PO_for_chunks(remainder)

                if 'bottom' in self.acquisition_geometry.config.panel.origin:
                    start = tot_slices-remainder
                    end = tot_slices
                else:
                    start = 0
                    end = remainder

                ret.array[start:end,:,:] = self._process_chunk(i, remainder)

                if verbose:
                    pbar.update(1)

            if verbose:
                pbar.close()

            if out is None:
                return ret

        else:

            if self.filter_inplace is False:
                proj_filtered = self.input.copy()
            else:
                proj_filtered = self.input

            self._pre_filtering(proj_filtered)

            operator = self._PO_class(self.image_geometry,self.acquisition_geometry)

            if out is None:
                return operator.adjoint(proj_filtered)
            else:
                operator.adjoint(proj_filtered, out = out)

    def reset(self) -> None:
        """Reset all optional configuration parameters to their default values."""
        super().reset()
        self.set_split_processing(0)

    def _calculate_weights(self, acquisition_geometry:AcquisitionGeometry) -> None:
        ag = acquisition_geometry
        scaling = 0.25 * (2 * np.pi/ ag.num_projections) / ag.pixel_size_h

        if self.backend=='astra':
            scaling /=  ag.pixel_size_v
        self._weights = np.full((ag.pixel_num_v,ag.pixel_num_h),scaling,dtype=np.float32)

    def _setup_PO_for_chunks(self, num_slices:int) -> None:
        if num_slices > 1:
            ag_slice = self.acquisition_geometry.copy()
            ag_slice.pixel_num_v = num_slices
        else:
            ag_slice = self.acquisition_geometry.get_slice(vertical=0)

        ig_slice = ag_slice.get_ImageGeometry()
        self.data_slice = ag_slice.allocate()
        self.operator = self._PO_class(ig_slice,ag_slice)

    def _process_chunk(self, i:int, step:int) -> np.ndarray:
        self.data_slice.fill(np.squeeze(self.input.array[:,i:i+step,:]))
        if not self.filter_inplace:
            self._pre_filtering(self.data_slice)

        return self.operator.adjoint(self.data_slice).array

    def __str__(self) -> str:
        """Return a string representation of the FBP reconstructor.

        Returns
        -------
        str
            String representation of FBP
        """
        repres = "FBP recon\n"

        repres += self._str_data_size()

        repres += "\nReconstruction Options:\n"
        repres += "\tBackend: {}\n".format(self._backend)
        repres += "\tFilter: {}\n".format(self._filter)
        if self._filter != 'custom':
            repres += "\tFilter cut-off frequency: {}\n".format(self._filter_cutoff)
        repres += "\tFFT order: {}\n".format(self._fft_order)
        repres += "\tFilter_inplace: {}\n".format(self._filter_inplace)
        repres += "\tSplit processing: {}\n".format(self._slices_per_chunk)

        if self._slices_per_chunk:
            num_chunks = int(np.ceil(self.image_geometry.shape[0] / self._slices_per_chunk))
        else:
            num_chunks = 1

        repres +="\nReconstructing in {} chunk(s):\n".format(num_chunks)

        return repres
