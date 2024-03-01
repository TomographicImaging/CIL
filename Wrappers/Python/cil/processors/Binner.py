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

from cil.processors import Slicer
import numpy as np

try:
    from cil.processors.cilacc_binner import Binner_IPP
    has_ipp = True
except:
    has_ipp = False

# Note to developers: Binner and Slicer share a lot of common code
# so Binner has been implemented as a child of Slicer. This makes use
# of commonality and redefines only the methods that differ. These methods
# dictate the style of slicer
class Binner(Slicer):

    """This creates a Binner processor.
    
    The processor will crop the data, and then average together n input pixels along a dimension from the starting index.

    The output will be a data container with the data, and geometry updated to reflect the operation.

    Parameters
    ----------

    roi : dict
        The region-of-interest to bin {'axis_name1':(start,stop,step), 'axis_name2':(start,stop,step)}
        The `key` being the axis name to apply the processor to, the `value` holding a tuple containing the ROI description

        Start: Starting index of input data. Must be an integer, or `None` defaults to index 0.
        Stop: Stopping index of input data. Must be an integer, or `None` defaults to index N.
        Step: Number of pixels to average together. Must be an integer or `None` defaults to 1.

    accelerated : boolean, default=True
        Uses the CIL accelerated backend if `True`, numpy if `False`.


    Example
    -------
    
    >>> from cil.processors import Binner
    >>> roi = {'horizontal':(10,-10,2),'vertical':(10,-10,2)}
    >>> processor = Binner(roi)
    >>> processor.set_input(data)
    >>> data_binned = processor.get_output()


    Example
    -------
    >>> from cil.processors import Binner
    >>> roi = {'horizontal':(None,None,2),'vertical':(None,None,2)}
    >>> processor = Binner(roi)
    >>> processor.set_input(data.geometry)
    >>> geometry_binned = processor.get_output()


    Note
    ----
    The indices provided are start inclusive, stop exclusive.

    All elements along a dimension will be included if the axis does not appear in the roi dictionary, or if passed as {'axis_name',-1}
    
    If only one number is provided, then it is interpreted as Stop. i.e. {'axis_name1':(stop)}
    If two numbers are provided, then they are interpreted as Start and Stop  i.e. {'axis_name1':(start, stop)}

    Negative indexing can be used to specify the index. i.e. {'axis_name1':(10, -10)} will crop the dimension symmetrically
    
    If Stop - Start is not multiple of Step, then 
    the resulted dimension will have (Stop - Start) // Step 
    elements, i.e. (Stop - Start) % Step elements will be ignored

    """

    def __init__(self,
                 roi = None, accelerated=True):

        if accelerated and not has_ipp:
            raise RuntimeError("Cannot run accelerated Binner without the IPP libraries.")

        super(Binner,self).__init__(roi = roi)
        self._accelerated = True


    def _configure(self):
        """
        Once the ROI has been parsed this configures the input specifically for use with Binner        
        """

        #as binning we only include bins that are inside boundaries
        self._shape_out = [int((x.stop - x.start)//x.step) for x in self._roi_ordered]
        self._pixel_indices = []

        # fix roi_ordered for binner based on shape out
        for i in range(4):
            start = self._roi_ordered[i].start
            stop = self._roi_ordered[i].start + self._shape_out[i] * self._roi_ordered[i].step

            self._roi_ordered[i] = range(
                start,
                stop,
                self._roi_ordered[i].step
                )

            self._pixel_indices.append((start, stop-1))


    def _get_slice_position(self, roi):
        """
        Return the vertical position to extract a single slice for binned geometry
        """
        return roi.start + roi.step/2


    def _get_angles(self, roi):
        """
        Returns the binned angles according to the roi
        """
        n_elements = len(roi)
        shape = (n_elements, roi.step)
        return self._geometry.angles[roi.start:roi.start+n_elements*roi.step].reshape(shape).mean(1)


    def _bin_array_numpy(self, array_in, array_binned):
        """
        Bins the array using numpy. This method is slower and less memory efficient than self._bin_array_acc
        """
        shape_object = []
        slice_object = []

        for i in range(4):
            # reshape the data to add each 'bin' dimensions
            shape_object.append(self._shape_out[i]) 
            shape_object.append(self._roi_ordered[i].step)

        shape_object = tuple(shape_object)
        slice_object = tuple([slice(x.start, x.stop) for x in self._roi_ordered])
      
        data_resized = array_in.reshape(self._shape_in)[slice_object].reshape(shape_object)

        mean_order = (-1, 1, 2, 3)
        for i in range(4):
            data_resized = data_resized.mean(mean_order[i])
            
        np.copyto(array_binned, data_resized)


    def _bin_array_acc(self, array_in, array_binned):
        """
        Bins the array using the accelerated CIL backend
        """
        indices_start = [x.start for x in self._roi_ordered]
        bins = [x.step for x in self._roi_ordered]

        binner_ipp = Binner_IPP(self._shape_in, self._shape_out, indices_start, bins)

        res = binner_ipp.bin(array_in, array_binned)
        if res != 0:
            raise RuntimeError("Call failed")


    def _process_data(self, dc_in, dc_out):
        """
        Bin the data array
        """
        if self._accelerated:
            self._bin_array_acc(dc_in.array, dc_out.array)
        else:
            self._bin_array_numpy(dc_in.array, dc_out.array)

