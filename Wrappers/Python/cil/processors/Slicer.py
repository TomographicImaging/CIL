# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from .DownsampleBase import DownsampleBase
import numpy as np

class Slicer(DownsampleBase):

    """This creates a Slicer processor.
    
    The processor will crop the data, and then return every n input pixels along a dimension from the starting index.

    The output will be a data container with the data, and geometry updated to reflect the operation.

    Parameters
    ----------

    roi : dict
        The region-of-interest to bin {'axis_name1':(start,stop,step), 'axis_name2':(start,stop,step)}
        The `key` being the axis name to apply the processor to, the `value` holding a tuple containing the ROI description

        Start: Starting index of input data. Must be an integer, or `None` defaults to index 0.
        Stop: Stopping index of input data. Must be an integer, or `None` defaults to index N.
        Step: Number of pixels to average together. Must be an integer or `None` defaults to 1.

    force: boolean, default=False
        enforce slicing even if the returned geometry is not meaningful, will return a DataContainer and not a geometry


    Example
    -------
    
    >>> from cil.processors import Slicer
    >>> roi = {'horizontal':(10,-10,2),'vertical':(10,-10,2)}
    >>> processor = Slicer(roi)
    >>> processor.set_input(data)
    >>> data_binned = processor.get_output()


    Example
    -------
    >>> from cil.processors import Slicer
    >>> roi = {'horizontal':(None,None,2),'vertical':(None,None,2)}
    >>> processor = Slicer(roi)
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
                 roi = None, force=False):

        kwargs = {
            '_roi_input': roi,
            '_force':force, 
        }

        super(Slicer,self).__init__(**kwargs)
    

    def _configure(self):
        """
        Configure the input specifically for use with Slicer        
        """
        self._shape_out = [len(x) for x in self._roi_ordered]
        self._pixel_indices = [(x[0],x[-1]) for x in self._roi_ordered]


    def _process_acquisition_geometry(self):
        """
        Creates the sliced acquisition geometry
        """
        geometry_new = super(Slicer,self)._process_acquisition_geometry(type='Slice')
        return geometry_new


    def _process_data(self, dc_in, dc_out):
        slice_obj = tuple([slice(x.start, x.stop, x.step) for x in self._roi_ordered])
        arr_in = dc_in.array.reshape(self._shape_in)
        dc_out.fill(np.squeeze(arr_in[slice_obj]))

