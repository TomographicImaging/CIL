#  Copyright 2023 United Kingdom Research and Innovation
#  Copyright 2023 The University of Manchester
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

from cil.framework import cilacc

import numpy as np


class Binner_IPP(object):

   def __init__(self, shape_in, shape_out, start_index, binning):
      """This constructs a configured cilacc Binner object used by cil.processors.Binner. This class should not be used directly.

      This performs no checks on the inputs. Expects input for 4 dimensions.

      Parameters
      ----------

      shape_in : list
         Input array shape. List of 4 integers corresponding to each dimension. i.e. [1,1,512,512]

      shape_out : list
         Output array shape. List of 4 integers corresponding to each dimension. i.e. [1,1,246,246]

      start_index : list
         Starting pixel index for cropping. List of 4 integers corresponding to each dimension. i.e. [0,0,1,1]

      binning : list
         Number of pixels to average together in each direction. List of 4 integers corresponding to each dimension. i.e. [1,1,2,2]

      """


      shape_in_arr = np.array(shape_in, np.uintp)
      shape_out_arr = np.array(shape_out, np.uintp)
      start_index_arr = np.array(start_index, np.uintp)
      binning_arr = np.array(binning, np.uintp)

      for i in range(4):
         if shape_out_arr[i] * binning_arr[i] + start_index_arr[i] > shape_in_arr[i]:
            raise ValueError("Input dimension mismatch on dimension {0}".format(i))

      self.obj = cilacc.Binner_new(shape_in_arr, shape_out_arr, start_index_arr, binning_arr)


   def bin(self, array_in, array_binned):
      """This bins the input array and writes the result in array_binned.

      This  performs no checks on the inputs.

      Parameters
      ----------

      array_in : ndarray
         Must have shape corresponding to shape_in. Data type float32. C ordered, contiguous memory

      array_binned : ndarray
         Must have shape corresponding to shape_out. Data type float32. C ordered, contiguous memory
      """

      return cilacc.Binner_bin(self.obj, array_in, array_binned)

   def __del__(self):
      """This deletes the cilacc Binner object
      """

      if hasattr(self,'obj'):
         cilacc.Binner_delete(self.obj)
