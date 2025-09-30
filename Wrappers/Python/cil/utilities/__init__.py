#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
def dtype_like(input_value, reference_array):
    """`input_value.astype(reference_array.dtype, copy=False)` with fallback to `input_value`"""
    if hasattr(reference_array, 'dtype'):
        if hasattr(input_value, 'astype'):
            return input_value.astype(reference_array.dtype, copy=False)
        return reference_array.dtype.type(input_value)
    return input_value
