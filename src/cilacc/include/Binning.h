//  Copyright 2023 United Kingdom Research and Innovation
//  Copyright 2023 The University of Manchester
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
// Authors:
// CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using Shape = nb::ndarray<const size_t>;
using DataInput = nb::ndarray<const float>;
using DataBinned = nb::ndarray<float>;

void Binner_delete(void* binner);
void* Binner_new(Shape shape_in, 
		Shape shape_out, 
		Shape pixel_index_start, 
		Shape binning_list
);
int Binner_bin(void* binner, DataInput data_in, DataBinned data_binned);

