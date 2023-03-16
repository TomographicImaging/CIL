// -*- coding: utf-8 -*-
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

#define _BINNER_H_

#include <cstddef>
#include "dll_export.h"

extern "C"
{
    DLL_EXPORT void Binner_delete(void* binner);
    DLL_EXPORT void* Binner_new(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list);
    DLL_EXPORT int Binner_bin(void* binner, const float* data_in, float* data_binned);
}
#endif
