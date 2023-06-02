// -*- coding: utf-8 -*-
//  Copyright 2021 United Kingdom Research and Innovation
//  Copyright 2021 The University of Manchester
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

#include <iostream>
#include <stdio.h>
#include <ipp.h>
#include <ipps.h>
#include <chrono>
#include <omp.h>
#include <random>
#include "dll_export.h"
#include "utilities.h"


#ifdef __cplusplus
extern "C" {
#endif
	DLL_EXPORT int filter_projections_avh(float* data, const float* filter, const float* weights, int order, long num_proj, long pix_y, long pix_x);
	DLL_EXPORT int filter_projections_vah(float* data, const float* filter, const float* weights, int order, long pix_y, long num_proj, long pix_x);
#ifdef __cplusplus
}
#endif