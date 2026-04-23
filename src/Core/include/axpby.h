//  Copyright 2019 United Kingdom Research and Innovation
//  Copyright 2019 The University of Manchester
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
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"
#include "utilities.h"

int saxpby_asbs(const float * x, const float * y, float * out, float a, float b, int64 size, int nThreads);
int saxpby_avbv(const float * x, const float * y, float * out, const float * a, const float * b, int64 size, int nThreads);
int saxpby_asbv(const float * x, const float * y, float * out, float a, const float * b, int64 size, int nThreads);
int daxpby_asbs(const double * x, const double * y, double * out, double a, double b, int64 size, int nThreads);
int daxpby_avbv(const double * x, const double * y, double * out, const double * a, const double * b, int64 size, int nThreads);
int daxpby_asbv(const double * x, const double * y, double * out, double a, const double * b, int64 size, int nThreads);

#ifdef __cplusplus
extern "C" {
#endif

DLL_EXPORT int saxpby(const float * x, const float * y, float * out, const float * a, int type_a, const float * b, int type_b, int64 size, int nThreads);
DLL_EXPORT int daxpby(const double * x, const double * y, double * out, const double * a, int type_a, const double * b, int type_b, int64 size, int nThreads);

#ifdef __cplusplus
}
#endif
