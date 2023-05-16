// -*- coding: utf-8 -*-
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

int fdiff_direct_neumann(const float *inimagefull, float *outimageXfull, float *outimageYfull, float *outimageZfull, float *outimageCfull, size_t nx, size_t ny, size_t nz, size_t nc);
int fdiff_direct_periodic(const float *inimagefull, float *outimageXfull, float *outimageYfull, float *outimageZfull, float *outimageCfull, size_t nx, size_t ny, size_t nz, size_t nc);
int fdiff_adjoint_neumann(float *outimagefull, const float *inimageXfull, const float *inimageYfull, const float *inimageZfull, const float *inimageCfull, size_t nx, size_t ny, size_t nz, size_t nc);
int fdiff_adjoint_periodic(float *outimagefull, const float *inimageXfull, const float *inimageYfull, const float *inimageZfull, const float *inimageCfull, size_t nx, size_t ny, size_t nz, size_t nc);

#ifdef __cplusplus
extern "C" {
#endif

DLL_EXPORT int openMPtest(int nThreads);
DLL_EXPORT int fdiff4D(float *imagefull, float *gradCfull, float *gradZfull, float *gradYfull, float *gradXfull, size_t nc, size_t nz, size_t ny, size_t nx, int boundary, int direction, int nThreads);
DLL_EXPORT int fdiff3D(float *imagefull, float *gradZfull, float *gradYfull, float *gradXfull, size_t nz, size_t ny, size_t nx, int boundary, int direction, int nThreads);
DLL_EXPORT int fdiff2D(float *imagefull, float *gradYfull, float *gradXfull, size_t ny, size_t nx, int boundary, int direction, int nThreads);

#ifdef __cplusplus
}
#endif
