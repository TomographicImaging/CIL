#include <iostream>
#include <stdio.h>
#include <ipp.h>
#include <ipps.h>
#include <ippcore.h>
#include <chrono>
#include <omp.h>
#include <random>
#include "dll_export.h"
#include "utilities.h"

ippInit() 
	
#ifdef __cplusplus
extern "C" {
#endif
	DLL_EXPORT int filter_projections_avh(float* data, const float* filter, const float* weights, int order, long num_proj, long pix_y, long pix_x);
	DLL_EXPORT int filter_projections_vah(float* data, const float* filter, const float* weights, int order, long pix_y, long num_proj, long pix_x);
#ifdef __cplusplus
}
#endif
