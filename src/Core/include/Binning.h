#include <iostream>
#include <stdio.h>
#include <ipp.h>
#include <ipps.h>
#include <omp.h>
#include "dll_export.h"
#include "utilities.h"


void bin_2D(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list);
void bin_4D(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list);

#ifdef __cplusplus
extern "C" {
#endif
	DLL_EXPORT int bin_ipp(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list);
#ifdef __cplusplus
}
#endif
