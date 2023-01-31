#ifndef _BINNER_H_
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
