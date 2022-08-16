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
