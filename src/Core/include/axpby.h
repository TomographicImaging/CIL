#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"
#include "utilities.h"

int saxpby_asbs(const float * x, const float * y, float * out, float a, float b, int64_t size, int nThreads);
int saxpby_avbv(const float * x, const float * y, float * out, const float * a, const float * b, int64_t size, int nThreads);
int saxpby_asbv(const float * x, const float * y, float * out, float a, const float * b, int64_t size, int nThreads);
int daxpby_asbs(const double * x, const double * y, double * out, double a, double b, int64_t size, int nThreads);
int daxpby_avbv(const double * x, const double * y, double * out, const double * a, const double * b, int64_t size, int nThreads);
int daxpby_asbv(const double * x, const double * y, double * out, double a, const double * b, int64_t size, int nThreads);

DLL_EXPORT int saxpby(const float * x, const float * y, float * out, const float * a, int type_a, const float * b, int type_b, long size, int nThreads);
DLL_EXPORT int daxpby(const double * x, const double * y, double * out, const double * a, int type_a, const double * b, int type_b, long size, int nThreads);
