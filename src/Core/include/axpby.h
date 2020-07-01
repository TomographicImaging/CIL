#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"
#include "utilities.h"

DLL_EXPORT int saxpby(float * x, float * y, float * out, float * a, int type_a, float * b, int type_b, long size, int nThreads);
DLL_EXPORT int daxpby(double * x, double * y, double * out, double * a, int type_a, double * b, int type_b, long size, int nThreads);
