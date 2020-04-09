#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"
#include "utilities.h"


DLL_EXPORT int saxpby(float * x, float * y, float * out, float a, float b, long size, int nThreads);
DLL_EXPORT int daxpby(double * x, double * y, double * out, double a, double b, long size, int nThreads);
