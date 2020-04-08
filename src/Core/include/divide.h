#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"
#include "utilities.h"
#include "Python.h"
#include "npy_math.h"


DLL_EXPORT int fdivide(float * x, float * y, float * out, float default_value, int is_zero_by_zero, long size, int nThreads);

DLL_EXPORT int ddivide(double * x, double * y, double * out, double default_value, int is_zero_by_zero, long size, int nThreads);

DLL_EXPORT int idivide(int * x, int * y, int * out, int default_value, int is_zero_by_zero, long size, int nThreads);
