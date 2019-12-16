#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"


DLL_EXPORT int padd(float * x, float * y, float * out, long size);
DLL_EXPORT int psubtract(float * x, float * y, float * out, long size);
DLL_EXPORT int pmultiply(float * x, float * y, float * out, long size);
DLL_EXPORT int pdivide(float * x, float * y, float * out, long size, float default_value);
DLL_EXPORT int ppower(float * x, float * y, float * out, long size);
DLL_EXPORT int pminimum(float * x, float * y, float * out, long size);
DLL_EXPORT int pmaximum(float * x, float * y, float * out, long size);

DLL_EXPORT int saxpby(float * x, float * y, float * out, float a, float b, long size);
DLL_EXPORT int daxpby(double * x, double * y, double * out, double a, double b, long size);
