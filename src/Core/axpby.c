#include "axpby.h"


DLL_EXPORT int padd(float * x, float * y, float * out, long size){
    long i = 0;
#pragma omp parallel for
    for (i=0; i < size; i++)
    {
        *(out + i ) = *(x + i) + *(y+i); 
    }
    return 0;
}

DLL_EXPORT int psubtract(float * x, float * y, float * out, long size){
    long i = 0;
#pragma omp parallel
{
//#pragma omp single
//{
//		printf("current number of threads %d\n", omp_get_num_threads());
//}
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = *(x + i) - *(y+i); 
    }
}
    return 0;
    
}

DLL_EXPORT int pmultiply(float * x, float * y, float * out, long size){
    long i = 0;
#pragma omp parallel for
    for (i=0; i < size; i++)
    {
        *(out + i ) = *(x + i) * *(y+i); 
    }
    return 0;
}

DLL_EXPORT int pdivide(float * x, float * y, float * out, long size, float default_value)
{
    long i = 0;
#pragma omp parallel for
    for (i=0; i < size; i++)
    {
        *(out + i ) = *(y+i) ? *(x + i) / *(y+i) : default_value;
    }
    return 0;
}
DLL_EXPORT int ppower(float * x, float * y, float * out, long size){
    long i = 0;
#pragma omp parallel for
    for (i=0; i < size; i++)
    {
        *(out + i ) = (float)pow(*(x + i) , *(y+i)) ; 
    }
    return 0;
}

DLL_EXPORT int pminimum(float * x, float * y, float * out, long size){
    long i = 0;
#pragma omp parallel for
    for (i=0; i < size; i++)
    {
        *(out + i ) = *(y+i) > (*x+i) ? *(x + i) :  *(y+i); 
    }
    return 0;
}

DLL_EXPORT int pmaximum(float * x, float * y, float * out, long size) {
	long i = 0;
#pragma omp parallel for
	for (i = 0; i < size; i++)
	{
		*(out + i) = *(y + i) < (*x + i) ? *(x + i) : *(y + i);
	}
	return 0;
}


DLL_EXPORT int saxpby(float * x, float * y, float * out, float a, float b, long size){
    long i = 0;
#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = a * ( *(x + i) ) + b * ( *(y + i) ); 
    }
}
    return 0;
    
}

DLL_EXPORT int daxpby(double * x, double * y, double * out, double a, double b, long size) {
	long i = 0;
#pragma omp parallel
	{
#pragma omp for
		for (i = 0; i < size; i++)
		{
			*(out + i) = a * (*(x + i)) + b * (*(y + i));
		}
	}
	return 0;

}