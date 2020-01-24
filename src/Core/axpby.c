#include "axpby.h"


DLL_EXPORT int saxpby(float * x, float * y, float * out, float a, float b, long size, int nThreads){
    long i = 0;

    int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = a * ( *(x + i) ) + b * ( *(y + i) ); 
    }
}
    omp_set_num_threads(nThreads_initial);
    return 0;
    
}

DLL_EXPORT int daxpby(double * x, double * y, double * out, double a, double b, long size, int nThreads) {
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