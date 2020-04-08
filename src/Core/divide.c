#include "divide.h"


DLL_EXPORT int fdivide(float * x, float * y, float * out, float default_value, int is_zero_by_zero, long size, int nThreads){
    long i = 0;

    int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        if ( * (y + i) == 0. ) {
            if (is_zero_by_zero == 0){
                // handle division by 0
                * (out + i) = default_value;
            } else {
                if ( * ( x + i ) == 0. ) {
                    * (out + i) = default_value;
                } else {
                    * (out + i) = NPY_INFINITYF;
                }
            }
        }
        else {
            *(out + i ) = ( *(x + i) ) / ( *(y + i) ); 
        }
    }
}
    omp_set_num_threads(nThreads_initial);
    return 0;
    
}

DLL_EXPORT int ddivide(double * x, double * y, double * out, double default_value, int is_zero_by_zero, long size, int nThreads){
    long i = 0;

    int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        if ( * (y + i) == 0. ) {
            if (is_zero_by_zero == 0){
                // handle division by 0
                * (out + i) = default_value;
            } else {
                if ( * ( x + i ) == 0. ) {
                    * (out + i) = default_value;
                } else {
                    * (out + i) = NPY_INFINITY ;
                }
            }
        }
        else {
            *(out + i ) = ( *(x + i) ) / ( *(y + i) ); 
        }
    }
}
    omp_set_num_threads(nThreads_initial);
    return 0;
    
}


DLL_EXPORT int idivide(int * x, int * y, int * out, int default_value, int is_zero_by_zero, long size, int nThreads){
    long i = 0;

    int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        if ( * (y + i) == 0. ) {
            if (is_zero_by_zero == 0){
                // handle division by 0
                * (out + i) = default_value;
            } else {
                if ( * ( x + i ) == 0. ) {
                    * (out + i) = default_value;
                } else {
                    * (out + i) = NPY_INFINITYL ;
                }
            }
        }
        else {
            *(out + i ) = ( *(x + i) ) / ( *(y + i) ); 
        }
    }
}
    omp_set_num_threads(nThreads_initial);
    return 0;
    
}
