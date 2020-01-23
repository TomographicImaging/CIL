#include "utilities.h"


void threads_setup(int nThreads_requested, int *nThreads_current)
{
#pragma omp parallel
	{
		if (omp_get_thread_num() == 0)
		{
			*nThreads_current = omp_get_num_threads();
		}
	}
	omp_set_num_threads(nThreads_requested);
}
