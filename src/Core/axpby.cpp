// -*- coding: utf-8 -*-
//  Copyright 2019 United Kingdom Research and Innovation
//  Copyright 2019 The University of Manchester
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
// Authors:
// CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt


#include "axpby.h"


int saxpby_asbs(const float * x, const float * y, float * out, float a, float b, int64 size, int nThreads)
{
	int64 i = 0;

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = a * *(x + i) + b * *(y + i); 
    }
}
    return 0;
}

int saxpby_avbv(const float * x, const float * y, float * out, const float * a, const float * b, int64 size, int nThreads)
{
    int64 i = 0;

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = *(a + i) * *(x + i) + *(b + i) * *(y + i); 
    }
}
    return 0;    
}

int saxpby_asbv(const float * x, const float * y, float * out, float a, const float * b, int64 size, int nThreads)
{
    int64 i = 0;

#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = a * *(x + i) + *(b + i) * *(y + i); 
    }
}
    return 0; 
}

int daxpby_asbs(const double * x, const double * y, double * out, double a, double b, int64 size, int nThreads)
{
	int64 i = 0;
#pragma omp parallel
	{
#pragma omp for
		for (i = 0; i < size; i++)
		{
			*(out + i) = a * *(x + i) + b * *(y + i);
		}
	}
	return 0;
}
int daxpby_avbv(const double * x, const double * y, double * out, const double * a, const double * b, int64 size, int nThreads)
{
	int64 i = 0;
#pragma omp parallel
	{
#pragma omp for
		for (i = 0; i < size; i++)
		{
			*(out + i) = *(a + i) * *(x + i) + *(b + i) * *(y + i);
		}
	}
	return 0;
}
int daxpby_asbv(const double * x, const double * y, double * out, double a, const double * b, int64 size, int nThreads)
{
	int64 i = 0;
#pragma omp parallel
	{
#pragma omp for
		for (i = 0; i < size; i++)
		{
			*(out + i) = a * *(x + i) + *(b + i) * *(y + i);
		}
	}
	return 0;
}
DLL_EXPORT int saxpby(const float * x, const float * y, float * out, const float *a, int a_type, const float* b, int b_type, int64 size, int nThreads)
{
	//type = 0 float
	//type = 1 array of floats

	int64 i = 0;

	int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

	if (a_type == 0 && b_type == 0)
		saxpby_asbs(x, y, out, *a, *b, size, nThreads);
	else if (a_type == 1 && b_type == 1)
		saxpby_avbv(x, y, out, a, b, size, nThreads);
	else if (a_type == 0 && b_type == 1)
		saxpby_asbv(x, y, out, *a, b, size, nThreads);
	else if (a_type == 1 && b_type == 0)
		saxpby_asbv(y, x, out, *b, a, size, nThreads);

	omp_set_num_threads(nThreads_initial);

	return 0;
}
DLL_EXPORT int daxpby(const double * x, const double * y, double * out, const double *a, int a_type, const double* b, int b_type, int64 size, int nThreads)
{
	//type = 0 double
	//type = 1 array of double

	int64 i = 0;

	int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

	if (a_type == 0 && b_type == 0)
		daxpby_asbs(x, y, out, *a, *b, size, nThreads);
	else if (a_type == 1 && b_type == 1)
		daxpby_avbv(x, y, out, a, b, size, nThreads);
	else if (a_type == 0 && b_type == 1)
		daxpby_asbv(x, y, out, *a, b, size, nThreads);
	else if (a_type == 1 && b_type == 0)
		daxpby_asbv(y, x, out, *b, a, size, nThreads);

	omp_set_num_threads(nThreads_initial);

	return 0;
}