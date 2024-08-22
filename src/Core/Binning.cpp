//  Copyright 2023 United Kingdom Research and Innovation
//  Copyright 2023 The University of Manchester
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

#include "Binning.h"
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cstddef>
#include "ipp.h"
#include <omp.h>
#include "utilities.h"

class Binner {

private:

	size_t shape_in[4];
	size_t shape_out[4];
	size_t pixel_index_start[4];
	size_t binning_list[4];

	int srcStep, dstStep;
	IppiSize srcSize, dstSize;
	IppiPoint dstOffset = { 0,0 };
	int bufSize = 0;
	IppiResizeSpec_32f* pSpec = NULL;
	int specSize = 0, initSize = 0;

	void bin_2D(const float* data_in, float* data_binned);
	void bin_4D(const float* data_in, float* data_binned);

public:
	Binner(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list);
	int bin(const float* data_in, float* data_binned);

	~Binner()
	{
		if (pSpec != NULL)
			ippsFree(pSpec);
	}

	Binner(Binner const&) = delete;
	Binner& operator=(Binner const&) = delete;
	Binner(Binner&&) = delete;
	Binner& operator=(Binner&&) = delete;

};


Binner::Binner(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list)
{
	memcpy(this->shape_in, shape_in, 4 * sizeof(size_t));
	memcpy(this->shape_out, shape_out, 4 * sizeof(size_t));
	memcpy(this->pixel_index_start, pixel_index_start, 4 * sizeof(size_t));
	memcpy(this->binning_list, binning_list, 4 * sizeof(size_t));

	srcStep = (int)shape_in[3] * sizeof(float);
	dstStep = (int)shape_out[3] * sizeof(float);

	srcSize.width = (int)(shape_out[3] * binning_list[3]);
	srcSize.height = (int)(shape_out[2] * binning_list[2]);

	dstSize.width = (int)shape_out[3];
	dstSize.height = (int)shape_out[2];

	ippiResizeGetSize_32f(srcSize, dstSize, ippSuper, 0, &specSize, &initSize);

	pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
	ippiResizeSuperInit_32f(srcSize, dstSize, pSpec);

	ippiResizeGetBufferSize_8u(pSpec, dstSize, 1, &bufSize);
}

int Binner::bin(const float* data_in, float* data_binned)
{
	if ((binning_list[0] > 1) || (binning_list[1] > 1))
	{
		Binner::bin_4D(data_in, data_binned);
	}
	else
	{
		Binner::bin_2D(data_in, data_binned);
	}
	return 0;
}

void Binner::bin_2D(const float* data_in, float* data_binned)
{
	/*bin last 2 dimensions in dataset*/

	size_t index_channel_in;
	long long k;

	for (int l = 0; l < this->shape_out[0]; l++)
	{

		index_channel_in = (l + pixel_index_start[0]) * shape_in[3] * shape_in[2] * shape_in[1];

#pragma omp parallel
		{
			Ipp8u* pBuffer = ippsMalloc_8u(bufSize);
			float* data_temp = new float[shape_out[2] * shape_out[3]];

#pragma omp for
			for (k = 0; k < shape_out[1]; k++)
			{
				size_t index_vol_in = index_channel_in + (k + pixel_index_start[1]) * shape_in[3] * shape_in[2] + pixel_index_start[2] * shape_in[3] + pixel_index_start[3];
				size_t index_vol_out = l * shape_out[3] * shape_out[2] * shape_out[1] + k * shape_out[3] * shape_out[2];

				ippiResizeSuper_32f_C1R(&data_in[index_vol_in], srcStep, &data_binned[index_vol_out], dstStep, dstOffset, dstSize, pSpec, pBuffer);
			}

			if (pBuffer != NULL)
				ippsFree(pBuffer);
		}
	}

}

void Binner::bin_4D(const float* data_in, float* data_binned)
{
	/*bin up to 4 dimensions in dataset*/

	size_t index_channel_in;
	long long k;

	for (int l = 0; l < shape_out[0]; l++)
	{

#pragma omp parallel for
		for (k = 0; k < shape_out[1]; k++)
		{
			ippiSet_32f_C1R(0, &data_binned[l * shape_out[3] * shape_out[2] * shape_out[1] + k * shape_out[2] * shape_out[3]], dstStep, dstSize);
		}


		for (int bl = 0; bl < binning_list[0]; bl++)
		{
			index_channel_in = (l * binning_list[0] + bl + pixel_index_start[0]) * shape_in[3] * shape_in[2] * shape_in[1];

#pragma omp parallel
			{
				Ipp8u* pBuffer = ippsMalloc_8u(bufSize);
				float* data_temp = new float[shape_out[2] * shape_out[3]];

#pragma omp for
				for (k = 0; k < shape_out[1]; k++)
				{
					size_t index_vol_in = index_channel_in + (k * binning_list[1] + pixel_index_start[1]) * shape_in[3] * shape_in[2] + pixel_index_start[2] * shape_in[3] + pixel_index_start[3];
					size_t index_vol_out = l * shape_out[3] * shape_out[2] * shape_out[1] + k * shape_out[3] * shape_out[2];

					for (int bk = 0; bk < binning_list[1]; bk++)
					{
						ippiResizeSuper_32f_C1R(&data_in[index_vol_in], srcStep, data_temp, dstStep, dstOffset, dstSize, pSpec, pBuffer);
						ippiAdd_32f_C1IR(data_temp, dstStep, &data_binned[index_vol_out], dstStep, dstSize);
						index_vol_in += shape_in[3] * shape_in[2];
					}
				}

				delete[] data_temp;

				if (pBuffer != NULL)
					ippsFree(pBuffer);
			}

		}

		float denom = (float)binning_list[0] * binning_list[1];
#pragma omp parallel for
		for (k = 0; k < shape_out[1]; k++)
		{
			ippiDivC_32f_C1IR(denom, &data_binned[l * shape_out[3] * shape_out[2] * shape_out[1] + k * shape_out[3] * shape_out[2]], dstStep, dstSize);
		}
	}
}


extern "C"
{
	void Binner_delete(void* binner) { delete (Binner*)binner; }
	void* Binner_new(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list) { return new Binner(shape_in, shape_out, pixel_index_start, binning_list); }
	int Binner_bin(void* binner, const float* data_in, float* data_binned) { return ((Binner*)binner)->bin(data_in, data_binned); }
}
