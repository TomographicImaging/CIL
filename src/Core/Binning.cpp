#include "Binning.h"

#include <iostream>
#include <stdio.h>
#include <ipp.h>
#include <ipps.h>
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

	//std::cout << "Initialising binner" << std::endl;


	memcpy(this->shape_in, shape_in, 4 * sizeof(size_t));
	memcpy(this->shape_out, shape_out, 4 * sizeof(size_t));
	memcpy(this->pixel_index_start, pixel_index_start, 4 * sizeof(size_t));
	memcpy(this->binning_list, binning_list, 4 * sizeof(size_t));

	//std::cout << "shape_in\t" << this->shape_in[0] << '\t' << this->shape_in[1] << '\t' << this->shape_in[2] << '\t' << this->shape_in[3] << std::endl;
	//std::cout << "shape_out\t" << this->shape_out[0] << '\t' << this->shape_out[1] << '\t' << this->shape_out[2] << '\t' << this->shape_out[3] << std::endl;
	//std::cout << "pixel_index_start\t" << this->pixel_index_start[0] << '\t' << this->pixel_index_start[1] << '\t' << this->pixel_index_start[2] << '\t' << this->pixel_index_start[3] << std::endl;
	//std::cout << "binning_list\t" << this->binning_list[0] << '\t' << this->binning_list[1] << '\t' << this->binning_list[2] << '\t' << this->binning_list[3] << std::endl;


	srcStep = (int)shape_in[3] * sizeof(float);
	dstStep = (int)shape_out[3] * sizeof(float);

	//std::cout << "srcStep\t" << srcStep << std::endl;
	//std::cout << "dstStep\t" << dstStep << std::endl;

	srcSize.width = (int)(shape_out[3] * binning_list[3]);
	srcSize.height = (int)(shape_out[2] * binning_list[2]);

	//std::cout << "srcSize\t" << srcSize.width << '\t' << srcSize.height << std::endl;

	dstSize.width = (int)shape_out[3];
	dstSize.height = (int)shape_out[2];

	//std::cout << "dstSize:\t" << dstSize.width << '\t' << dstSize.height << std::endl;

	ippiResizeGetSize_32f(srcSize, dstSize, ippSuper, 0, &specSize, &initSize);

	//std::cout << "specSize\t" << specSize << std::endl;
	//std::cout << "initSize\t" << initSize << std::endl;


	pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
	ippiResizeSuperInit_32f(srcSize, dstSize, pSpec);

	ippiResizeGetBufferSize_8u(pSpec, dstSize, 1, &bufSize);

	//std::cout << "bufSize\t" << bufSize << std::endl;

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
	DLL_EXPORT void Binner_delete(void* binner) { delete (Binner*)binner; }
	DLL_EXPORT void* Binner_new(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list) { return new Binner(shape_in, shape_out, pixel_index_start, binning_list); }
	DLL_EXPORT int Binner_bin(void* binner, const float* data_in, float* data_binned) { return ((Binner*)binner)->bin(data_in, data_binned); }
}