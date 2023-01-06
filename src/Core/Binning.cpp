#include "Binning.h"

int bin_ipp(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list)
{
	if ((binning_list[0] > 1) || (binning_list[1] > 1))
	{
		bin_4D(data_in, shape_in, data_binned, shape_out, pixel_index_start, binning_list);
	}
	else
	{
		bin_2D(data_in, shape_in, data_binned, shape_out, pixel_index_start, binning_list);
	}
	return 0;
}

void setup_binning_dimensions(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t *binning_list,  int* srcStep, int* dstStep, IppiSize* srcSize, IppiSize* dstSize)
{
	*srcStep = (int)shape_in[3] * sizeof(float);
	*dstStep = (int)shape_out[3] * sizeof(float);

	srcSize->width = (int)(pixel_index_start[3] + shape_out[3] * binning_list[3] - pixel_index_start[3]);
	srcSize->height = (int)(pixel_index_start[2] + shape_out[2] * binning_list[2] - pixel_index_start[2]);

	dstSize->width = (int)shape_out[3];
	dstSize->height = (int)shape_out[2];

}

void binning_ipp_init(IppiSize srcSize, IppiSize dstSize, int * bufSize, IppiResizeSpec_32f ** pSpec)
{

	int specSize = 0, initSize = 0;

	ippiResizeGetSize_32f(srcSize, dstSize, ippLinear, 0, &specSize, &initSize);

	*pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
	ippiResizeSuperInit_32f(srcSize, dstSize, *pSpec);

	ippiResizeGetBufferSize_8u(*pSpec, dstSize, 1, bufSize);
}



void bin_2D(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list)//, bool antialiasing)
{
	/*bin last 2 dimensions in dataset*/

	int srcStep, dstStep;
	IppiSize srcSize, dstSize;
	IppiPoint dstOffset = { 0,0 };

	setup_binning_dimensions(shape_in, shape_out, pixel_index_start, binning_list, &srcStep, &dstStep, &srcSize, &dstSize);

	int bufSize = 0;
	IppiResizeSpec_32f * pSpec = NULL;
	binning_ipp_init(srcSize, dstSize, &bufSize, &pSpec);

	size_t index_channel_in;
	long long k;

	for (int l = 0; l < shape_out[0]; l++)
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


	if (pSpec != NULL)
		ippsFree(pSpec);
}


void bin_4D(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list)
{
	/*bin up to 4 dimensions in dataset*/

	int srcStep, dstStep;
	IppiSize srcSize, dstSize;
	IppiPoint dstOffset = { 0,0 };

	setup_binning_dimensions(shape_in, shape_out, pixel_index_start, binning_list, &srcStep, &dstStep, &srcSize, &dstSize);

	int bufSize = 0;
	IppiResizeSpec_32f* pSpec = NULL;
	binning_ipp_init(srcSize, dstSize, &bufSize, &pSpec);

	size_t index_channel_in;
	long long k;

	for (int l = 0; l < shape_out[0]; l++)
	{
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

	if (pSpec != NULL)
		ippsFree(pSpec);
}