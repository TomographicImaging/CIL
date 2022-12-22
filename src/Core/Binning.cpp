#include "Binning.h"

void print_array_3D(const float* data, size_t z, size_t y, size_t x)
{

	for (int k = 0; k < z; k++)
	{
		for (int j = 0; j < y; j++)
		{
			for (int i = 0; i < x; i++)
			{
				std::cout << data[k * x * y + j * x + i] << '\t';
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << std::endl;
	}

}


void setup_binning(const size_t* shape_in, const size_t* shape_out, const size_t* pixel_index_start, const size_t* pixel_index_end, int* srcStep, int* dstStep, IppiSize* srcSize, IppiSize* dstSize, int* border)
{

	*srcStep = (int)shape_in[3] * sizeof(float);
	*dstStep = (int)shape_out[3] * sizeof(float);

	srcSize->width = (int)(pixel_index_end[1] - pixel_index_start[1]);
	srcSize->height = (int)(pixel_index_end[0] - pixel_index_start[0]);

	dstSize->width = (int)shape_out[3];
	dstSize->height = (int)shape_out[2];

	*border = ippBorderRepl;
	//use border from memory when it exists, otherwise replicate out
	if (pixel_index_start[1] > 0)
	{
		*border = *border | ippBorderInMemLeft;
	}
	if (pixel_index_end[1] < shape_in[3] - 1)
	{
		*border = *border | ippBorderInMemRight;
	}
	if (pixel_index_start[0] > 0)
	{
		*border = *border | ippBorderInMemTop;
	}
	if (pixel_index_end[0] < shape_in[2] - 1)
	{
		*border = *border | ippBorderInMemBottom;
	}


}
int bin_2D(const float* data_in, const size_t* shape_in, float* data_binned, const size_t* shape_out, const size_t* pixel_index_start, const size_t* binning_list, bool antialiasing)
{
	//std::cout << "Input arguments" << std::endl;

	//std::cout << "shape_in\t" << shape_in[0] << ", " << shape_in[1] << ", " << shape_in[2] << ", " << shape_in[3] << std::endl;
	//std::cout << "shape_out\t" << shape_out[0] << ", " << shape_out[1] << ", " << shape_out[2] << ", " << shape_out[3] << std::endl;
	//std::cout << "start_ind\t" << pixel_index_start[0] << ", " << pixel_index_start[1] << std::endl;
	//std::cout << "binning_list\t" << binning_list[0] << ", " << binning_list[1] << std::endl;
	//std::cout << "antialiasing\t" << antialiasing << std::endl;
	//std::cout << std::endl;



	/*
	Bins last two dimensions of 4D data

	const float* data_in, input array
	const size_t* shape_in, int[4]
	float* data_binned, preallocated output array
	const size_t* shape_out, int[4]
	const size_t* pixel_index_start, starting index in each dimension int[2]
	const size_t* binning_list, int[2] pixels to bin in y and x
	bool antialiasing, antialiasing filter on/off
	*/

	int srcStep, dstStep;
	IppiSize srcSize, dstSize;
	IppiPoint dstOffset = { 0,0 };
	int borderT;


	size_t pixel_index_end[2] = {
		pixel_index_start[0] + shape_out[2] * binning_list[0],
		pixel_index_start[1] + shape_out[3] * binning_list[1]
	};

	setup_binning(shape_in, shape_out, pixel_index_start, pixel_index_end, &srcStep, &dstStep, &srcSize, &dstSize, &borderT);

	//std::cout << "Calculated parameters" << std::endl;

	//std::cout << "pixel_index_end\t" << pixel_index_end[0] << ", " << pixel_index_end[1] << std::endl;

	//std::cout << "srcStep\t" << srcStep << std::endl;
	//std::cout << "dstStep\t" << dstStep << std::endl;
	//std::cout << "srcSize h\t" << srcSize.height << std::endl;
	//std::cout << "srcSize w\t" << srcSize.width << std::endl;

	//std::cout << "dstSize h\t" << dstSize.height << std::endl;
	//std::cout << "dstSize w\t" << dstSize.width << std::endl;
	//std::cout << std::endl;


	IppiResizeSpec_32f* pSpec = NULL;
	Ipp8u* pInitBuf = NULL;


	int specSize = 0, initSize = 0;

	if (antialiasing)
	{
		ippiResizeGetSize_32f(srcSize, dstSize, ippLinear, 1, &specSize, &initSize);
		pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
		pInitBuf = ippsMalloc_8u(initSize);
		ippiResizeAntialiasingLinearInit(srcSize, dstSize, pSpec, pInitBuf);
	}
	else
	{
		ippiResizeGetSize_32f(srcSize, dstSize, ippLinear, 0, &specSize, &initSize);
		pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
		ippiResizeLinearInit_32f(srcSize, dstSize, pSpec);
	}


	int bufSize = 0;
	ippiResizeGetBufferSize_8u(pSpec, dstSize, 1, &bufSize);

	long long k;

	IppStatus st = 0;
	if (antialiasing)
	{
#pragma omp parallel
		{
			Ipp8u* pBuffer = ippsMalloc_8u(bufSize);
#pragma omp for
			for (k = 0; k < shape_in[1] * shape_in[0]; k++)
			{
				ippiResizeAntialiasing_32f_C1R(&data_in[(size_t)k * shape_in[3] * shape_in[2] + pixel_index_start[0] * shape_in[3] + pixel_index_start[1]], srcStep, &data_binned[(size_t)k * shape_out[3] * shape_out[2]], dstStep, dstOffset, dstSize, (IppiBorderType)borderT, 0, pSpec, pBuffer);
			}
			if (pBuffer != NULL)
				ippsFree(pBuffer);
		}
	}
	else
	{
#pragma omp parallel
		{
			Ipp8u* pBuffer = ippsMalloc_8u(bufSize);
#pragma omp for
			for (k = 0; k < shape_in[1] * shape_in[0]; k++)
			{
				std::cout << "starting index " << (size_t)k * shape_in[3] * shape_in[2] + pixel_index_start[0] * shape_in[3] + pixel_index_start[1] << std::endl;
				st = ippiResizeLinear_32f_C1R(&data_in[(size_t)k * shape_in[3] * shape_in[2] + pixel_index_start[0] * shape_in[3] + pixel_index_start[1]], srcStep, &data_binned[(size_t)k * shape_out[3] * shape_out[2]], dstStep, dstOffset, dstSize, (IppiBorderType)borderT, 0, pSpec, pBuffer);
			}

			if (pBuffer != NULL)
				ippsFree(pBuffer);
		}
	}

	//std::cout << "IPP status: " << st << "\tmsg:\t" << ippGetStatusString(st) << std::endl;
	if(st !=0)
		return 1
	//print_array_3D(data_in, shape_in[1], shape_in[2], shape_in[3]);
	//print_array_3D(data_binned, shape_out[1], shape_out[2], shape_out[3]);

	if (pInitBuf != NULL)
		ippsFree(pInitBuf);

	if (pSpec != NULL)
		ippsFree(pSpec);

	return 0;

}