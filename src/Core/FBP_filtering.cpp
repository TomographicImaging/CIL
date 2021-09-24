#include "FBP_filtering.h"

void ipp_status(IppStatus st)
{
	if (st)
	{
		std::cout << "Ipp Status " << st << ": " << ippGetStatusString(st) << std::endl;
	}
}

int filter_projections(float * data, const float * filter, const float* weights, int order, long num_proj, long pix_y, long pix_x)
{
	std::cout << "here7" << std::endl;

	auto start_all = std::chrono::system_clock::now();
	std::cout << "num_proj: " << num_proj << "\npix_y: " << pix_y << "\npix_x: " << pix_x <<  std::endl;
	std::cout << "order: " << order << std::endl;

	omp_set_num_threads(8);
#pragma omp parallel
	{
#pragma omp single
		{
			std::cout << "Spawned " << omp_get_num_threads() << " threads" << std::endl;
		}
	}

	//set up
	int width = 1 << order;

	std::cout << "width: " << width << std::endl;
	int offset = int(floor((width - pix_x) / 2));


	IppsFFTSpec_C_32fc* pSpec = 0;
	Ipp8u* pMemSpec = 0;
	Ipp8u* pMemInit = 0;

	int sizeSpec = 0;
	int sizeInit = 0;
	int sizeBuffer = 0;

	int flag = IPP_FFT_NODIV_BY_ANY;

	ippsFFTGetSize_C_32fc(order, IPP_FFT_DIV_FWD_BY_N, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer);
	pMemSpec = (Ipp8u*)ippMalloc(sizeSpec);
	pMemInit = (Ipp8u*)ippMalloc(sizeInit);
	ippsFFTInit_C_32fc(&pSpec, order, IPP_FFT_DIV_FWD_BY_N, ippAlgHintNone, pMemSpec, pMemInit);
	ippFree(pMemInit);

	long j;
	long k = 0;
	int half_pixy = floor(pix_y / 2);
	while(k < num_proj)
	{
		size_t proj_start = (size_t)k * pix_y * pix_x;
		k++;

#pragma omp parallel
		{
			Ipp8u* pMemBuffer = (Ipp8u*)ippMalloc(sizeBuffer);
			Ipp32fc* src = ippsMalloc_32fc(width);
			Ipp32fc* dst = ippsMalloc_32fc(width);

			float* out_ptr;
			size_t row_start;

#pragma omp for
			for (j = 0; j < half_pixy; j++)
			{
				row_start = (size_t)2 * j * pix_x;
				out_ptr = &data[proj_start + row_start];
				ippsMul_32f_I(weights+row_start, out_ptr, 2* pix_x);
				ippsSet_32fc({ 0.f,0.f }, src, width);
				ippsRealToCplx_32f(out_ptr, out_ptr + pix_x, src + offset, pix_x);
				ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
				ippsMul_32f32fc_I(filter, dst, width);
				ippsFFTInv_CToC_32fc(dst, src, pSpec, pMemBuffer);
				ippsCplxToReal_32fc(src + offset, out_ptr, out_ptr+pix_x, pix_x);
			}

#pragma omp single
			{
				if (pix_y % 2)
				{
					row_start = (size_t)pix_y * pix_x - pix_x;
					out_ptr = &data[proj_start + row_start];

					ippsMul_32f_I(weights + row_start, out_ptr, pix_x);
					ippsSet_32fc({ 0.f,0.f }, src, width);
					ippsRealToCplx_32f(out_ptr, NULL, src + offset, pix_x);
					ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
					ippsMul_32f32fc_I(filter, dst, width);
					ippsFFTInv_CToC_32fc(dst, src, pSpec, pMemBuffer);
					ippsReal_32fc(src + offset, out_ptr, pix_x);
				}
			}

			ippFree(src);
			ippFree(dst);
			ippFree(pMemBuffer);

		}
	}

	ippFree(pMemSpec);
	std::chrono::duration<double, std::milli> elapsed_milliseconds = std::chrono::system_clock::now() - start_all;
	std::cout << "\ttime c : " << elapsed_milliseconds.count() / 1000 << "s" << std::endl;

	return 0;
}
int filter_projections_reorder(float* data, const float* filter, const float* weights, int order, long pix_y, long num_proj, long pix_x)
{
	auto start_all = std::chrono::system_clock::now();
	std::cout << "num_proj: " << num_proj << "\npix_y: " << pix_y << "\npix_x: " << pix_x << std::endl;
	std::cout << "order: " << order << std::endl;

	omp_set_num_threads(7);
#pragma omp parallel
	{
#pragma omp single
		{
			std::cout << "Spawned " << omp_get_num_threads() << " threads" << std::endl;
		}
	}

	//set up
	int width = 1 << order;

	std::cout << "width: " << width << std::endl;
	int offset = int(floor((width - pix_x) / 2));


	IppsFFTSpec_C_32fc* pSpec = 0;
	Ipp8u* pMemSpec = 0;
	Ipp8u* pMemInit = 0;

	int sizeSpec = 0;
	int sizeInit = 0;
	int sizeBuffer = 0;

	int flag = IPP_FFT_NODIV_BY_ANY;

	ippsFFTGetSize_C_32fc(order, IPP_FFT_DIV_FWD_BY_N, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer);
	pMemSpec = (Ipp8u*)ippMalloc(sizeSpec);
	pMemInit = (Ipp8u*)ippMalloc(sizeInit);
	ippsFFTInit_C_32fc(&pSpec, order, IPP_FFT_DIV_FWD_BY_N, ippAlgHintNone, pMemSpec, pMemInit);
	ippFree(pMemInit);

	long j;
	long k = 0;
	int half_proj = floor(num_proj / 2);
	while (k < pix_y)
	{
		size_t col_start = (size_t)k * num_proj * pix_x;
		const float* weights_ptr = &weights[(size_t)k * pix_x];
		k++;

#pragma omp parallel
		{
			Ipp8u* pMemBuffer = (Ipp8u*)ippMalloc(sizeBuffer);
			Ipp32fc* src = ippsMalloc_32fc(width);
			Ipp32fc* dst = ippsMalloc_32fc(width);

			float* out_ptr;
			size_t row_start;

#pragma omp for
			for (j = 0; j < half_proj; j++)
			{
				row_start = (size_t)2 * j * pix_x;
				out_ptr = &data[col_start + row_start];
				ippsMul_32f_I(weights_ptr, out_ptr, pix_x);
				ippsMul_32f_I(weights_ptr, out_ptr + pix_x, pix_x);

				ippsSet_32fc({ 0.f,0.f }, src, width);
				ippsRealToCplx_32f(out_ptr, out_ptr + pix_x, src + offset, pix_x);
				ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
				ippsMul_32f32fc_I(filter, dst, width);
				ippsFFTInv_CToC_32fc(dst, src, pSpec, pMemBuffer);
				ippsCplxToReal_32fc(src + offset, out_ptr, out_ptr + pix_x, pix_x);
			}

#pragma omp single
			{
				if (num_proj % 2)
				{
					row_start = (size_t)num_proj * pix_x - pix_x;
					out_ptr = &data[col_start + row_start];
					ippsMul_32f_I(weights_ptr, out_ptr, pix_x);
					ippsSet_32fc({ 0.f,0.f }, src, width);
					ippsRealToCplx_32f(out_ptr, NULL, src + offset, pix_x);
					ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
					ippsMul_32f32fc_I(filter, dst, width);
					ippsFFTInv_CToC_32fc(dst, src, pSpec, pMemBuffer);
					ippsReal_32fc(src + offset, out_ptr, pix_x);
				}
			}

			ippFree(src);
			ippFree(dst);
			ippFree(pMemBuffer);

		}
	}

	ippFree(pMemSpec);
	std::chrono::duration<double, std::milli> elapsed_milliseconds = std::chrono::system_clock::now() - start_all;
	std::cout << "\ttime c : " << elapsed_milliseconds.count() / 1000 << "s" << std::endl;

	return 0;
}
void main()
{}