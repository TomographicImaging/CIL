//  Copyright 2021 United Kingdom Research and Innovation
//  Copyright 2021 The University of Manchester
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

#include "FBP_filtering.h"

int filter_projections_avh(DataFloat data, DataFloatConst filter, DataFloatConst weights, int order, long num_proj, long pix_y, long pix_x)
{
	//set up
	int width = 1 << order;
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
				out_ptr = &data.data()[proj_start + row_start];
				ippsMul_32f_I(weights.data()+row_start, out_ptr, 2* pix_x);
				ippsSet_32fc({ 0.f,0.f }, src, width);
				ippsRealToCplx_32f(out_ptr, out_ptr + pix_x, src + offset, pix_x);
				ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
				ippsMul_32f32fc_I(filter.data(), dst, width);
				ippsFFTInv_CToC_32fc(dst, src, pSpec, pMemBuffer);
				ippsCplxToReal_32fc(src + offset, out_ptr, out_ptr+pix_x, pix_x);
			}

#pragma omp single
			{
				if (pix_y % 2)
				{
					row_start = (size_t)pix_y * pix_x - pix_x;
					out_ptr = &data.data()[proj_start + row_start];

					ippsMul_32f_I(weights.data() + row_start, out_ptr, pix_x);
					ippsSet_32fc({ 0.f,0.f }, src, width);
					ippsRealToCplx_32f(out_ptr, NULL, src + offset, pix_x);
					ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
					ippsMul_32f32fc_I(filter.data(), dst, width);
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
	return 0;
}
int filter_projections_vah(DataFloat data, DataFloatConst filter, DataFloatConst weights, int order, long pix_y, long num_proj, long pix_x)
{
	//set up
	int width = 1 << order;
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
		const float* weights_ptr = &weights.data()[(size_t)k * pix_x];
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
				out_ptr = &data.data()[col_start + row_start];
				ippsMul_32f_I(weights_ptr, out_ptr, pix_x);
				ippsMul_32f_I(weights_ptr, out_ptr + pix_x, pix_x);

				ippsSet_32fc({ 0.f,0.f }, src, width);
				ippsRealToCplx_32f(out_ptr, out_ptr + pix_x, src + offset, pix_x);
				ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
				ippsMul_32f32fc_I(filter.data(), dst, width);
				ippsFFTInv_CToC_32fc(dst, src, pSpec, pMemBuffer);
				ippsCplxToReal_32fc(src + offset, out_ptr, out_ptr + pix_x, pix_x);
			}

#pragma omp single
			{
				if (num_proj % 2)
				{
					row_start = (size_t)num_proj * pix_x - pix_x;
					out_ptr = &data.data()[col_start + row_start];
					ippsMul_32f_I(weights_ptr, out_ptr, pix_x);
					ippsSet_32fc({ 0.f,0.f }, src, width);
					ippsRealToCplx_32f(out_ptr, NULL, src + offset, pix_x);
					ippsFFTFwd_CToC_32fc(src, dst, pSpec, pMemBuffer);
					ippsMul_32f32fc_I(filter.data(), dst, width);
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
	return 0;
}
