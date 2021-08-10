#include "GradientFunctions.h"
#include <iostream>

void L21Norm_3D(const float * data, double * out_value, long dimX, long dimY, long dimZ)
{
	L21Norm grad = L21Norm(data, dimX, dimY, dimZ)
	gradient_direct_foward_3D<L21Norm>(&grad);
    *out_value = grad.m_sum;
}
void L21Norm_2D(const float * data, double * out_value, long dimX, long dimY)
{
	L21Norm grad = L21Norm(data, dimX, dimY);
	gradient_direct_foward_2D<L21Norm>(&grad);
    *out_value = grad.m_sum;
}
