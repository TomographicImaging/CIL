#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"
#include "utilities.h"

#ifdef __cplusplus
extern "C" {
#endif
	DLL_EXPORT void L21Norm_3D(const float * data, double * out_value, long dimX, long dimY, long dimZ);
	DLL_EXPORT void L21Norm_2D(const float * data, double * out_value, long dimX, long dimY);
#ifdef __cplusplus
}
#endif

//Templated function for looping over a volume with boundary conditions at N
//inline functions must be difined in unique class
template <class T>
void gradient_direct_foward_3D(T *grad)
{
	long dimX = grad->get_dimX();
	long dimY = grad->get_dimY();
	long dimZ = grad->get_dimZ();
	long slice = dimX * dimY;
	long vol = slice * dimZ;

	long i, j, k, index;
	float val_x, val_y, val_z;

	for (k = 0; k < dimZ - 1; k++)
	{
#pragma omp parallel for private(i, j, index, val_x, val_y, val_z)
		for (j = 0; j < dimY - 1; j++)
		{
			index = k * slice + j * dimX;
			for (i = 0; i < dimX - 1; i++)
			{
				val_x = grad->get_val_x(index);
				val_y = grad->get_val_y(index);
				val_z = grad->get_val_z(index);
				grad->set_output_3D(index, val_x, val_y, val_z);

				index++;
			}

			val_x = grad->get_val_x_bc(index);
			val_y = grad->get_val_y(index);
			val_z = grad->get_val_z(index);
			grad->set_output_3D(index, val_x, val_y, val_z);
		}

		index = k * slice + (dimY-1)*dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y_bc(index);
			val_z = grad->get_val_z(index);
			grad->set_output_3D(index, val_x, val_y, val_z);
			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y_bc(index);
		val_z = grad->get_val_z(index);
		grad->set_output_3D(index, val_x, val_y, val_z);
	}

	k = dimZ - 1;
#pragma omp parallel for private(i, j, index)
	for (j = 0; j < dimY - 1; j++)
	{
		index = k * slice + j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y(index);
			val_z = grad->get_val_z_bc(index);
			grad->set_output_3D(index, val_x, val_y, val_z);
			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y(index);
		val_z = grad->get_val_z_bc(index);
		grad->set_output_3D(index, val_x, val_y, val_z);
	}

	index = vol - dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		val_x = grad->get_val_x(index);
		val_y = grad->get_val_y_bc(index);
		val_z = grad->get_val_z_bc(index);
		grad->set_output_3D(index, val_x, val_y, val_z);
		index++;
	}

	val_x = grad->get_val_x_bc(index);
	val_y = grad->get_val_y_bc(index);
	val_z = grad->get_val_z_bc(index);
	grad->set_output_3D(index, val_x, val_y, val_z);
}
//Templated function for looping over a volume with boundary conditions at N
//inline functions must be defined in unique class
template <class T>
void gradient_direct_foward_2D(T *grad)
{
	long dimX = grad->get_dimX();
	long dimY = grad->get_dimY();
	long slice = dimX * dimY;

	long i, j, index;
	float val_x, val_y;

#pragma omp parallel for private(i, j, index, val_x, val_y)
	for (j = 0; j < dimY - 1; j++)
	{
		index = j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y(index);
			grad->set_output_2D(index, val_x, val_y);

			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y(index);
		grad->set_output_2D(index, val_x, val_y);
	}

	index = slice - dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		val_x = grad->get_val_x(index);
		val_y = grad->get_val_y_bc(index);
		grad->set_output_2D(index, val_x, val_y);
		index++;
	}

	val_x = grad->get_val_x_bc(index);
	val_y = grad->get_val_y_bc(index);
	grad->set_output_2D(index, val_x, val_y);
}

class base_gradient
{
protected:
	long m_dimX;
	long m_dimY;
	long m_dimZ;
	long m_slice;

public:
	inline long get_dimX()
	{
		return m_dimX;
	}
	inline long get_dimY()
	{
		return m_dimY;
	}
	inline long get_dimZ()
	{
		return m_dimZ;
	}
	static inline float FowardDifference(const float *arr, long index, long stride)
	{
		return arr[index] - arr[index + stride];
	}
};

class L21Norm : public base_gradient
{
private:
	const float * m_data = data;

public:
	double * m_sum;

	inline void set_output_2D(long index, float val_x, float val_y)
	{
		m_sum += math.sqrt(val_x * val_x + val_y * val_y);
	}
	inline void set_output_3D(long index, float val_x, float val_y, float val_z)
	{
		m_sum += math.sqrt(val_x * val_x + val_y * val_y + val_z * val_z);

	}
	inline float get_val_x(long index)
	{
		return FowardDifference(m_data, index, 1);
	}
	inline float get_val_y(long index)
	{
		return FowardDifference(m_data, index, m_dimX);
	}
	inline float get_val_z(long index)
	{
		return FowardDifference(m_data, index, m_slice);
	}
	inline float get_val_x_bc(long index)
	{
		return 0.0f;
	}
	inline float get_val_y_bc(long index)
	{
		return 0.0f;
	}
	inline float get_val_z_bc(long index)
	{
		return 0.0f;
	}
	L21Norm::L21Norm(const float * data, long dimX, long dimY, long dimZ)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimZ = dimZ;
		m_slice = dimX * dimY;

		m_data = data;

        m_sum=0.0;
	}
	L21Norm::L21Norm(float * data, float * out_value, long dimX, long dimY)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_slice = dimX * dimY;

		m_data = data;

        m_sum=0.0;
	}
};
