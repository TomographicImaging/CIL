#define dll_EXPORTS = 1

#include "FiniteDifferenceLibrary.h"

DLL_EXPORT int openMPtest(int nThreads)
{
	omp_set_num_threads(nThreads);

	int nThreads_running;
#pragma omp parallel
	{
		if (omp_get_thread_num() == 0)
		{
			nThreads_running = omp_get_num_threads();
		}
	}
	return nThreads_running;
}

int fdiff_direct_neumann(const float *inimagefull, float *outimageXfull, float *outimageYfull, float *outimageZfull, float *outimageCfull, long nx, long ny, long nz, long nc)
{
	size_t volume = nx * ny * nz;

	const float *inimage = inimagefull;
	float *outimageX = outimageXfull;
	float *outimageY = outimageYfull;
	float *outimageZ = outimageZfull;

	int offset1 = (nz - 1) * nx * ny;	  //ind to beginning of last slice
	int offset2 = offset1 + (ny - 1) * nx; //ind to beginning of last row

	long c;
	
	int z_dim = nz > 1 ? 1: 0;

	for (c = 0; c < nc; c++)
	{
#pragma omp parallel
		{
			long ind, k, j, i;
			float pix0;
			//run over all and then fix boundaries

#pragma omp for nowait
			for (ind = 0; ind < nx * ny * (nz - 1); ind++)
			{
				pix0 = -inimage[ind];
				outimageX[ind] = pix0 + inimage[ind + 1];
				outimageY[ind] = pix0 + inimage[ind + nx];
				outimageZ[ind] = pix0 + inimage[ind + nx * ny];
			}

#pragma omp for nowait
			for (ind = 0; ind < nx * (ny - 1); ind++)
			{
				pix0 = -inimage[ind + offset1];

				outimageX[ind + offset1] = pix0 + inimage[ind + offset1 + 1];
				outimageY[ind + offset1] = pix0 + inimage[ind + offset1 + nx];
			}

#pragma omp for
			for (ind = 0; ind < nx - 1; ind++)
			{
				pix0 = -inimage[ind + offset2];
				outimageX[ind + offset2] = pix0 + inimage[ind + offset2 + 1];
			}

			//boundaries
#pragma omp for nowait
			for (k = 0; k < nz; k++)
			{
				for (i = 0; i < nx; i++)
				{
					outimageY[(k * ny * nx) + (ny - 1) * nx + i] = 0;
				}
			}
#pragma omp for nowait
			for (k = 0; k < nz; k++)
			{
				for (j = 0; j < ny; j++)
				{
					outimageX[k * ny * nx + j * nx + nx - 1] = 0;
				}
			}
			if (z_dim)
			{
#pragma omp for
				for (ind = 0; ind < ny * nx; ind++)
				{
					outimageZ[nx * ny * (nz - 1) + ind] = 0;
				}
			}
		}

		inimage += volume;
		outimageX += volume;
		outimageY += volume;
		outimageZ += volume;
	}

	//now the rest of the channels
	if (nc > 1)
	{
		long ind;

		for (c = 0; c < nc - 1; c++)
		{
			float *outimageC = outimageCfull + c * volume;
			const float *inimage = inimagefull + c * volume;

#pragma omp parallel for
			for (ind = 0; ind < volume; ind++)
			{
				outimageC[ind] = -inimage[ind] + inimage[ind + volume];
			}
		}

#pragma omp parallel for
		for (ind = 0; ind < volume; ind++)
		{
			outimageCfull[(nc - 1) * volume + ind] = 0;
		}
	}

	return 0;
}
int fdiff_direct_periodic(const float *inimagefull, float *outimageXfull, float *outimageYfull, float *outimageZfull, float *outimageCfull, long nx, long ny, long nz, long nc)
{
	size_t volume = nx * ny * nz;

	const float *inimage = inimagefull;
	float *outimageX = outimageXfull;
	float *outimageY = outimageYfull;
	float *outimageZ = outimageZfull;

	int offset1 = (nz - 1) * nx * ny;	  //ind to beginning of last slice
	int offset2 = offset1 + (ny - 1) * nx; //ind to beginning of last row

	long c;
	for (c = 0; c < nc; c++)
	{

#pragma omp parallel
		{
			long ind, k;
			float pix0;
			//run over all and then fix boundaries
#pragma omp for nowait
			for (ind = 0; ind < nx * ny * (nz - 1); ind++)
			{
				pix0 = -inimage[ind];

				outimageX[ind] = pix0 + inimage[ind + 1];
				outimageY[ind] = pix0 + inimage[ind + nx];
				outimageZ[ind] = pix0 + inimage[ind + nx * ny];
			}

#pragma omp for nowait
			for (ind = 0; ind < nx * (ny - 1); ind++)
			{
				pix0 = -inimage[ind + offset1];

				outimageX[ind + offset1] = pix0 + inimage[ind + offset1 + 1];
				outimageY[ind + offset1] = pix0 + inimage[ind + offset1 + nx];
			}

#pragma omp for
			for (ind = 0; ind < nx - 1; ind++)
			{
				pix0 = -inimage[ind + offset2];

				outimageX[ind + offset2] = pix0 + inimage[ind + offset2 + 1];
			}

			//boundaries
#pragma omp for nowait
			for (k = 0; k < nz; k++)
			{
				for (int i = 0; i < nx; i++)
				{
					int ind1 = (k * ny * nx);
					int ind2 = ind1 + (ny - 1) * nx;

					outimageY[ind2 + i] = -inimage[ind2 + i] + inimage[ind1 + i];
				}
			}

#pragma omp for nowait
			for (k = 0; k < nz; k++)
			{
				for (int j = 0; j < ny; j++)
				{
					int ind1 = k * ny * nx + j * nx;
					int ind2 = ind1 + nx - 1;

					outimageX[ind2] = -inimage[ind2] + inimage[ind1];
				}
			}

			if (nz > 1)
			{
#pragma omp for nowait
				for (ind = 0; ind < ny * nx; ind++)
				{
					outimageZ[nx * ny * (nz - 1) + ind] = -inimage[nx * ny * (nz - 1) + ind] + inimage[ind];
				}
			}
		}

		inimage += volume;
		outimageX += volume;
		outimageY += volume;
		outimageZ += volume;
	}

	//now the rest of the channels
	if (nc > 1)
	{
		long ind;

		for (c = 0; c < nc - 1; c++)
		{
			float *outimageC = outimageCfull + c * volume;
			const float *inimage = inimagefull + c * volume;

#pragma omp parallel for
			for (ind = 0; ind < volume; ind++)
			{
				outimageC[ind] = -inimage[ind] + inimage[ind + volume];
			}
		}

#pragma omp parallel for
		for (ind = 0; ind < volume; ind++)
		{
			outimageCfull[(nc - 1) * volume + ind] = -inimagefull[(nc - 1) * volume + ind] + inimagefull[ind];
		}
	}

	return 0;
}
int fdiff_adjoint_neumann(float *outimagefull, const float *inimageXfull, const float *inimageYfull, const float *inimageZfull, const float *inimageCfull, long nx, long ny, long nz, long nc)
{
	//runs over full data in x, y, z. then corrects elements for bounday conditions and sums
	size_t volume = nx * ny * nz;

	//assumes nx and ny > 1
	int z_dim = nz - 1;

	float *outimage = outimagefull;
	const float *inimageX = inimageXfull;
	const float *inimageY = inimageYfull;
	const float *inimageZ = inimageZfull;

	float *tempX = (float *)malloc(volume * sizeof(float));
	float *tempY = (float *)malloc(volume * sizeof(float));
	float *tempZ;

	if (z_dim)
	{
		tempZ = (float *)malloc(volume * sizeof(float));
	}

	long c;
	for (c = 0; c < nc; c++) //just calculating x, y and z in each channel here
	{
#pragma omp parallel
		{
			long ind, k;

#pragma omp for
			for (ind = 1; ind < nx * ny * nz; ind++)
			{
				tempX[ind] = -inimageX[ind] + inimageX[ind - 1];
			}
#pragma omp for
			for (ind = nx; ind < nx * ny * nz; ind++)
			{
				tempY[ind] = -inimageY[ind] + inimageY[ind - nx];
			}

			//boundaries
#pragma omp for
			for (k = 0; k < nz; k++)
			{
				for (int j = 0; j < ny; j++)
				{
					tempX[k * ny * nx + j * nx] = -inimageX[k * ny * nx + j * nx];
					tempX[k * ny * nx + j * nx + nx - 1] = inimageX[k * ny * nx + j * nx + nx - 2];
				}
			}
#pragma omp for
			for (k = 0; k < nz; k++)
			{
				for (int i = 0; i < nx; i++)
				{
					tempY[(k * ny * nx) + i] = -inimageY[(k * ny * nx) + i];
					tempY[(k * ny * nx) + nx * (ny - 1) + i] = inimageY[(k * ny * nx) + nx * (ny - 2) + i];
				}
			}

			if (z_dim)
			{
#pragma omp for
				for (ind = nx * ny; ind < nx * ny * nz; ind++)
				{
					tempZ[ind] = -inimageZ[ind] + inimageZ[ind - nx * ny];
				}
#pragma omp for
				for (ind = 0; ind < ny * nx; ind++)
				{
					tempZ[ind] = -inimageZ[ind];
					tempZ[nx * ny * (nz - 1) + ind] = inimageZ[nx * ny * (nz - 2) + ind];
				}
#pragma omp for
				for (ind = 0; ind < volume; ind++)
				{
					outimage[ind] = tempX[ind] + tempY[ind] + tempZ[ind];
				}
			}
			else
			{
#pragma omp for
				for (ind = 0; ind < volume; ind++)
				{
					outimage[ind] = tempX[ind] + tempY[ind];
				}
			}
		}

		outimage += volume;
		inimageX += volume;
		inimageY += volume;
		inimageZ += volume;
	}
	free(tempX);
	free(tempY);

	if (z_dim)
		free(tempZ);

	//	//now the rest of the channels
	if (nc > 1)
	{
		long ind;

		for (c = 1; c < nc - 1; c++)
		{

#pragma omp parallel for
			for (ind = 0; ind < volume; ind++)
			{
				outimagefull[ind + c * volume] += -inimageCfull[ind + c * volume] + inimageCfull[ind + (c - 1) * volume];
			}
		}

#pragma omp parallel for
		for (ind = 0; ind < volume; ind++)
		{
			outimagefull[ind] += -inimageCfull[ind];
			outimagefull[(nc - 1) * volume + ind] += inimageCfull[(nc - 2) * volume + ind];
		}
	}

	return 0;
}
int fdiff_adjoint_periodic(float *outimagefull, const float *inimageXfull, const float *inimageYfull, const float *inimageZfull, const float *inimageCfull, long nx, long ny, long nz, long nc)
{
	//runs over full data in x, y, z. then correctects elements for bounday conditions and sums
	size_t volume = nx * ny * nz;

	//assumes nx and ny > 1
	int z_dim = nz - 1;

	float *outimage = outimagefull;
	const float *inimageX = inimageXfull;
	const float *inimageY = inimageYfull;
	const float *inimageZ = inimageZfull;

	float *tempX = (float *)malloc(volume * sizeof(float));
	float *tempY = (float *)malloc(volume * sizeof(float));
	float *tempZ;

	if (z_dim)
	{
		tempZ = (float *)malloc(volume * sizeof(float));
	}

	long c;
	for (c = 0; c < nc; c++) //just calculating x, y and z in each channel here
	{
#pragma omp parallel
		{
			long ind, k;

			//run over all and then fix boundaries
#pragma omp for
			for (ind = 1; ind < volume; ind++)
			{
				tempX[ind] = -inimageX[ind] + inimageX[ind - 1];
			}
#pragma omp for
			for (ind = nx; ind < volume; ind++)
			{
				tempY[ind] = -inimageY[ind] + inimageY[ind - nx];
			}

			//boundaries
#pragma omp for
			for (k = 0; k < nz; k++)
			{
				for (int i = 0; i < nx; i++)
				{
					tempY[(k * ny * nx) + i] = -inimageY[(k * ny * nx) + i] + inimageY[(k * ny * nx) + nx * (ny - 1) + i];
				}
			}
#pragma omp for
			for (k = 0; k < nz; k++)
			{
				for (int j = 0; j < ny; j++)
				{
					tempX[k * ny * nx + j * nx] = -inimageX[k * ny * nx + j * nx] + inimageX[k * ny * nx + j * nx + nx - 1];
				}
			}

			if (z_dim)
			{

#pragma omp for
				for (ind = nx * ny; ind < nx * ny * nz; ind++)
				{
					tempZ[ind] = -inimageZ[ind] + inimageZ[ind - nx * ny];
				}
#pragma omp for
				for (ind = 0; ind < ny * nx; ind++)
				{
					tempZ[ind] = -inimageZ[ind] + inimageZ[nx * ny * (nz - 1) + ind];
				}

#pragma omp for
				for (ind = 0; ind < volume; ind++)
				{
					outimage[ind] = tempX[ind] + tempY[ind] + tempZ[ind];
				}
			}
			else
			{
#pragma omp for
				for (ind = 0; ind < volume; ind++)
				{
					outimage[ind] = tempX[ind] + tempY[ind];
				}
			}
		}

		outimage += volume;
		inimageX += volume;
		inimageY += volume;
		inimageZ += volume;
	}
	free(tempX);
	free(tempY);

	if (z_dim)
		free(tempZ);

	//now the rest of the channels
	if (nc > 1)
	{
		long ind;

		for (c = 1; c < nc; c++)
		{

#pragma omp parallel for
			for (ind = 0; ind < volume; ind++)
			{
				outimagefull[ind + c * volume] += -inimageCfull[ind + c * volume] + inimageCfull[ind + (c - 1) * volume];
			}
		}

#pragma omp parallel for
		for (ind = 0; ind < volume; ind++)
		{
			outimagefull[ind] += -inimageCfull[ind] + inimageCfull[(nc - 1) * volume + ind];
		}
	}

	return 0;
}

DLL_EXPORT int fdiff4D(float *imagefull, float *gradCfull, float *gradZfull, float *gradYfull, float *gradXfull, long nc, long nz, long ny, long nx, int boundary, int direction, int nThreads)
{
	int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

	if (boundary)
	{
		if (direction)
			fdiff_direct_periodic(imagefull, gradXfull, gradYfull, gradZfull, gradCfull, nx, ny, nz, nc);
		else
			fdiff_adjoint_periodic(imagefull, gradXfull, gradYfull, gradZfull, gradCfull, nx, ny, nz, nc);
	}
	else
	{
		if (direction)
			fdiff_direct_neumann(imagefull, gradXfull, gradYfull, gradZfull, gradCfull, nx, ny, nz, nc);
		else
			fdiff_adjoint_neumann(imagefull, gradXfull, gradYfull, gradZfull, gradCfull, nx, ny, nz, nc);
	}

	omp_set_num_threads(nThreads_initial);
	return 0;
}
DLL_EXPORT int fdiff3D(float *imagefull, float *gradZfull, float *gradYfull, float *gradXfull, long nz, long ny, long nx, int boundary, int direction, int nThreads)
{
	int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

	if (boundary)
	{
		if (direction)
			fdiff_direct_periodic(imagefull, gradXfull, gradYfull, gradZfull, NULL, nx, ny, nz, 1);
		else
			fdiff_adjoint_periodic(imagefull, gradXfull, gradYfull, gradZfull, NULL, nx, ny, nz, 1);
	}
	else
	{
		if (direction)
			fdiff_direct_neumann(imagefull, gradXfull, gradYfull, gradZfull, NULL, nx, ny, nz, 1);
		else
			fdiff_adjoint_neumann(imagefull, gradXfull, gradYfull, gradZfull, NULL, nx, ny, nz, 1);
	}

	omp_set_num_threads(nThreads_initial);
	return 0;
}
DLL_EXPORT int fdiff2D(float *imagefull, float *gradYfull, float *gradXfull, long ny, long nx, int boundary, int direction, int nThreads)
{
	int nThreads_initial;
	threads_setup(nThreads, &nThreads_initial);

	if (boundary)
	{
		if (direction)
			fdiff_direct_periodic(imagefull, gradXfull, gradYfull, NULL, NULL, nx, ny, 1, 1);
		else
			fdiff_adjoint_periodic(imagefull, gradXfull, gradYfull, NULL, NULL, nx, ny, 1, 1);
	}
	else
	{
		if (direction)
			fdiff_direct_neumann(imagefull, gradXfull, gradYfull, NULL, NULL, nx, ny, 1, 1);
		else
			fdiff_adjoint_neumann(imagefull, gradXfull, gradYfull, NULL, NULL, nx, ny, 1, 1);
	}

	omp_set_num_threads(nThreads_initial);
	return 0;
}

