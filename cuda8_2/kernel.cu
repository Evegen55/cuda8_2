#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
	cudaError_t cudaStatus;
	int N = 1 << 20;
	float *x, *y, *d_x, *d_y;
	//first, allocate memory in RAM
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));

	//second, allocate memory in GPU buffer
	cudaStatus = cudaMalloc(&d_x, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for d_x failed with error: %d", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_y, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for d_y failed with error: %d", cudaStatus);
		goto Error;
	}
	//initialize array in RAM
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	//move arrays from RAM to GPU buffer
	cudaStatus = cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for transferring x to d_x failed with error: %d", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for transferring y to d_y failed with error: %d", cudaStatus);
		goto Error;
	}

	// Perform SAXPY on 1M elements in GPU
	saxpy << <(N + 255) / 256, 256 >> >(N, 2.0f, d_x, d_y);

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = max(maxError, abs(y[i] - 4.0f));
		//printf("%f\t", y[i]);
	}

	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

Error:
	cudaFree(d_x);
	cudaFree(d_y);
	//free(x);
	//free(y);
	return cudaStatus;
}
