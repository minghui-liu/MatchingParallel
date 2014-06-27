#include <stdio.h>
#define BLOCK_SIZE 32
#define BLOCK_SIZE_DIM1 1024

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;

//function to print a matrix
void printMatrix(Matrix A) {
	printf("\n");
	for (int i=0; i<A.height; i++) {
		for (int j=0; j<A.width; j++) {
			printf("%.4f ", A.elements[i*A.width+j]); 
		}
		printf("\n");
	}
	printf("\n");
}

__global__
void sumReduceKernel(double *elements, int size, double *d_part) {
	int  thread2;
	__shared__ double sdata[BLOCK_SIZE_DIM1];
	
	// Load elements from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = 0;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = blockDim.x;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;
			// Get the shared value stored by another thread and sum it to sdata
			sdata[threadIdx.x] += sdata[thread2];

		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[0];
	}
}

double matSum(Matrix d_A) {

	// allocate d_part1 on device memory
	double *d_part1;
	cudaError_t err = cudaMalloc(&d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA malloc d_part1; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, 0,  BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA memset d_part1 to 0: %s\n", cudaGetErrorString(err));	
	
	// allocate d_part2 on device memory
	double *d_part2;
	err = cudaMalloc(&d_part2, BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA malloc d_part2; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part2, 0, BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA memset d_part2 to 0: %s\n", cudaGetErrorString(err));	
	
	// allocate d_sum on device memory
	double *d_sum;
	err = cudaMalloc(&d_sum, sizeof(double));
	printf("CUDA malloc d_sum; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_sum, 0, sizeof(double));
	printf("CUDA memset d_sum to 0: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid((d_A.width*d_A.height + dimBlock.x - 1)/dimBlock.x);

	// first pass
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_A.width*d_A.height, d_part1);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	
	// second pass
	dimGrid = dim3(BLOCK_SIZE_DIM1);
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1, d_part2);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));
	
	// third pass
	dimGrid = dim3(1);
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part2, BLOCK_SIZE_DIM1, d_sum);
	err = cudaThreadSynchronize();
	printf("Run kernel 3rd pass: %s\n", cudaGetErrorString(err));

	// read sum from device memory
	double sum;
	err = cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy sum off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_part1);
	cudaFree(d_part2);
	cudaFree(d_sum);
	
	return sum;
}
// Usage: matTimesScaler
int main(int argc, char* argv[]){
	
	Matrix A;
	A.width = 3; A.height = 3;
	A.elements = (double*)malloc(A.height*A.width*sizeof(double));
	double AE[3][3] = {{1, 3, 7},{2, 4, 8},{3, 6, 9}};
	memcpy(A.elements, AE, A.height*A.width*sizeof(double));
	
	printf("A:\n");
	printMatrix(A);

	double sum = matSum(A);

	printf("Sum = %.2f\n", sum);

	free(A.elements);
}

