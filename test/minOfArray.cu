#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_DIM2 32
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)

__global__
void minArrayKernel(double *elements, int size, double *d_part) {
	// Reduction min, works for any blockDim.x:
	int  thread2;
	double temp;
	__shared__ double sdata[BLOCK_SIZE];
	
	// Load min from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MAX;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE;	// Total number of threads, rounded up to the next power of two
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;

			// Skipping the fictious threads blockDim.x ... blockDim_2-1
			if (thread2 < blockDim.x) {
				// Get the shared value stored by another thread 
				temp = sdata[thread2];
				if (temp < sdata[threadIdx.x]) 
					 sdata[threadIdx.x] = temp;
			}
		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 copy the min to d_min
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[threadIdx.x];
	}
}

double minOfArray(double* A, int elements) {
	cudaEvent_t start, stop;
	float time;
	// create events and start the timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// load A to device memory
	double* d_A;
	size_t size = elements * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load d_part to device memory
	double *d_part;
	err = cudaMalloc(&d_part, BLOCK_SIZE*sizeof(double));
	printf("CUDA malloc d_part; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part, DBL_MAX, BLOCK_SIZE*sizeof(double));
	printf("CUDA memset d_part to DBL_MAX: %s\n", cudaGetErrorString(err));

	// load d_min to device memory
	double *d_min;
	err = cudaMalloc(&d_min, sizeof(double));
	printf("CUDA malloc d_min; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_min, DBL_MAX, sizeof(double));
	printf("CUDA memset d_min to DBL_MAX: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((elements + dimBlock.x - 1)/dimBlock.x);
	
	// first pass
	minArrayKernel<<<dimGrid, dimBlock>>>(d_A, elements, d_part);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	// second pass
	dimGrid = dim3(1);
	minArrayKernel<<<dimGrid, dimBlock>>>(d_part, BLOCK_SIZE, d_min);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));

	// read min from device memory
	double min;
	err = cudaMemcpy(&min, d_min, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy min off of device: %s\n",cudaGetErrorString(err));
	
	// stop the timer
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);

	// free device memory
	cudaFree(d_A);
	cudaFree(d_min);
	return min;
}

// matrix populate kernel called by populate()
__global__
void populateKernel(double* d_A, int size) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row >= size) return;
	d_A[row] = threadIdx.y; 
}

void populate(double* A, int elements) {
	srand(time(0));
	cudaEvent_t start, stop;
	float time;
	// create events and start the timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// load A to device memory
	double* d_A;
	size_t size = elements * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	  //invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM2, BLOCK_SIZE_DIM2);
	dim3 dimGrid( (elements + dimBlock.x - 1)/dimBlock.x, 1 );
	populateKernel<<<dimGrid, dimBlock>>>(d_A, size);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(err));

	// stop the timer
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);

	// free device memory
	cudaFree(d_A);
}

//usage : minOfArray height width
int main(int argc, char* argv[]) {
	double* A;
	int a1;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* elements in A */
	if (a1 > 1048576) {
		printf("Arrays bigger than 1048576 elements are not supported yet\n");
		return 0;
	}
	A = (double*)malloc(a1 * sizeof(double));
	// give A values
	populate(A, a1);
	//printMatrix(A);
	for(int i=0; i<a1; i++){
		printf("%f \t", A[i]);
	}
	printf("\n");
	// call zeros
	double min = minOfArray(A, a1);
	printf("\nThe min element is: %.4f\n", min);
}
