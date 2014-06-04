#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_DIM2 32
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
/*typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;
*/

__global__
void arraySumKernel(double *elements, int size, double *d_part) {
	// Reduction max, works for any blockDim.x:
	int  thread2;
	double temp;
	__shared__ double sdata[BLOCK_SIZE];
	
	// Load max from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MIN;
	
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
					 sdata[threadIdx.x] += temp;
			}
		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 copy the max to d_max
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[threadIdx.x];
	}
}

/*int NearestPowerOf2(int n) {
  if (!n) return n;  //(0 == 2^0)
  int x = 1;
  while(x < n) {
      x <<= 1;
  }
  return x;
}*/

double arraySum(double* A, int elements) {
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
	err = cudaMemset(d_part, DBL_MIN, BLOCK_SIZE*sizeof(double));
	printf("CUDA memset d_part to DBL_MIN: %s\n", cudaGetErrorString(err));

	// load d_max to device memory
	double *d_max;
	err = cudaMalloc(&d_max, sizeof(double));
	printf("CUDA malloc d_max; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_max, DBL_MIN, sizeof(double));
	printf("CUDA memset d_max to DBL_MIN: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((elements + dimBlock.x - 1)/dimBlock.x);
	//int blockDim_2 = NearestPowerOf2(d_A.width*d_A.height);
	//printf("nearest power of 2 (blockDim_2): %d\n",blockDim_2);
	// first pass
	arraySumKernel<<<dimGrid, dimBlock>>>(d_A, elements, d_part);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	// second pass
	dimGrid = dim3(1);
	arraySumKernel<<<dimGrid, dimBlock>>>(d_part, BLOCK_SIZE, d_max);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));

	// read max from device memory
	double max;
	err = cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy max off of device: %s\n",cudaGetErrorString(err));
	
	// stop the timer
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);

	// free device memory
	cudaFree(d_A);
	cudaFree(d_max);
	return max;
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
/*void printMatrix(Matrix A) {
	printf("\n");
	for (int i=0; i<A.height; i++) {
		for (int j=0; j<A.width; j++) {
			printf("%.4f ", A.elements[i*A.width+j]); 
		}
		printf("\n");
	}
	printf("\n");
}*/

//usage : maxOfArray height width
int main(int argc, char* argv[]) {
	double* A;
	int a1;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* elements in A */
	if (a1 > 1048576) {
		printf("Matrices bigger than 1048576 elements are not supported yet\n");
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
	double min = arraySum(A, a1);
	printf("\nThe max element is: %.4f\n", min);
}
