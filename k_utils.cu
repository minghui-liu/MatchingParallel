#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#pragma once
#define BLOCK_SIZE 32
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

// check if a square matrix is symmetric
__global__
void isSymmetricKernel(Matrix d_A, int *d_result) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	if(d_A.elements[row*d_A.width+col] != d_A.elements[row + col*d_A.width])
		*(d_result) = 0;
}

int isSymmetric(Matrix A) {
	printf("isSymmetric()\n");
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load result to device memory
	int result = 1;
	int *d_result;
	err = cudaMalloc(&d_result, sizeof(int));
	printf("CUDA malloc d_result: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);	
	printf("Copy result to device: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	isSymmetricKernel<<<dimGrid, dimBlock>>>(d_A, d_result);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	//read result from fdevice memory
	err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Copy result off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_result);

	return result;
}

//create an m-by-n tiling of a given matrix
__global__
void repmatKernel(Matrix d_A, int m, int n, Matrix d_B) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	for(int i=0; i < m; i++) {
		for(int j=0; j < n; j++) {
			d_B.elements[(row + i*d_A.height)*d_B.width + (col + j*d_A.width)] = d_A.elements[row*d_A.width + col];
		}
	}
}

//matrix transpose kernel
__global__
void transposeKernel(Matrix d_A, Matrix d_B){
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_B.elements[col*d_B.width + row] = d_A.elements[row*d_A.width + col];
}

// matrix zeros kernel called by zeros()
__global__
void zerosKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_A.elements[row*d_A.width+col] = 0;
}

// matrix getRow kernel
__global__
void getRowKernel(Matrix d_In, Matrix d_Out, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(row == num) 
		d_Out.elements[col] = d_In.elements[row*d_In.width+col];
}

// matrix getCol kernel
__global__
void getColKernel(Matrix d_In, Matrix d_Out, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(col == num) 
		d_Out.elements[row] = d_In.elements[row*d_In.width+col];
}

// matSub kernel
__global__
void matSubKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_A.width + col] = d_A.elements[row*d_A.width + col] - d_B.elements[row*d_A.width + col];
}

// matAdd kernel
__global__
void matAddKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] + d_B.elements[row*d_B.width + col];
}

__global__
void maxReduceKernel(double *elements, int size, double *d_part) {
	int  thread2;
	double temp;
	__shared__ double sdata[BLOCK_SIZE_DIM1];
	
	// Load max from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MIN;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE_DIM1;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;

			// Skipping the fictious threads blockDim.x ... blockDim_2-1
			if (thread2 < blockDim.x) {
				// Get the shared value stored by another thread 
				temp = sdata[thread2];
				if (temp > sdata[threadIdx.x]) 
					 sdata[threadIdx.x] = temp;
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

double maxOfMatrix(Matrix d_A) {
	cudaEvent_t start, stop;
	float time;
	// create events and start the timer
	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );*/

	// allocate d_part1 on device memory
	double *d_part1;
	err = cudaMalloc(&d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA malloc d_part1; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, DBL_MIN,  BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA memset d_part1 to DBL_MIN: %s\n", cudaGetErrorString(err));	
	
	// allocate d_part2 on device memory
	double *d_part2;
	err = cudaMalloc(&d_part2, BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA malloc d_part2; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, DBL_MIN, BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA memset d_part2 to DBL_MIN: %s\n", cudaGetErrorString(err));	
	
	// allocate d_max on device memory
	double *d_max;
	err = cudaMalloc(&d_max, sizeof(double));
	printf("CUDA malloc d_max; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_max, DBL_MIN, sizeof(double));
	printf("CUDA memset d_max to DBL_MIN: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid((d_A.width*d_A.height + dimBlock.x - 1)/dimBlock.x);
	
	// first pass
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_A.width*d_A.height, d_part1);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	
	// second pass
	dimGrid = dim3(BLOCK_SIZE_DIM1);
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1, d_part2);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));
	
	// third pass
	dimGrid = dim3(1);
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_part2, BLOCK_SIZE_DIM1, d_max);
	err = cudaThreadSynchronize();
	printf("Run kernel 3rd pass: %s\n", cudaGetErrorString(err));

	// read max from device memory
	double max;
	err = cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy max off of device: %s\n",cudaGetErrorString(err));
	
	// stop the timer
	/*cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);*/

	// free device memory
	cudaFree(d_part1);
	cudaFree(d_part2);
	cudaFree(d_max);
	
	return max;
}

__global__
void maxOfMatrixRow(Matrix d_A, Matrix d_col) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double max = d_A.elements[row*d_A.width];
	for (int col=0; col<d_A.width; col++) {
		max = (d_A.elements[row*d_A.width+col] > max)? d_A.elements[row*d_A.width+col] : max;
	}
	d_col[row] = max;
}


__global__
void maxOfMatrixCol(Matrix d_A, Matrix d_row) {
	int col = blockIdx.x * blockDdim.x + threadIdx.x;
	double max = d_A.elements[col];
	for (int row=0; row<d_A.height; row++) {
		max = (d_A.elements[row*d_A.width+col] > max)? d_A.elements[row*d_A.width+col] : max;
	}
	d_row[col] = max;
}

// matrix indexOfElement kernel
__global__
void indexOfElementKernel(Matrix d_A, double element, int *index) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	int idx = row*d_A.width+col;
	if (d_A.elements[idx] == element)
		*(index) = idx;
}

int indexOfElement(Matrix d_A, double element) {
	int index = -1;	

	// allocate d_index on device memory
	int *d_index;
	cudaError_t err = cudaMalloc(&d_index, sizeof(int));
	printf("CUDA malloc index; %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_index, &index, sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy index to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_A.width + dimBlock.x - 1)/dimBlock.x, (d_A.height + dimBlock.y - 1)/dimBlock.y );
	indexOfElementKernel<<<dimGrid, dimBlock>>>(d_A, element, d_index);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read index from device memory
	err = cudaMemcpy(&index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Copy index off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_index);
	return index;
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

	int nTotalThreads = BLOCK_SIZE_DIM1;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;

			// Skipping the fictious threads blockDim.x ... blockDim_2-1
			if (thread2 < blockDim.x) {
				// Get the shared value stored by another thread and sum it to sdata
				sdata[threadIdx.x] += sdata[thread2];
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

double matSum(Matrix d_A) {
	/*cudaEvent_t start, stop;
	float time;
	// create events and start the timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );*/

	// allocate d_part1 on device memory
	double *d_part1;
	err = cudaMalloc(&d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA malloc d_part1; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, 0,  BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA memset d_part1 to 0: %s\n", cudaGetErrorString(err));	
	
	// allocate d_part2 on device memory
	double *d_part2;
	err = cudaMalloc(&d_part2, BLOCK_SIZE_DIM1*sizeof(double));
	printf("CUDA malloc d_part2; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, 0, BLOCK_SIZE_DIM1*sizeof(double));
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
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part2, BLOCK_SIZE_DIM1, d_max);
	err = cudaThreadSynchronize();
	printf("Run kernel 3rd pass: %s\n", cudaGetErrorString(err));

	// read sum from device memory
	double sum;
	err = cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy sum off of device: %s\n",cudaGetErrorString(err));
	
	// stop the timer
	/*cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);*/

	// free device memory
	cudaFree(d_part1);
	cudaFree(d_part2);
	cudaFree(d_sum);
	
	return sum;
}

// matrix matTimesScaler kernel called by matTimesScaler()
__global__
void matTimesScalerKernel(Matrix d_In, double scaler, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	int idx = row * d_In.width +  col;
	d_Out.elements[idx] = d_In.elements[idx] * scaler;
}


