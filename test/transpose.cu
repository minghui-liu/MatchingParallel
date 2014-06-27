#include <stdio.h>
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

__global__
void transposeKernel(Matrix d_A, Matrix d_B) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_B.elements[col*d_B.width+row] = d_A.elements[row*d_A.width+col];
}

void transpose(Matrix In, Matrix Out) {
	printf("transpose()\n");
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);

	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);
	printf("Copy In to device: %s\n", cudaGetErrorString(err));

	// allocate Out on device memory
	Matrix d_Out;
	d_Out.width = Out.width;
	d_Out.height = Out.height;
	size = d_Out.width * d_Out.height * sizeof(double);
	err = cudaMalloc(&d_Out.elements, size);
	printf("CUDA malloc d_Out: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	transposeKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// Usage: reshape
int main(int argc, char* argv[]) {
 
	Matrix A, At;
	A.height = 3; A.width = 4;
	At.height = A.width; At.width = A.height;
	A.elements = (double*)malloc(A.height*A.width*sizeof(double));
	At.elements = (double*)malloc(At.height*At.width*sizeof(double));
	double AE[3][4] = {	{1, 4, 7, 10},
											{2, 5, 8, 11},
											{3, 6, 9, 12}
										};
	memcpy(A.elements, AE, A.height*A.width*sizeof(double));
	
	printf("A:\n");
	printMatrix(A);
	
	transpose(A, At);
	
	printf("At:\n");
	printMatrix(At);
	
	// free device memory
	free(A.elements);
	free(At.elements);
}

