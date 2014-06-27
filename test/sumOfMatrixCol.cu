#include <stdio.h>
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
void sumOfMatrixColKernel(Matrix d_A, Matrix d_row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= d_A.width) return;
	for (int row=0; row<d_A.height; row++) {
		d_row.elements[col] += d_A.elements[row*d_A.width+col];
	}
}

void sumOfMatrixCol(Matrix In, Matrix Out) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x );
	sumOfMatrixColKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// Usage: sumOfMatrixCol
int main() {

	Matrix In, Out;
	In.width = 4; In.height = 3;
	Out.height = 1;
	Out.width = In.width;
	In.elements = (double*)malloc(In.height*In.width*sizeof(double));
	Out.elements = (double*)malloc(Out.height*Out.width*sizeof(double));
	double InE[3][4] = {
											{1, 2, 3, 4},
											{6, 7, 8, 9},
											{11,12,13,14},
										 };
	memcpy(In.elements, InE, In.height*In.width*sizeof(double));
	
	printf("In:\n");
	printMatrix(In);
	
	sumOfMatrixCol(In, Out);
	
	printf("Sum of cols:\n");
	printMatrix(Out);
	
	// free memory
	free(In.elements);
	free(Out.elements);

}

