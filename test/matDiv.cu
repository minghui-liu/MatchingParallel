#include <stdio.h>
#include <stdlib.h>


#define BLOCK_SIZE 32 


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;

// matrix zeros kernel called by zeros()
__global__
void matDivKernel(Matrix d_A, Matrix d_B, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_A.width+col;
	if(row > d_A.height || col > d_A.width) return;
	d_Out.elements[idx] = d_A.elements[idx] / d_B.elements[idx];
}

void matDiv(Matrix A, Matrix B, Matrix Out) {
	if (A.width != B.width || A.height != B.height) {
		printf("Input matrices must have the same dimension!\n");
		return;
	}
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matDivKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_Out.elements);

}

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


//usage : matDiv height width 
int main(int argc, char* argv[]) {
	Matrix A, B, Out;
	int a1, a2;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* Height of A and B */
	a2 = atoi(argv[2]); /* Width of A and B */
	Out.height = B.height = A.height = a1;
	Out.width = B.width = A.width = a2;
	A.elements = (double*)malloc(A.width * A.height * sizeof(double));
	B.elements = (double*)malloc(B.width * B.height * sizeof(double));
	Out.elements = (double*)malloc(Out.width * Out.height * sizeof(double));
	// give A random values
	for(int i = 0; i < A.height; i++)
		for(int j = 0; j < A.width; j++) {
			A.elements[i*A.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
			B.elements[i*B.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
		}
	printMatrix(A);
	printMatrix(B);
	// call reshape()
	matDiv(A, B, Out);
	printMatrix(Out);
}
