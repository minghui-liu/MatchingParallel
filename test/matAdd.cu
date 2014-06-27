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

// matAdd kernel
__global__
void matAddKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] + d_B.elements[row*d_B.width + col];
}


// Usage: matAdd
int main(int argc, char* argv[]){
	
	/*int m, n;
	// Read some values from the commandline
	m = atoi(argv[1]); 
	n = atoi(argv[2]); */

	Matrix A, B, Out;
	Out.width = A.width = B.width = 3; 
	Out.height = A.height = B.height = 3;
	A.elements = (double*)malloc(A.height*A.width*sizeof(double));
	B.elements = (double*)malloc(B.height*B.width*sizeof(double));
	Out.elements = (double*)malloc(Out.height*Out.width*sizeof(double));
	double AE[3][3] = {{1, 3, 7},{2, 4, 8},{3, 6, 9}};
	double BE[3][3] = {{9, 9, 9},{9, 9, 9},{9, 9, 9}};
	memcpy(A.elements, AE, A.height*A.width*sizeof(double));
	memcpy(B.elements, BE, B.height*B.width*sizeof(double));
	
	printf("A:\n");
	printMatrix(A);
	printf("B:\n");
	printMatrix(B);

	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = d_A.width * d_A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = d_B.width * d_B.height * sizeof(double);
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(err));

	// allocate Out on device memory
	Matrix d_Out;
	d_Out.width = Out.width;
	d_Out.height = Out.height;
	size = d_Out.width * d_Out.height * sizeof(double);
	err = cudaMalloc(&d_Out.elements, size);
	printf("CUDA malloc d_Out: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_A.width + dimBlock.x - 1)/dimBlock.x, (d_A.height + dimBlock.y - 1)/dimBlock.y );
	matAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

	printf("Out:\n");
	printMatrix(Out);
	
	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_Out.elements);
}

