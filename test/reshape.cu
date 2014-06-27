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

// matrix reshape kernel called by reshape()
__global__
void reshapeKernel(Matrix d_In, Matrix d_Out) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(y >= d_In.height || x >= d_In.width) return;
	int c = x * d_In.height + y;
	d_Out.elements[(c%d_Out.height)*d_Out.width+(c/d_Out.height)] = d_In.elements[(c%d_In.height)*d_In.width+(c/d_In.height)];

}

void reshape(Matrix In, Matrix Out) {
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
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	reshapeKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// Usage: reshape
int main(int argc, char* argv[]) {
 
	Matrix A, Out;
	A.height = 3; A.width = 4;
	Out.height = 2; Out.width = 6;
	A.elements = (double*)malloc(A.height*A.width*sizeof(double));
	Out.elements = (double*)malloc(Out.height*Out.width*sizeof(double));
	double AE[3][4] = {{1, 4, 7, 10},{2, 5, 8, 11},{3, 6, 9, 12}};
	memcpy(A.elements, AE, A.height*A.width*sizeof(double));
	
	printf("A:\n");
	printMatrix(A);

	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = d_A.width * d_A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

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
	reshapeKernel<<<dimGrid, dimBlock>>>(d_A, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

	printf("Out:\n");
	printMatrix(Out);
	
	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_Out.elements);
}

