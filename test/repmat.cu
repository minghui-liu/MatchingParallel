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

// Usage: repmat m n
int main(int argc, char* argv[]){
	
	int m, n;
	// Read some values from the commandline
	m = atoi(argv[1]); 
	n = atoi(argv[2]); 

	Matrix In, Out;
	In.width = 1; In.height = 3;
	Out.height = In.height * m;
	Out.width = In.width * n;
	In.elements = (double*)malloc(In.height*In.width*sizeof(double));
	Out.elements = (double*)malloc(Out.height*Out.width*sizeof(double));
	double InE[3][1] = {{3},{5},{7}};
	memcpy(In.elements, InE, In.height*In.width*sizeof(double));
	
	printf("In:\n");
	printMatrix(In);

	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = d_In.width * d_In.height * sizeof(double);
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
	cudaMemcpy(d_Out.elements, Out.elements, size, cudaMemcpyHostToDevice);
	printf("Copy d_Out to device: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	repmatKernel<<<dimGrid, dimBlock>>>(d_In, m, n, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

	printf("Out:\n");
	printMatrix(Out);
	
	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

