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

// matrix getCol kernel
__global__
void getColKernel(Matrix d_In, Matrix d_Out, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(col == num) 
		d_Out.elements[row] = d_In.elements[row*d_In.width+col];
}

// Usage: getCol num
int main(int argc, char* argv[]){

	// Read some values from the commandline
	int num = atoi(argv[1]); 
	

	Matrix In, Out;
	In.width = 5; In.height = 5;
	Out.height = In.height;
	Out.width = 1;
	In.elements = (double*)malloc(In.height*In.width*sizeof(double));
	Out.elements = (double*)malloc(Out.height*Out.width*sizeof(double));
	double InE[5][5] = {{1, 2, 3, 4, 5},
											{6, 7, 8, 9, 10},
											{11,12,13,14,15},
											{16,17,18,19,20},
											{21,22,23,24,25}
											};
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
	getColKernel<<<dimGrid, dimBlock>>>(d_In, d_Out, num);
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

