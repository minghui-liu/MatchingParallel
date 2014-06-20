#include <stdio.h>
#include "graphMatching.cu"
#include "utils.cu"
#include <curand.h>
#include <cuda.h>
#include <stdlib.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

#define NODES 1024

__global__
void binaryKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	if(d_A.elements[row*d_A.width+col] < 0.5)
		d_A.elements[row*d_A.width+col] = 0;
	else
		d_A.elements[row*d_A.width+col] = 1;
}

void binary(Matrix A){
// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	binaryKernel<<<dimGrid, dimBlock>>>(d_A);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}

__global__
void makeSymmetricKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	if(row > col)
		d_A.elements[row*d_A.width+col] = d_A.elements[col*d_A.width+row];
}

void makeSymmetric(Matrix A){
// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	makeSymmetricKernel<<<dimGrid, dimBlock>>>(d_A);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}

int main(){

	size_t n = NODES*NODES;
	size_t i;
	size_t m = (NODES*NODES)/2 - NODES/2;

	curandGenerator_t gen, gen1;
	double *devData, *hostData, *devData1, *hostAdjacent;

	hostData = (double *)calloc(n, sizeof(double));
	hostAdjacent = (double *)calloc(m, sizeof(double));

	CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double)));
	CUDA_CALL(cudaMalloc((void **)&devData1, m*sizeof(double)));

	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen1, 1234ULL));

	CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
	CURAND_CALL(curandGenerateUniformDouble(gen1, devData1, m));

	CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(hostAdjacent, devData1, m * sizeof(double), cudaMemcpyDeviceToHost));

	for(i = 0; i < 10; i++){
		for(size_t j = 0; j < 10; j++){
			printf("%1.4f\t", hostData[i*NODES + j]);
		}
		printf("\n");
	}
	printf("\n");

	CURAND_CALL(curandDestroyGenerator(gen));
	CUDA_CALL(cudaFree(devData));	
	CURAND_CALL(curandDestroyGenerator(gen1));
	CUDA_CALL(cudaFree(devData1));

	Matrix Adjacency;
	Adjacency.width = NODES;
	Adjacency.height = NODES;
	Adjacency.elements = (double*)malloc(m*sizeof(double));
	memcpy(Adjacency.elements, hostAdjacent, m*sizeof(double));

/*	for(i = 0; i < 10; i++){
		printf("%1.4f\t", Adjacency.elements[i]);
	}
	printf("\n");
*/
	Matrix G1, G2;
	G2.width = G1.width = NODES;
	G2.height = G1.height  = NODES;
	G1.elements = (double*)malloc(G1.height*G1.width*sizeof(double));
	G2.elements = (double*)malloc(G2.height*G2.width*sizeof(double));

	memcpy(G1.elements, hostData, n*sizeof(double));

	makeSymmetric(G1);

	memcpy(G2.elements, G1.elements, G2.height*G2.width*sizeof(double));

	printf("G1:\n");
	for(int i=0; i<10; i++){
		for(int j=0; j < 10; j++){
			printf("%1.4f\t", G1.elements[i*NODES + j]);
		}
		printf("\n");
	}
	printf("\n");
/*	printf("G2:\n");
	for(int i=0; i<10; i++){
		printf("%1.4f\t", G2.elements[i]);
	}
	printf("\n");
*/
	binary(Adjacency);

	Matrix X, Y, Z;
	X.width = Y.width = Z.width = NODES;
	X.height = Y.height = Z.height  = NODES;
	X.elements = (double*)malloc(X.height*X.width*sizeof(double));
	Y.elements = (double*)malloc(Y.height*Y.width*sizeof(double));
	Z.elements = (double*)malloc(Z.height*Z.width*sizeof(double));
	
	// sigma = 1, numberOfMatches = 3
	graphMatching(G1, G2, 1, NODES, X, Z, Y);

/*	printf("G1:\n");
	for(int i=0; i<10; i++){
		for(int j=0; j < 10; j++){
			printf("%1.4f\t", G1.elements[i*NODES + j]);
		}
		printf("\n");
	}*/
	printf("\n");
	for(int i=0; i<10; i++){
		for(int j=0; j < 10; j++){
			printf("%1.4f\t", Y.elements[i*NODES + j]);
		}
		printf("\n");
	}
	printf("\n");
/*	printf("X(hard):\n");
	printMatrix(X);
	printf("Z(soft):\n");
	printMatrix(Z);
	printf("Y(debug):\n");
	printMatrix(Y);*/
}