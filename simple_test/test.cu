#include <stdio.h>
#include "graphMatching.cu"
#include "test_utils.cu"
//#include "utils.cu"
#include <curand.h>
#include <cuda.h>
#include <stdlib.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0) 

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

int main(){

	size_t n = 1024*1024;
	size_t i;

	curandGenerator_t gen;
	double *devData, *hostData;

	hostData = (double *)calloc(n, sizeof(double));

	CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double)));

	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

	CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));

	CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double), cudaMemcpyDeviceToHost));

	for(i = 0; i < 10; i++){
		printf("%1.4f\t", hostData[i]);
	}
	printf("\n");

	CURAND_CALL(curandDestroyGenerator(gen));
	CUDA_CALL(cudaFree(devData));

	Matrix G1, G2;
	G2.width = G1.width = 1024;
	G2.height = G1.height  = 1024;
	G1.elements = (double*)malloc(G1.height*G1.width*sizeof(double));
	G2.elements = (double*)malloc(G2.height*G2.width*sizeof(double));

	memcpy(G1.elements, hostData, n*sizeof(double));

	memcpy(G2.elements, G1.elements, G2.height*G2.width*sizeof(double));

	printf("G1:\n");
	for(int i=0; i<10; i++){
		printf("%1.4f\t", G1.elements[i]);
	}
	printf("\n");
	printf("G2:\n");
	for(int i=0; i<10; i++){
		printf("%1.4f\t", G2.elements[i]);
	}
	printf("\n");

	Matrix X, Y, Z;
	X.width = Y.width = Z.width = 1024;
	X.height = Y.height = Z.height  = 1024;
	X.elements = (double*)malloc(X.height*X.width*sizeof(double));
	Y.elements = (double*)malloc(Y.height*Y.width*sizeof(double));
	Z.elements = (double*)malloc(Z.height*Z.width*sizeof(double));
	
	// sigma = 1, numberOfMatches = 3
	graphMatching(G1, G2, 1, 1024, X, Z, Y);
	printf("G1:\n");
	for(int i=0; i<10; i++){
		printf("%f\t", X.elements[i]);
	}
	printf("\n");
/*	printf("X(hard):\n");
	printMatrix(X);
	printf("Z(soft):\n");
	printMatrix(Z);
	printf("Y(debug):\n");
	printMatrix(Y);*/	
}
