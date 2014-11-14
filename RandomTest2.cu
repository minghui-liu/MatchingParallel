/*
 * File: RandomTest2.c
 *
 * Testing the probablistic graph matching algorithm
 * by rotating a set of artificial points and then calculating
 * the similarity score for the edges
 *
 * Special thanks to Reid Delaney, who came up with the rotation
 * method.
 *
 * Author: Kevin Liu
 * Date: Nov 8 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include "graphMatching.cu"
#include "matlib.cu"

#define PI 3.14159265
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      return EXIT_FAILURE;}} while(0)


/* Give a random double vale in the interval [0, 2PI] */
double randomFloatAngle(void) {
	//generate random double between 0 and 1
  double r = (double)rand()/(double)RAND_MAX;
  //scale to the 0 to 2PI range
  r *= (2*PI);
  return r;
}

/* Rotation Kernel */
__global__
void RotateKernel(Matrix d_V1, Matrix d_V2, double centerX, double centerY, double theta) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// if out of bound return
	if (idx >= d_V1.height) return;
	
	*(d_V2.elements + idx * d_V2.width) = cos(theta) * (*(d_V1.elements + idx * d_V1.width) - centerX) - sin(theta) * (*(d_V1.elements + idx * d_V1.width + 1) - centerY) + centerX;
	*(d_V2.elements + idx * d_V2.width + 1) = sin(theta) * (*(d_V1.elements + idx * d_V1.width) - centerX) + cos(theta) * (*(d_V1.elements + idx * d_V1.width + 1) - centerY) + centerY;

}


/* Distort kernel 
__global__
void DistortKernel(Matrix d_V) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// if out of bound return
	if (idx >= d_V.height) return;

	*(d_V.elements + idx * d_V.width) = (2 * (double)rand() / (double)RAND_MAX - 1)/100 + *(d_V.elements + idx * d_V.width);
	*(d_V.elements + idx * d_V.width + 1) = (2 * (double)rand() / (double)RAND_MAX - 1)/100 + *(d_V.elements + idx * d_V.width + 1);
  
} */

/* Node distance kernel */
__global__
void DistanceKernel(Matrix d_V, Matrix d_E) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;	
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	// if out of bound return
	if (r >= d_E.height || c >= d_E.width) return;	

	if (r == c)
		*(d_E.elements + r * d_E.width + c) = 0;
	else {
		*(d_E.elements + r * d_E.width + c) = sqrt( (*(d_V.elements + r * d_V.width) - *(d_V.elements + c * d_V.width)) * (*(d_V.elements + c * d_V.width) - *(d_V.elements + c * d_V.width)) + (*(d_V.elements + r * d_V.width + 1) - *(d_V.elements + c * d_V.width + 1)) * (*(d_V.elements + r * d_V.width + 1) - *(d_V.elements + c * d_V.width + 1)) );
	}

}


int main(int argc, char *argv[]) {
	char vflag = 0;
	char sflag = 0;
	char opt;
	
	// process command-line options
	while ((opt = getopt(argc, argv, "vm"))	!= -1) {
		switch (opt) {
			case 'v':
				vflag = 1;	
				break;
			case 's':
				sflag = 1;
				break;
			case '?':
				if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				abort();
		}
	}
	
	if (argc < 2 || optind == argc) {
	  	fprintf(stderr, "Must specify test size.\n");
		return EXIT_FAILURE;
	}
	
	// test size
	int test_size = atoi(argv[optind++]);
	// DEBUG CODE
	//printf("vflag = %d, sflag = %d, test_size = %d\n", vflag, sflag, test_size);
	// print non-opt args
	for (int index = optind; index < argc; index++)
		printf ("Non-option argument %s\n", argv[index]);
	
	/* Allocate node cordinates on host */
	Matrix V1, V2;
	V1.width = V2.width = 2;
	V1.height = V2.height = test_size;
	V1.elements = (double*)malloc(test_size*2*sizeof(double));
	V2.elements = (double*)malloc(test_size*2*sizeof(double));
	
	/* Allocate V1 and V2 on device memory */
	Matrix d_V1, d_V2;
	d_V1.width = d_V2.width = V1.width;
	d_V1.height = d_V2.height = V1.height;
	size_t size = d_V1.width * d_V1.height * sizeof(double);
	CUDA_CALL(cudaMalloc(&d_V1.elements, size));
	CUDA_CALL(cudaMalloc(&d_V2.elements, size));

	curandGenerator_t gen;
	/* Create pseudo-random number generator */
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    
	/* Set seed */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
	/* Generate n * 2 floats on device */	
	CURAND_CALL(curandGenerateUniformDouble(gen, d_V1.elements, d_V1.width * d_V1.height));

	double theta = randomFloatAngle(); 
	/* Rotate V1's cordinates to get V2 */
	RotateKernel<<<(d_V1.height + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_V1, d_V2, 0.5, 0.5, theta);
	CUDA_CALL(cudaThreadSynchronize());

	/* Distort V2 */
	//DistortKernel<<<(d_V2.height + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_V2);

	/* Copy device memory to host */
	CUDA_CALL(cudaMemcpy(V1.elements, d_V1.elements, size, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(V2.elements, d_V2.elements, size, cudaMemcpyDeviceToHost));

	/* Allocate edge weight matrices on host */
	Matrix E1, E2;
	E1.width = E1.height = test_size;
	E2.width = E2.height = test_size;
	size = E1.width * E1.height * sizeof(double);
	E1.elements = (double *)malloc(size);
	E2.elements = (double *)malloc(size);
	
	/* Calculate E1 and E2 to device memory */
	Matrix d_E1, d_E2;
	d_E1.width = d_E1.height = E1.width;
	d_E2.width = d_E2.height = E2.width;
	CUDA_CALL(cudaMalloc(&d_E1.elements, size));
	CUDA_CALL(cudaMalloc(&d_E2.elements, size));

	/* Calculate block and grid size */
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_E1.width + dimBlock.x - 1)/dimBlock.x, (d_E1.height + dimBlock.y - 1)/dimBlock.y );

	/* Calculate distances between nodes */
	DistanceKernel<<<dimGrid, dimBlock>>>(d_V1, d_E1);
	DistanceKernel<<<dimGrid, dimBlock>>>(d_V2, d_E2);

	/* Copy E1 and E2 from device memory to host */
	CUDA_CALL(cudaMemcpy(E1.elements, d_E1.elements, size, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(E2.elements, d_E2.elements, size, cudaMemcpyDeviceToHost));

	/* Allocate X Y Z on host */
	Matrix X, Y, Z;
	X.width = X.height = test_size;
	Y.width = Y.height = test_size;
	Z.width = Z.height = test_size;
	size = X.width * X.height * sizeof(double);
	X.elements = (double *)malloc(size);
	Y.elements = (double *)malloc(size);
	Z.elements = (double *)malloc(size);

	// Graph Matching
	graphMatching(E1, E2, 0.01, test_size, X, Z, Y);

	// Verbose mode
	if (vflag) {
		printf("Node coordinates of Graph 1\n");
		printMatrix(V1);
		printf("Node coordinates of Graph 2\n");
		printMatrix(V2);
		printf("Edge weights of Graph 1\n");
		printMatrix(E1);
		printf("Edge weights of Graph 2\n");
		printMatrix(E2);
  	printf("X - hard result\n");
  	printMatrix(X);
  	printf("Z - soft result\n");
  	printMatrix(Z);
  	printf("Y - debug information\n");
  	printMatrix(Y);
	}

	/* Cleanup */
	CURAND_CALL(curandDestroyGenerator(gen));
	CUDA_CALL(cudaFree(d_V1.elements));
	CUDA_CALL(cudaFree(d_V2.elements));
	CUDA_CALL(cudaFree(d_E1.elements));
	CUDA_CALL(cudaFree(d_E2.elements));
	free(V1.elements);
	free(V2.elements);
	free(E1.elements);
	free(E2.elements);
	free(X.elements);
	free(Y.elements);
	free(Z.elements);

	return 0;
}


