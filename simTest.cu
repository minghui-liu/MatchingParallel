/*
 * file: simTest.cu
 *
 * test hypergraph matching algorithm on a simple graph of randomly 
 * generated points using distortion of types rotation and translation
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "graphMatching.c"
#include "utils.c"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_DIM2 32
#define EPS 2.2204e-16
#define PI 3.14159265
#define TEST_SIZE 1024

typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;

//function returns a random double value in the interval [-1,1]
double randomDouble(){

	//generate random double between 0 and 1
	double r = (double)rand()/(double)RAND_MAX;

	if(rand()%2 == 0)
		return r;
	else
		return -r;
} //end of function

//function returns a random double value in the interval [0, 2PI]
double randomDoubleAngle(){

	//generate random double between 0 and 1
	double r = fabs(randomDouble());

	//scale to the 0 to 2PI range
	r *= (2*PI);

	return r;
} //end of function

void rotate(Matrix G1, Matrix G2, double centerX, double centerY){

	double theta = randomDoubleAngle();
	for(int i = 0; i < G1.height; i++){
		G2.elements[i*G2.width] = (cos(theta))*(G1.elements[i*2] - centerX) - (sin(theta))*(G1.elements[i*2 + 1] - centerY) + centerX;
		G2.elements[i*G2.width + 1] = (sin(theta))*(G1.elements[i*2] - centerX) + (cos(theta))*(G1.elements[i*2 + 1] - centerY) + centerX;
	}

} // end of function

void translate(Matrix G1, Matrix G2){
	double distortionX = randomDouble();
	double distortionY = randomDouble();
	for(int i=0; i < G1.height; i++){
		G2.elements[i*G2.width] = distortionX + G1.elements[i*G1.width];
		G2.elements[i*G2.width + 1] = distortionY + G1.elements[i*G1.width + 1];
	}
} // end of function

void graphDistortion(Matrix G1, Matrix G2, double centerX, double centerY){

	rotate(G1, G2, centerX, centerY);
	translate(G1, G2);

} // end of function

__global__
void neighborDistancesKernel(Matrix d_G, Matrix d_dist){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_A.width+col;
	if(row > d_dist.height || col > d_dist.width) return;
	double Xdist, Ydist;

	if(row == col)
		d_dist.elements[idx] = 0;
	else{
		Xdist = d_G.elements[row*d_G.width] - d_G.elements[col*d_G.width];
		Ydist = d_G.elements[row*d_G.width + 1] - d_G.elements[col*d_G.width + 1];
		d_dist.elements[idx] = sqrt( Xdist*Xdist + Ydist*Ydist);
	}

}

void neighborDistances(Matrix G, Matrix dist){

	// load G to device memory
	Matrix d_G;
	d_G.width = G.width;
	d_G.height = G.height;
	size_t sizeG = G.width * G.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_G.elements, sizeG);
	printf("CUDA malloc G: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_G.elements, G.elements, sizeG, cudaMemcpyHostToDevice);	
	printf("Copy G to device: %s\n", cudaGetErrorString(err));

	// load dist to device memory
	Matrix d_dist;
	d_dist.width = dist.width;
	d_dist.height = dist.height;
	size_t size = dist.width * dist.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_dist.elements, size);
	printf("CUDA malloc dist: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_dist.elements, dist.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy dist to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (dist.width + dimBlock.x - 1)/dimBlock.x, (dist.height + dimBlock.y - 1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_G, d_dist);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(dist.elements, d_dist.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy dist off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_G.elements);
	cudaFree(d_dist.elements);

}

void main(){

	clock_t start = clock(), diff;

	srand(time(0));

//create random set of points
	int size = TEST_SIZE;
	Matrix G1;
	G1.width = 2;
	G1.height = size;
	double* G1.elements = (double *)malloc(size*2*sizeof(double));
	for(int i=0; i < size; i++){
		G1.elements[i*2] = randomdouble();
		G1.elements[i*2 + 1] = randomdouble();
	}

//copy original set of points for distortion
	Matrix G2;
	G2.width = 2;
	G2.height = G1.height;
	double* G2.elements = (double *)malloc(size*2*sizeof(double));
	for(int i=0; i < size; i++){
		G2.elements[i*2] = G1.elements[i*2];
		G2.elements[i*2 + 1] = G1[i*2 + 1];
	}

//distort the graph for testing purposes
	graphDistortion(G1, G2, 0, 0);

//calculate the distances to each neighbor
	Matrix neighborDist1, neighborDist2;
	neighborDist1.width = neighborDist2.width = size;
	nieghborDist1.height = neighborDist2.height = size;
	double* neighborDist1.elements = (double *) malloc(size*size*sizeof(double)); 
	double* neighborDist2.elements = (double *) malloc(size*size*sizeof(double));
	neighborDistances(G1, neighborDist1);
	neighborDistances(G2, neighborDist2);

//free up memory that was dynamically allocated
	free(G1);
	free(G2);

	Matrix X, Y, Z;
	X.width = Y.width = Z.width = size;
	X.height = Y.height = Z.height = size;
	double *X = (double *)malloc(size*size*sizeof(double));
	double *Z = (double *)malloc(size*size*sizeof(double));
	double *Y = (double *)malloc(size*size*sizeof(double));
	zeros(size, size, X);
	zeros(size, size, Z);
	zeros(size, size, Y);

	printf("Got to graph matching\n");
	graphMatching(neighborDist1,neighborDist2, 0.01, size, X, Z, Y);

	printf("Finished\n");

	free(neighborDist1);
	free(neighborDist2);

	printf("X(hard):\n");
	printMatrix(size, size, X);
	printf("Z(soft):\n");
	printMatrix(size, size, Z);
	printf("Y(debug):\n");
	printMatrix(size, size, Y);

	free(X);
	free(Z);
	free(Y);

	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

}
