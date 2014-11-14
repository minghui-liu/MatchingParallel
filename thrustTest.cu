#include <stdio.h>
#include "utils.cu"
#include <curand.h>
#include <cuda.h>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

#define NODES 1024

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

	CURAND_CALL(curandGenerateUniformfloat(gen, devData, n));
	CURAND_CALL(curandGenerateUniformfloat(gen1, devData1, m));

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

	thrust::host_vector<double> H(hostData, hostData + NODES);

	std::cout << "H has size " << H.size() << std::endl;

	thrust::device_vector<double> D = H;

//	double* maxResult = thrust::max_element(thrust::host, D.begin(), D.end());

	thrust::detail::normal_iterator<thrust::device_ptr<double> > maxResult = thrust::max_element(D.begin(), D.end());

	//thrust::host_vector<double> h_maxResult = maxResult;

	std::cout << "maxResult is " << *maxResult << std::endl;

}
