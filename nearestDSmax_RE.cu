#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include "matlib.cu"
#include "exactTotalSum.cu"
#include "maxColSumP.cu"

#define BLOCK_SIZE 32
#define EPS 2.2204e-16


__global__
void HKernel(Matrix d_A, Matrix d_B, Matrix d_C, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	int idx = row*d_A.width+col;
	d_Out.elements[idx] = d_A.elements[idx] - (d_B.elements[idx] / (d_C.elements[idx]+EPS));
}

void H(Matrix A, Matrix B, Matrix C, Matrix Out) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	//printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));

	// load C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	//printf("CUDA malloc C: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix C to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	HKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run H kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	cudaFree(d_Out.elements);
}

// matrix lambda kernel called by lambda()
__global__
void lambdaKernel(Matrix d_A, Matrix d_B, Matrix d_C, Matrix d_D, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	int idx = row*d_A.width+col;
	d_Out.elements[idx] = d_A.elements[idx] - (d_B.elements[idx] / (d_C.elements[idx]+EPS)) + (d_B.elements[idx] / (d_D.elements[idx]+EPS));
}

void lambda(Matrix A, Matrix B, Matrix C, Matrix D, Matrix Out) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	//printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));

	// load C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	//printf("CUDA malloc C: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix C to device: %s\n", cudaGetErrorString(err));
	
	// load C to device memory
	Matrix d_D;
	d_D.width = D.width;
	d_D.height = D.height;
	err = cudaMalloc(&d_D.elements, size);
	//printf("CUDA malloc D: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_D.elements, D.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix D to device: %s\n", cudaGetErrorString(err));

	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	err = cudaMalloc(&d_Out.elements, size);
	//printf("CUDA malloc Out: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	lambdaKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_D, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run lambda kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	cudaFree(d_D.elements);
	cudaFree(d_Out.elements);
}

__global__
void FKernel(Matrix d_A, Matrix d_B, Matrix d_C, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_A.width+col;
	if(row >= d_A.height || col >= d_A.width) return;
	d_Out.elements[idx] = (d_A.elements[idx] + d_B.elements[idx] + d_C.elements[idx]) / 3;
}

void Fun(Matrix A, Matrix B, Matrix C, Matrix Out) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	//printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));

	// load C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	//printf("CUDA malloc C: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix C to device: %s\n", cudaGetErrorString(err));

	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	err = cudaMalloc(&d_Out.elements, size);
	//printf("CUDA malloc Out: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	FKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	cudaFree(d_Out.elements);
}

void nearestDSmax_RE(Matrix Y, Matrix maxRowSum, Matrix maxColSum, double totalSum, double maxLoops, double precision, Matrix F) {
	int size = Y.height * Y.width * sizeof(double);

	// lambda1 = zeroes(size(Y));
	Matrix lambda1, lambda2, lambda3;
	lambda1.height = lambda2.height = lambda3.height = Y.height;
	lambda1.width = lambda2.width = lambda3.width = Y.width;
	lambda1.elements = (double*)malloc(size);
	lambda2.elements = (double*)malloc(size);
	lambda3.elements = (double*)malloc(size);
	zeros(lambda1);
	// lambda2 = lambda1;  lambda3 = lambda1;
	memcpy(lambda2.elements, lambda1.elements, size);
	memcpy(lambda3.elements, lambda1.elements, size);

	// F1 = totalsum * (Y ./ sum(Y(:)));
	Matrix F1, F2, F3;
	F1.height = F2.height = F3.height = Y.height;
	F1.width = F2.width = F3.width = Y.width;
	F1.elements = (double*)malloc(size);
	F2.elements = (double*)malloc(size);
	F3.elements = (double*)malloc(size);
	
	//printf("before sum(Y(:))\n");
	// sum(Y(:))
	thrust::host_vector<double> h_Y(Y.elements, Y.elements + Y.width * Y.height);
	thrust::device_vector<double> d_Y = h_Y;
	double Ysum = thrust::reduce(d_Y.begin(), d_Y.end(), (double) 0, thrust::plus<double>());
	//printf("after sum(Y(:))\n");

	// Y ./ sum(Y(:))
	Matrix YdivYsum;
	YdivYsum.width = Y.width;
	YdivYsum.height = Y.height;
	YdivYsum.elements = (double*)malloc(size);
	matTimesScaler(Y, 1/Ysum, YdivYsum);
	matTimesScaler(YdivYsum, totalSum, F1);
	// F2 = F1;  F3 = F1;
	memcpy(F2.elements, F1.elements, size);
	memcpy(F3.elements, F1.elements, size);

	Matrix H1, H2, H3;
	H1.width = H2.width = H3.width = Y.width;
	H1.height = H2.height = H3.height = Y.height;
	H1.elements = (double*)malloc(size);
	H2.elements = (double*)malloc(size);
	H3.elements = (double*)malloc(size);

	Matrix F1eps, F2eps, F3eps;
	F1eps.width = F2eps.width = F3eps.width = Y.width;
	F1eps.height = F2eps.height = F3eps.height = Y.height;
	F1eps.elements = (double*)malloc(size);
	F2eps.elements = (double*)malloc(size);
	F3eps.elements = (double*)malloc(size);

	Matrix YdivF1eps, YdivF2eps, YdivF3eps;
	YdivF1eps.width = YdivF2eps.width = YdivF3eps.width = Y.width;
	YdivF1eps.height = YdivF2eps.height = YdivF3eps.height = Y.height;
	YdivF1eps.elements = (double*)malloc(size);
	YdivF2eps.elements = (double*)malloc(size);
	YdivF3eps.elements = (double*)malloc(size);

	Matrix negH2, negH3;
	negH2.width = negH3.width = Y.width;
	negH2.height = negH3.height = Y.height;
	negH2.elements = (double*)malloc(size);
	negH3.elements = (double*)malloc(size);

	// transposed matrices
	Matrix H1t, negH1t, Yt, F1t, negH3t;
	H1t.width = negH1t.width = Yt.width = F1t.width = Y.height;
	H1t.height = negH1t.height = Yt.height = F1t.height = Y.width;
	negH3t.height = H3.width;
	negH3t.width = H3.height;
	negH3t.elements = (double*)malloc(size);
	H1t.elements = (double*)malloc(size);
	negH1t.elements = (double*)malloc(size);
	Yt.elements = (double*)malloc(size);
	F1t.elements = (double*)malloc(size);

	Matrix Fdiff1, Fdiff2;
	Fdiff1.width = Fdiff2.width = Y.width;
	Fdiff1.height = Fdiff2.height = Y.height;
	Fdiff1.elements = (double*)malloc(size);
	Fdiff2.elements = (double*)malloc(size); 

	// F3reshape is a col vector
	Matrix F3reshape;
	F3reshape.width = 1;
	F3reshape.height = Y.width*Y.height;
	F3reshape.elements = (double*)malloc(size);
			
	Matrix maxRowSumT;
	maxRowSumT.width = Y.height;
	maxRowSumT.height = 1;
	maxRowSumT.elements = (double*)malloc(maxRowSumT.width*sizeof(double));

	//for t = 1 : maxLoops
	for(int t=0; t < maxLoops; t++) {
	// Max row sum
		// H1 = lambda1 - (Y ./ (F3+eps));
		H(lambda1, Y, F3, H1);
		//F1 = maxColSumP(Y', -H1', maxRowSum', precision)';
		//-H1'
		transpose(H1, H1t);
		matTimesScaler(H1t, -1, negH1t);
		//Y'
		transpose(Y, Yt);
		//maxRowSum'
		transpose(maxRowSum, maxRowSumT);
	//	transpose(F1, F1t);
		//maxColSumP(Y', -H1', maxRowSum', precision)'
		maxColSumP(Yt, negH1t, maxRowSumT, 0.01, F1t);
		//F1
		transpose(F1t, F1);

		// lambda1 = lambda1 - (Y ./ (F3+eps)) + (Y ./ (F1+eps));
		lambda(lambda1, Y, F3, F1, lambda1);
		
	// Max col sum
		// H2 = lambda2 - (Y ./ (F1+eps));
		H(lambda2, Y, F1, H2);
		// F2 = maxColSumP (Y, -H2, maxColSum, precision);
		matTimesScaler(H2, -1, negH2);
		maxColSumP(Y, negH2, maxColSum, precision, F2);
		// lambda2 = lambda2 - (Y ./ (F1+eps)) + (Y ./ (F2+eps));
		lambda(lambda2, Y, F1, F2, lambda2);
	
	// Total sum
		// H3 = lambda3 - (Y ./ (F2 + eps));
		H(lambda3, Y, F2, H3);
		matTimesScaler(H3, -1, negH3);
		// F3 = reshape( exactTotalSum (Y(:), -H3(:), totalSum, precision), size(Y) );
		transpose(Y, Yt);
		transpose(negH3, negH3t);
		exactTotalSum(Yt, negH3t, totalSum, precision, F3reshape);
		reshape(F3reshape, F3);

		//lambda3 = lambda3 - (Y ./ (F2+eps)) + (Y ./ (F3+eps));
		lambda(lambda3, Y, F2, F3, lambda3);
		matSub(F1, F2, Fdiff1);
		matSub(F1, F3, Fdiff2);

		// max and min of Fdiff1
		thrust::host_vector<double> h_Fdiff1(Fdiff1.elements, Fdiff1.elements + Fdiff1.width*Fdiff1.height);
		thrust::device_vector<double> d_Fdiff1 = h_Fdiff1;
		thrust::detail::normal_iterator<thrust::device_ptr<double> > Fdiff1max = thrust::max_element(d_Fdiff1.begin(), d_Fdiff1.end());
		thrust::detail::normal_iterator<thrust::device_ptr<double> > Fdiff1min = thrust::min_element(d_Fdiff1.begin(), d_Fdiff1.end());
		
		// max and min of Fdiff2
		thrust::host_vector<double> h_Fdiff2(Fdiff2.elements, Fdiff2.elements + Fdiff2.width*Fdiff2.height);
		thrust::device_vector<double> d_Fdiff2 = h_Fdiff2;
		thrust::detail::normal_iterator<thrust::device_ptr<double> > Fdiff2max = thrust::max_element(d_Fdiff2.begin(), d_Fdiff2.end());
		thrust::detail::normal_iterator<thrust::device_ptr<double> > Fdiff2min = thrust::min_element(d_Fdiff2.begin(), d_Fdiff2.end());

		double fdMax1 = max(*Fdiff1max, fabs(*Fdiff1min));
		double fdMax2 = max(*Fdiff2max, fabs(*Fdiff2min));
	
		if(fabs(fdMax1) < precision && fabs(fdMax2) < precision)
			break;
	
	} // end of t for loop
	
	// F = (F1 + F2 + F3) / 3;
	Fun(F1, F2, F3, F);

	free(lambda1.elements);
	free(lambda2.elements);
	free(lambda3.elements);
	free(F1.elements);
	free(F2.elements);
	free(F3.elements);
	free(H1.elements);
	free(H2.elements);
	free(H3.elements);
	free(F1eps.elements);
	free(F2eps.elements);
	free(F3eps.elements);
	free(YdivF1eps.elements);
	free(YdivF2eps.elements);
	free(YdivF3eps.elements);
	free(negH1t.elements);
	free(negH2.elements);
	free(negH3.elements);
	free(H1t.elements);
	free(Yt.elements);
	free(maxRowSumT.elements);
	free(F1t.elements);
	free(F3reshape.elements);
	free(Fdiff1.elements);
	free(Fdiff2.elements);
}
