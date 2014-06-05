#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "utils.cu"

//#define BLOCK_SIZE 1024
//#define BLOCK_SIZE_DIM2 32
#define EPS 2.2204e-16

/*typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;
*/
void exactTotalSum(Matrix y, Matrix h, double totalSum, double precision, Matrix X){

	Matrix hAlpha;
	hAlpha.width = h.width;
	hAlpha.height = h.height;
	hAlpha.elements = (double*)malloc(hAlpha.width * hAlpha.height * sizeof(double));

// y and h are vectors, totalSum and precision are scalars
// X is the return vector and length is the length of y, h, and X
	double totalSumMinus = totalSum - precision;
	double curAlpha;

	double Min = minOfArray(h.elements, h.height*h.width);

	curAlpha = -Min + EPS;

	double stepAlpha, newAlpha, newSum;
	if(10 > fabs(curAlpha/10))
		stepAlpha = 10;
	else
		stepAlpha = fabs(curAlpha/10);

	for(int j=0; j < 50; j++){

		newAlpha = curAlpha + stepAlpha;
		newSum = 0;

		matPlusScaler(h, newAlpha, hAlpha);
		matDiv(y, hAlpha, X);
		newSum = arraySum(X.elements, X.width*X.height);

		if(newSum > totalSum) {
			curAlpha = newAlpha;
		} else {
			if (newSum < totalSumMinus)
				stepAlpha = stepAlpha / 2;
			else return;
		}

	}

} // end of function

__global__
void unconstrainedKernel(Matrix d_X){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_X.width+col;
	if(row > d_X.height || col > d_X.width) return;
	if(d_X.elements[idx] < EPS)
		d_X.elements[idx] = EPS;
}

void unconstrainedP(Matrix Y, Matrix H, Matrix X){

	matDiv(Y, H, X);
	
// load A to device memory
	Matrix d_X;
	d_X.width = X.width;
	d_X.height = X.height;
	size_t size = X.width * X.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_X.elements, size);
	printf("CUDA malloc X: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_X.elements, X.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (X.width + dimBlock.x - 1)/dimBlock.x, (X.height + dimBlock.y - 1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_X);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(X.elements, d_X.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy X off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_X.elements);

} // end of function

void maxColSumP(Matrix Y, Matrix H, Matrix maxColSum, double precision, Matrix X){

	unconstrainedP(Y, H, X);

	Matrix Xsum;
	Xsum.height = 1;
	Xsum.width = X.width;
	Xsum.elements = (double*) malloc(X.width * sizeof(double));

	for(int i=0; i < X.height; i++){
		Xsum.elements[i] = arraySum(X.elements + i*X.width, X.width);
	}

	Matrix yCol, hCol, Xcol;
	yCol.width = 1;
	hCol.width = 1;
	Xcol.width = 1;
	yCol.height = Y.height;
	hCol.height = H.height;
	Xcol.height = X.height;
	yCol.elements = (double*)malloc(Y.height * sizeof(double));
	hCol.elements = (double*)malloc(H.height * sizeof(double));
	Xcol.elements = (double*)malloc(X.height * sizeof(double));

	for(int i=0; i < Xsum.width; i++) {
		if(Xsum.elements[i] > maxColSum.elements[i]){

//X(:,i) = exactTotalSum (Y(:,i), H(:,i), maxColSum(i), precision);
			getCol(Y, yCol, i);
			getCol(H, hCol, i);

			exactTotalSum(yCol, hCol, maxColSum.elements[i], precision, Xcol);
			
			for(int j=0; j < X.width; j++){
				X.elements[j*X.width + i] = Xcol.elements[j];
			}

		}
	}

	cudaFree(yCol.elements);
	cudaFree(hCol.elements);
	cudaFree(Xcol.elements);
	cudaFree(Xsum.elements);

}

// matrix matDiv kernel called by matDiv()
__global__
void HKernel(Matrix d_A, Matrix d_B, Matrix d_C, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_A.width+col;
	if(row > d_A.height || col > d_A.width) return;
	d_Out.elements[idx] = d_A.elements[idx] - (d_B.elements[idx] / (d_C.elements[idx]+EPS));
}

void H(Matrix A, Matrix B, Matrix C, Matrix Out) {
	if (A.width != B.width || A.height != B.height) {
		printf("Input matrices must have the same dimension!\n");
		return;
	}
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));

	// load C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix C to device: %s\n", cudaGetErrorString(err));
	
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
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

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
	int idx = row*d_A.width+col;
	if(row > d_A.height || col > d_A.width) return;
	d_Out.elements[idx] = d_A.elements[idx] - (d_B.elements[idx] / (d_C.elements[idx]+EPS)) + (d_B.elements[idx] / (d_D.elements[idx]+EPS));
}

void lambda(Matrix A, Matrix B, Matrix C, Matrix D, Matrix Out) {
	if (A.width != B.width || B.width != C.width || A.height != B.height || B.height != C.height){
		printf("Input matrices must have the same dimension!\n");
		return;
	}
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));

	// load C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix C to device: %s\n", cudaGetErrorString(err));
	
	// load C to device memory
	Matrix d_D;
	d_D.width = D.width;
	d_D.height = D.height;
	err = cudaMalloc(&d_D.elements, size);
	printf("CUDA malloc D: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_D.elements, D.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix D to device: %s\n", cudaGetErrorString(err));

	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	lambdaKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_D, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	cudaFree(d_D.elements);
	cudaFree(d_Out.elements);

}

// F = (F1 + F2 + F3) / 3;
// matrix lambda kernel called by lambda()
__global__
void FKernel(Matrix d_A, Matrix d_B, Matrix d_C, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_A.width+col;
	if(row > d_A.height || col > d_A.width) return;
	d_Out.elements[idx] = (d_A.elements[idx] + d_B.elements[idx] + d_C.elements[idx]) / 3;
}

void Fun(Matrix A, Matrix B, Matrix C, Matrix Out) {
	if (A.width != B.width || B.width != C.width || A.height != B.height || B.height != C.height){
		printf("Input matrices must have the same dimension!\n");
		return;
	}
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));

	// load C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix C to device: %s\n", cudaGetErrorString(err));

	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	FKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	cudaFree(d_Out.elements);

}

void nearestDSmax_RE(Matrix Y, Matrix maxRowSum, Matrix maxColSum, double totalSum, double maxLoops, double precision, Matrix F){

	zeros(F);
	int m = Y.width;
	int n = Y.height;
	int size = m * n * sizeof(double);

	Matrix lambda1, lambda2, lambda3;
	lambda1.width = lambda2.width = lambda3.width = m;
	lambda1.height = lambda2.height = lambda3.height = n;
	lambda1.elements = (double*)malloc(size);
	lambda2.elements = (double*)malloc(size);
	lambda3.elements = (double*)malloc(size);

	zeros(lambda1);
	zeros(lambda2);
	zeros(lambda3);

	Matrix F1, F2, F3;
	F1.width = F2.width = F3.width = m;
	F1.height = F2.height = F3.height = n;
	F1.elements = (double*)malloc(size);
	F2.elements = (double*)malloc(size);
	F3.elements = (double*)malloc(size);

	double Ysum = matSum(Y);
	Matrix Ydiv;
	Ydiv.width = m;
	Ydiv.height = n;
	Ydiv.elements = (double*)malloc(size);
	matTimesScaler(Y, 1/Ysum, Ydiv);
	matTimesScaler(Ydiv, totalSum, F1);
	matTimesScaler(F1, 1, F2);
	matTimesScaler(F1, 1, F3);

	Matrix H1, H2, H3;
	H1.width = H2.width = H3.width = m;
	H1.height = H2.height = H3.height = n;
	H1.elements = (double*)malloc(size);
	H2.elements = (double*)malloc(size);
	H3.elements = (double*)malloc(size);

	Matrix F1eps, F2eps, F3eps;
	F1eps.width = F2eps.width = F3eps.width = m;
	F1eps.height = F2eps.height = F3eps.height = n;
	F1eps.elements = (double*)malloc(size);
	F2eps.elements = (double*)malloc(size);
	F3eps.elements = (double*)malloc(size);

	Matrix YdivF1eps, YdivF2eps, YdivF3eps;
	YdivF1eps.width = YdivF2eps.width = YdivF3eps.width = m;
	YdivF1eps.height = YdivF2eps.height = YdivF3eps.height = n;
	YdivF1eps.elements = (double*)malloc(size);
	YdivF2eps.elements = (double*)malloc(size);
	YdivF3eps.elements = (double*)malloc(size);

	Matrix negH1t, negH2, negH3;
	negH1t.width = negH2.width = negH3.width = m;
	negH1t.height = negH2.height = negH3.height = n;
	negH1t.elements = (double*)malloc(size);
	negH2.elements = (double*)malloc(size);
	negH3.elements = (double*)malloc(size);

	Matrix H1t, Yt, F1t, X, Yv, Xp;
	H1t.width = Yt.width = F1t.width = X.width = Yv.width = Xp.width = m;
	H1t.height = Yt.height = F1t.height = X.height = Yv.height = Xp.height = n;
	H1t.elements = (double*)malloc(size);
	Yt.elements = (double*)malloc(size);
	F1t.elements = (double*)malloc(size);
	X.elements = (double*)malloc(size);
	Yv.elements = (double*)malloc(size);
	Xp.elements = (double*)malloc(size);

	Matrix Fdiff1, Fdiff2;
	Fdiff1.width = Fdiff2.width = m;
	Fdiff1.height = Fdiff2.height = n;
	Fdiff1.elements = (double*)malloc(size);
	Fdiff2.elements = (double*)malloc(size); 

	Matrix maxRowSumT;
	maxRowSumT.width = m;
	maxRowSumT.height = 1;
	maxRowSumT.elements = (double*)malloc(size/n);

//for t = 1 : maxLoops
	for(int t=0; t < 50; t++){

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
		//maxColSumP(Y', -H1', maxRowSum', precision)'
		maxColSumP(Yt, negH1t, maxRowSumT, EPS, F1t);
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

		for(int i = 0; i < m*n; i++){
			Yv.elements[i] = Y.elements[i];
			negH3.elements[i] = H3.elements[i];
		}

		exactTotalSum(Yv, negH3, totalSum, precision, X);

		reshape(X, F3);

	//lambda3 = lambda3 - (Y ./ (F2+eps)) + (Y ./ (F3+eps));
		lambda(lambda3, Y, F2, F3, lambda3);

		matSub(F1, F2, Fdiff1);
		matSub(F1, F3, Fdiff2);
		double fdMax1 = max(maxOfMatrix(Fdiff1), fabs(minOfMatrix(Fdiff1)));
		double fdMax2 = max(maxOfMatrix(Fdiff2), fabs(minOfMatrix(Fdiff2)));

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
	free(H1t.elements);
	free(negH1t.elements);
	free(negH2.elements);
	free(negH3.elements);
	free(Yt.elements);
	free(maxRowSumT.elements);
	free(F1t.elements);
	free(X.elements);
	free(Yv.elements);
	free(Xp.elements);
	free(Fdiff1.elements);
	free(Fdiff2.elements);

}
