#include "matlib.cu"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#define EPS 2.2204e-16

__global__
void unconstrainedPKernel(Matrix d_X) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_X.width+col;
	if(row >= d_X.height || col >= d_X.width) return;
	if(d_X.elements[idx] < EPS)
		d_X.elements[idx] = EPS;
}

void unconstrainedP(Matrix Y, Matrix H, Matrix X) {
	printf("uncontraindedP()\n");

	matDiv(Y, H, X);
	
	// load X to device memory
	Matrix d_X;
	d_X.width = X.width;
	d_X.height = X.height;
	size_t size = X.width * X.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_X.elements, size);
	//printf("CUDA malloc X: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_X.elements, X.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (X.width + dimBlock.x - 1)/dimBlock.x, (X.height + dimBlock.y - 1)/dimBlock.y );
	unconstrainedPKernel<<<dimGrid, dimBlock>>>(d_X);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read X from device memory
	err = cudaMemcpy(X.elements, d_X.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy X off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_X.elements);

}

/*void exactTotalSum(Matrix y, Matrix h, double totalSum, double precision, Matrix x) {
	printf("exactTotalSum()\n");
	// y and h are vectors, totalSum and precision are scalars
	// x is the return vector and length is the length of y, h, and x
	
	// allocate hAlpha
	Matrix hAlpha;
	hAlpha.width = h.width;
	hAlpha.height = h.height;
	hAlpha.elements = (double*)malloc(hAlpha.width * hAlpha.height * sizeof(double));

	double totalSumMinus = totalSum - precision;

	thrust::host_vector<double> H_h(h.elements, h.elements + h.width*h.height);
	thrust::device_vector<double> D_h = H_h;
	thrust::detail::normal_iterator<thrust::device_ptr<double> > MinIt = thrust::min_element(D_h.begin(), D_h.end());
	double Min = *MinIt;
	double curAlpha = -Min + EPS;

	double stepAlpha, newAlpha, newSum;
	stepAlpha = (10 > fabs(curAlpha/10))? 10 : fabs(curAlpha/10);

	for(int j=0; j < 50; j++) {
		newAlpha = curAlpha + stepAlpha;
		newSum = 0;

		matPlusScaler(h, newAlpha, hAlpha);
		matDiv(y, hAlpha, x);

		thrust::host_vector<double> H_x(x.elements, x.elements + x.width * x.height);
		thrust::device_vector<double> D_x = H_x;
		newSum = thrust::reduce(D_x.begin(), D_x.end(), (double) 0, thrust::plus<double>());

		if(newSum > totalSum) {
			curAlpha = newAlpha;
		} else {
			if (newSum < totalSumMinus)
				stepAlpha = stepAlpha / 2;
			else return;
		}

		// empty vectors
		H_x.clear();
		D_x.clear();
		// deallocate any capacity which may currently be associated with vectors
		H_x.shrink_to_fit();
		D_x.shrink_to_fit();
	}

}*/

void maxColSumP(Matrix Y, Matrix H, Matrix maxColSum, double precision, Matrix X) {
	printf("maxColSumP()\n");
	// unconstrainedP is clean	
	unconstrainedP(Y, H, X);

	Matrix Xsum;
	Xsum.height = 1;
	Xsum.width = X.width;
	Xsum.elements = (double*)malloc(X.width * sizeof(double));
	
	Matrix Xcol;
	Xcol.width = 1; Xcol.height = X.height;
	Xcol.elements = (double*)malloc(Xcol.height * sizeof(double));
	
	for (int i=0; i<X.width; i++) {
		getCol(X, Xcol, i);
	//	printf("Xcol - %d",i);
	//	printMatrix(Xcol);
		thrust::host_vector<double> h_Xcol(Xcol.elements, Xcol.elements + Xcol.height);
		thrust::device_vector<double> d_Xcol = h_Xcol;
		Xsum.elements[i] = thrust::reduce(d_Xcol.begin(), d_Xcol.end(), (double) 0, thrust::plus<double>());
		
		// empty vectors
		h_Xcol.clear();
		d_Xcol.clear();
		// deallocate any capacity which may currently be associated with vectors
		h_Xcol.shrink_to_fit();
		d_Xcol.shrink_to_fit();	
	}

	// !!!
	// !!! DOES NOT RETURN THE CORRECT SUM
	// !!!
	//sumOfMatrixCol(X, Xsum);
/*	printf("Y\n");
	printMatrix(Y);
	printf("H\n");
	printMatrix(H);
	printf("X:\n");
	printMatrix(X);
	printf("Xsum:\n");
	printMatrix(Xsum);
	printf("MaxColSum\n");
	printMatrix(maxColSum);
*/
	Matrix yCol, hCol;
	yCol.width = 1;
	hCol.width = 1;
	//Xcol.width = 1;
	yCol.height = Y.height;
	hCol.height = H.height;
	//Xcol.height = X.height;
	yCol.elements = (double*)malloc(Y.height * sizeof(double));
	hCol.elements = (double*)malloc(H.height * sizeof(double));
	//Xcol.elements = (double*)malloc(X.height * sizeof(double));

	for(int i=0; i < Xsum.width; i++) {
		if(Xsum.elements[i] > maxColSum.elements[i]) {
			//printf("i is now %d\n", i);
			//X(:,i) = exactTotalSum (Y(:,i), H(:,i), maxColSum(i), precision);
			getCol(Y, yCol, i);
			getCol(H, hCol, i);
			exactTotalSum(yCol, hCol, maxColSum.elements[i], precision, Xcol);
			for(int j=0; j < X.height; j++) {
				X.elements[j*X.width + i] = Xcol.elements[j];
			}
		}
	}

	free(yCol.elements);
	free(hCol.elements);
	free(Xcol.elements);
	free(Xsum.elements);
}

