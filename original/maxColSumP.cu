#include "matlib.cu"
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
	// load  to device memory
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
	unconstrainedPKernel<<<dimGrid, dimBlock>>>(d_X);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(X.elements, d_X.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy X off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_X.elements);

}

void maxColSumP(Matrix Y, Matrix H, Matrix maxColSum, double precision, Matrix X) {
	printf("maxColSumP()\n");
	unconstrainedP(Y, H, X);

	Matrix Xsum;
	Xsum.height = 1;
	Xsum.width = X.width;
	Xsum.elements = (double*)malloc(X.width * sizeof(double));
	
	// Xsum = sum(X)
	//sumOfMatrixCol(X, Xsum); // write a host function for this kernel
	// temporary this is not very well parallelized
	for(int i=0; i < X.height; i++){
		thrust::host_vector<double> h_Xsum(X.elements + i*X.width, X.elements + i*X.width + X.width);
		thrust::device_vector<double> d_Xsum = h_Xsum;
		Xsum.elements[i] = thrust::reduce(d_Xsum.begin(), d_Xsum.end(), (double) 0, thrust::plus<double>());
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
		if(Xsum.elements[i] > maxColSum.elements[i]) {
			printf("Xsum: \n");
			printMatrix(Xsum);
			printf("maxColSum:\n");
			printMatrix(maxColSum);
			//X(:,i) = exactTotalSum (Y(:,i), H(:,i), maxColSum(i), precision);
			printf("****************\nbefore getCol(Y, yCol, i)\n\n");
			printf("i = %d\n", i);
			printf("Xsum.width = %d\n", Xsum.width);
			printMatrix(Y);
			printMatrix(yCol);
			getCol(Y, yCol, i);
			printf("****************\nafter getCol(Y, yCol, i)\n");
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

