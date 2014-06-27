#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#define BLOCK_SIZE 32
#define BLOCK_SIZE_DIM1 1024

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

// Usage: matTimesScaler
int main(int argc, char* argv[]){
	
	Matrix A;
	A.width = 3; A.height = 3;
	A.elements = (double*)malloc(A.height*A.width*sizeof(double));
	double AE[3][3] = {{1, 4, 7},{2, 5, 8},{3, 6, 9}};
	memcpy(A.elements, AE, A.height*A.width*sizeof(double));
	
	printf("A:\n");
	printMatrix(A);

	thrust::host_vector<double> h_A(A.elements, A.elements + A.width*A.height);
	thrust::device_vector<double> d_A = h_A;
	double Asum = thrust::reduce(d_A.begin(), d_A.end(), (double) 0, thrust::plus<double>());

	printf("Sum = %.2f\n", Asum);

	free(A.elements);
}


