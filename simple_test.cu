#include <stdio.h>
#include "graphMatching.cu"
//#include "test_utils.cu"
//#include "utils.cu"

int main() {
	Matrix G1, G2;
	G2.width = G1.width = 3;
	G2.height = G1.height  = 3;
	G1.elements = (double*)malloc(G1.height*G1.width*sizeof(double));
	G2.elements = (double*)malloc(G2.height*G2.width*sizeof(double));
	double G1E[3][3] = {{0,4,3},{4,0,5},{3,5,0}};
	double G2E[3][3] = {{0,3.8,2.9},{3.8,0,5.1},{2.9,5.1,0}};
	memcpy(G1.elements, G1E, G1.height*G1.width*sizeof(double));
	memcpy(G2.elements, G2E, G2.height*G2.width*sizeof(double));
	
	printf("G1:\n");
	printMatrix(G1);
	printf("G2:\n");
	printMatrix(G2);

	Matrix X, Y, Z;
	X.width = Y.width = Z.width = 3;
	X.height = Y.height = Z.height  = 3;
	X.elements = (double*)malloc(X.height*X.width*sizeof(double));
	Y.elements = (double*)malloc(Y.height*Y.width*sizeof(double));
	Z.elements = (double*)malloc(Z.height*Z.width*sizeof(double));
	
	// sigma = 1, numberOfMatches = 3
	graphMatching(G1, G2, 1, 3, X, Z, Y);

	printf("X(hard):\n");
	printMatrix(X);
	printf("Z(soft):\n");
	printMatrix(Z);
	printf("Y(debug):\n");
	printMatrix(Y);
}
