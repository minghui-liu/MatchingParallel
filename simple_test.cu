#include <stdio.h>
#include "graphMatching.cu"
//#include "utils.cu"
#include "matlib.cu"

int main() {
	Matrix G1, G2;
	G1.height = G1.width = 3;
	G2.height = G2.width = 4;
	G1.elements = (float*)malloc(G1.height*G1.width*sizeof(float));
	G2.elements = (float*)malloc(G2.height*G2.width*sizeof(float));
	float G1E[3][3] = {{0, 4, 3},
						{4, 0, 5},
						{3, 5, 0}};
	float G2E[4][4] = {{0.0, 3.8, 2.9, 0.0},
						{3.8, 0.0, 5.1, 0.0},
						{2.9, 5.1, 0.0, 0.0},
						{0.0, 0.0, 0.0, 0.0}};
	memcpy(G1.elements, G1E, G1.height*G1.width*sizeof(float));
	memcpy(G2.elements, G2E, G2.height*G2.width*sizeof(float));
	
	printf("G1:\n");
	printMatrix(G1);
	printf("G2:\n");
	printMatrix(G2);

	Matrix X, Y, Z;
	X.height = Y.height = Z.height  = G1.width;
	X.width = Y.width = Z.width = G2.width;
	X.elements = (float*)malloc(X.height*X.width*sizeof(float));
	Y.elements = (float*)malloc(Y.height*Y.width*sizeof(float));
	Z.elements = (float*)malloc(Z.height*Z.width*sizeof(float));
	
	// sigma = 1, numberOfMatches = 3
	graphMatching(G1, G2, 1, 3, X, Z, Y);

	printf("X(hard):\n");
	printMatrix(X);
	printf("Z(soft):\n");
	printMatrix(Z);
	printf("Y(debug):\n");
	printMatrix(Y);
}
