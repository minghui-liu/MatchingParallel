#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "utils.c"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_DIM2 32
#define EPS 2.2204e-16

typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;

void exactTotalSum(Matrix y, Matrix h, double totalSum, Matrix X){

// y and h are vectors, totalSum and precision are scalars
// X is the return vector and length is the length of y, h, and X
	double totalSumMinus = totalSum - precision;
	double curAlpha;

	double Min = minOfArray(h, length);

	curAlpha = -Min + EPS;

	double stepAlpha, newAlpha, newSum;
	if(10 > fabs(curAlpha/10))
		stepAlpha = 10;
	else
		stepAlpha = fabs(curAlpha/10);

	for(int j=0; j < 50; j++){

		newAlpha = curAlpha + stepAlpha;
		newSum = 0;

		matPlusScaler(h, hAlpha, newAlpha);
		matDiv(y, hAlpha, X);
		newSum = arraySum(X.elements[0]);

		if(newSum > totalSum) {
			curAlpha = newAlpha;
		} else {
			if (newSum < totalSumMinus)
				stepAlpha = stepAlpha / 2;
			else return;
		}

	}

} // end of function

void unconstrainedP(Matrix Y, Matrix H, Matrix X){

	matDiv(Y, H, X);

	for(int i=0; i < X.width; i++){
		for(int j=0; j < X.height; j++){
			if(X.elements[i*X.height + j] < EPS){
				X.elements[i*X.height + j] = EPS);
			}
		}
	}

} // end of function

void maxColSumP(Matrix Y, Matrix H, Matrix maxColSum, double precision, Matrix X){

	unconstrainedP(Y, H, X);

	Matrix Xsum;
	Xum.height = 1;
	Xsum.width = X.width;
	double* Xsum = (double*) malloc(X.width * sizeof(double));

	for(int i=0; i < X.height; i++){
		Xsum[i] = arraySum(X + i*X.width, X.width);
	}

	Matrix yCol, hCol, Xcol;
	yCol.width = 1;
	hCol.width = 1;
	Xcol.wdith = 1;
	yCol.height = Y.height;
	hCol.height = H.height;
	Xcol.height = X.height;
	double* yCol.elements = (double*)malloc(Y.height * sizeof(double));
	double* hCol.elements = (double*)malloc(H.height * sizeof(double));
	double* Xcol.elements = (double*)malloc(X.height * sizeof(double));

	for(int i=0; i < Xsum.width; i++) {
		if(Xsum[i] > maxColSum[i]){

//X(:,i) = exactTotalSum (Y(:,i), H(:,i), maxColSum(i), precision);
			getCol(Y, yCol, i);
			getCol(H, hCol, i);

			exactTotalSum(yCol, hCol, maxColSum[i], precision, Xcol);
			
			for(int j=0; j < x.width; j++){
				X[j*X.width + i] = Xcol[j];
			}

		}
	}

	cudaFree(yCol.elements);
	cudaFree(hCol.elements);
	cudaFree(Xcol.elements);
	cudaFree(Xsum.elements);

}

void nearestDSmax_RE(Matrix Y, Matrix maxRowSum, Matrix maxColSum, double totalSum, double precision, double maxLoops, Matrix F){

	zeros(F);
	int size = (double*)malloc(Y.width * Y.height * sizeof(double));

	Matrix lambda1, lambda2, lambda3;
	lambda1.width = Y.width;
	lambda2.width = Y.width;
	lambda3.width = Y.width;
	lambda1.height = Y.height;
	lambda2.height = Y.height;
	lambda3.height = Y.height;
	double* lambda1.elements = size;
	double* lambda2.elements = size;
	double* lambda3.elements = size;

	zeros(lambda1);
	zeros(lambda2);
	zeros(lambda3);

	Matrix F1, F2, F3;
	F1.width = Y.width;
	F2.width = Y.width;
	F3.width = Y.width;
	F1.height = Y.height;
	F2.height = Y.height;
	F3.height = Y.height;
	double* F1.elements = size;
	double* F2.elements = size;
	double* F3.elements = size;

	double Ysum = matSum(Y);

	matTimesScaler(matTimesScaler(Y, 1/Ysum, Ydiv), totalSum, F1);
	matTimesScaler(F1, 1, F2);
	matTimesScaler(F1, 1, F3);

	Matrix H1, H2, H3;
	H1.width = H2.width = H3.width = Y.width;
	H1.height = H2.height = H3.height = Y.height;
	double* H1.elements = size;
	double* H2.elements = size;
	double* H3.elements = size;


}
