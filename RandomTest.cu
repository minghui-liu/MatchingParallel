/*
* file: RandomTest.c
*
* Testing the probablistic graph matching algorithm
* by rotating a set of artificial points and then calculating
* the similarity score for the edges
*
* Kevin Liu & Reid Delaney
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "graphMatching.cu"
#include "matlib.cu"

#define PI 3.14159265
#define TEST_SIZE	28 

//function returns a random float value in the interval [0, 2PI]
float randomfloatAngle() {
	//generate random float between 0 and 1
  float r = (float)rand()/(float)RAND_MAX;
  //scale to the 0 to 2PI range
  r *= (2*PI);
  return r;
}

//function returns a random float value in the interval [0,1]
float randomfloat() {
  //generate random float between 0 and 1
  float r = (float)rand() / (float)RAND_MAX;
  return r;
}

//function takes in a set of points, rotates them and then returns the new set
void rotate(Matrix V1, Matrix V2, float centerX, float centerY) {
  float theta = randomfloatAngle();
  for(int i=0; i < V1.height; i++){
    *(V2.elements + i * V2.width) = cos(theta)*(*(V1.elements + i * V1.width) - centerX) - sin(theta)*(*(V1.elements + i * V1.width + 1) - centerY) + centerX;
    *(V2.elements + i * V2.width + 1) = sin(theta)*(*(V1.elements + i * V1.width) - centerX) + cos(theta)*(*(V1.elements + i * V1.width + 1) - centerY) + centerY;
  }
}

void pointDistort(Matrix V1, Matrix V2, float centerX, float centerY) {
  float distortionX;
  float distortionY;
  for(int i=0; i < V1.height; i++){
    distortionX = (2*randomfloat()-1)/100;
    distortionY = (2*randomfloat()-1)/100;
    *(V2.elements + i * V2.width) = distortionX + *(V1.elements + i * V1.width);
    *(V2.elements + i * V2.width + 1) = distortionY + *(V2.elements + i * V2.width + 1);
  }
}

//create a matrix of distances between nodes
void neighborDistances(Matrix V1, Matrix neighborDist) {
  float distance = 0;
  
  for(int i = 0; i < V1.height; i++) {
    for(int j = 0; j < V1.height; j++) {
      if(i == j)
        *(neighborDist.elements + i * neighborDist.width + j) = 0;
      else {
        distance = sqrt((*(V1.elements+i*V1.width) - *(V1.elements+j*V1.width))*(*(V1.elements+i*V1.width) - *(V1.elements+j*V1.width)) + (*(V1.elements+i*V1.width+1) - *(V1.elements+j*V1.width+1))*(*(V1.elements+i*V1.width+1)- *(V1.elements+j*V1.width+1)));
        *(neighborDist.elements+i*neighborDist.width+j) = distance;
      }
    }
  }

}

int main(int argc, char *argv[]) {
	// initialize random generator with time as seed
  srand(time(NULL));

  int size = atoi(argv[1]);
  
  Matrix V1, V2;
  V1.width = V2.width = 2;
  V1.height = V2.height = size;
  V1.elements = (float*)malloc(size*2*sizeof(float));
  V2.elements = (float*)malloc(size*2*sizeof(float));
  
  for(int i=0; i < size; i++) {
    *(V2.elements + i * V2.width) = *(V1.elements + i * V1.width) = randomfloat();
    *(V2.elements + i * V2.width + 1) = *(V1.elements + i * V1.width + 1) = randomfloat();
  }
  printMatrix(V1);
	saveMatrix(V1, "output/nodes1.txt");
  
  rotate(V1, V2, 0.5, 0.5);
  pointDistort(V2, V2, 0, 0);
  
  printMatrix(V2);
	saveMatrix(V2, "output/nodes2.txt");
  
 	Matrix neighborDist1, neighborDist2;
	neighborDist1.width = neighborDist1.height = size;
	neighborDist2.width = neighborDist2.height = size;
	neighborDist1.elements = (float*)malloc(size*size*sizeof(float));
	neighborDist2.elements = (float*)malloc(size*size*sizeof(float));
	
  neighborDistances(V1, neighborDist1);
  neighborDistances(V2, neighborDist2);
  
  printf("neighbor Distances 1\n");
  printMatrix(neighborDist1);
	saveMatrix(neighborDist1, "output/edges1.txt");

  printf("neighbor distances 2\n");
  printMatrix(neighborDist2);
	saveMatrix(neighborDist2, "output/edges2.txt");
  
	Matrix X, Y, Z;
	X.width = X.height = size;
	Y.width = Y.height = size;
	Z.width = Z.height = size;
	X.elements = (float *)malloc(size*size*sizeof(float));
	Y.elements = (float *)malloc(size*size*sizeof(float));
	Z.elements = (float *)malloc(size*size*sizeof(float));
	
  graphMatching(neighborDist1, neighborDist2, 1, size, X, Z, Y);
  
  printf("X(hard):\n");
  printMatrix(X);
  printf("Z(soft):\n");
  printMatrix(Z);
  printf("Y(debug):\n");
  printMatrix(Y);
	saveMatrix(X, "output/hard.txt");
	saveMatrix(Z, "output/soft.txt");
}
