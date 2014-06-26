#include "matlib.cu"
#define EPS 2.220446049250313e-16

void exactTotalSum(Matrix y, Matrix h, double totalSum, double precision, Matrix x) {
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

}
