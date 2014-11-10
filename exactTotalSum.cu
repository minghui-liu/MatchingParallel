#include "matlib.cu"
#define EPS 2.2204e-16

void exactTotalSum(Matrix y, Matrix h, float totalSum, float precision, Matrix x) {
	// y and h are vectors, totalSum and precision are scalars
	// x is the return vector and length is the length of y, h, and x
	
	// allocate hAlpha
	Matrix hAlpha;
	hAlpha.width = h.width;
	hAlpha.height = h.height;
	hAlpha.elements = (float*)malloc(hAlpha.width * hAlpha.height * sizeof(float));

	float totalSumMinus = totalSum - precision;
	thrust::host_vector<float> H_h(h.elements, h.elements + h.width*h.height);
	thrust::device_vector<float> D_h = H_h;
	thrust::detail::normal_iterator<thrust::device_ptr<float> > MinIt = thrust::min_element(D_h.begin(), D_h.end());
	float Min = *MinIt;
	float curAlpha = -Min + EPS;

	float stepAlpha, newAlpha, newSum;
	stepAlpha = (10 > fabs(curAlpha/10))? 10 : fabs(curAlpha/10);

	for(int j=0; j < 50; j++) {
		newAlpha = curAlpha + stepAlpha;
		newSum = 0;

		matPlusScaler(h, newAlpha, hAlpha);
		matDiv(y, hAlpha, x);

		thrust::host_vector<float> H_x(x.elements, x.elements + x.width * x.height);
		thrust::device_vector<float> D_x = H_x;
		newSum = thrust::reduce(D_x.begin(), D_x.end(), (float) 0, thrust::plus<float>());

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
