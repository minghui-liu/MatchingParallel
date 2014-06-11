#include <stdio.h>
#include "k_utils.cu"

#define BLOCK_SIZE 32

void soft2hard(Matrix d_soft, int numberOfMatches, Matrix d_hard) {
	zeros(hard);
	for (int i=0; i < numberOfMatches; i++) {
		double maxSoft = maxOfMatrix(soft);
		dummy = maxOfMatrix(maxSoft);
		r = indexOfElement(maxSoft);
		Matrix soft_r;
		soft_r.height = 1;
		soft_r.width = d_soft.width;
		getRow(soft, soft_r, r);
		val = maxOfMatrix(soft_r);
		c = indexOfElement(soft_r);
		if (val < 0) { 
			return;
		}
		Matrix hard_rc;
		hard_rc.height = 1;
		hard_rc.width = 1;
		ones(hard_rc);
		negInf(soft_r);
		Matrix soft_c;
		soft_c.height = d_soft.height;
		sget_c.width = 1;
		getCol(soft, soft_c, c);
		negInf(soft_c);
	}
}


/*******************************************************************************
 function [X,Z] = hypergraphMatching (Y, numberOfMatches)

 Optimal soft hyergraph matching.

 Algorithm due to R. Zass and A. Shashua.,
 'Probabilistic Graph and Hypergraph Matching.',
 Computer Vision and Pattern Recognition (CVPR) Anchorage, Alaska, June 2008.

 Y - Marginalization of the hyperedge-to-hyperedge correspondences matrix.
 numberOfMatches - number of matches required.

 X [Output] - an n1 by n2 matrix with the hard matching results.
             The i,j entry is one iff the i-th feature of the first object
             match the j-th feature of the second object. Zero otherwise.
 Z [Output] - an n1 by n2 matrix with the soft matching results.
             The i,j entry is the probablity that the i-th feature of the
             first object match the j-th feature of the second object.

 See also:
 - graphMatching() as an example on how to use this for graphs with a
  specific similarity function.

 Author: Ron Zass, zass@cs.huji.ac.il, www.cs.huji.ac.il/~zass
*******************************************************************************/
void hypergraphMatching(Matrix d_Y, int numberOfMatches, Matrix X, Matrix Z) {
	Matrix maxRowSum, maxColSum;
	maxRowSum.height = d_Y.height;
	maxRowSum.width = 1;
	maxColSum.height = 1;
	maxColSum.width = d_Y.width;
	
	Z = nearestDSmax_RE(Y, maxRowSum, maxColSum, numberOfMatches);
	X = soft2hard(Z, numberOfMatches);
}
