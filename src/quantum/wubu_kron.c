#include "wubu_kron.h"

void wubu_kron2(const double *A, const double *B, double *out) {
    /* A,B complex 2x2 (8 doubles). out complex 4x4 (32 doubles). */
    for (int i=0;i<2;i++) for (int j=0;j<2;j++)
      for (int k=0;k<2;k++) for (int l=0;l<2;l++) {
        int r = i*2+k, c = j*2+l;
        /* multiply complex: A[i,j] * B[k,l] */
        double are=A[(i*2+j)*2], aim=A[(i*2+j)*2+1];
        double bre=B[(k*2+l)*2], bim=B[(k*2+l)*2+1];
        out[(r*4+c)*2]   = are*bre - aim*bim;
        out[(r*4+c)*2+1] = are*bim + aim*bre;
      }
}