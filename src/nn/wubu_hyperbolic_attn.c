#include "wubu_hyperbolic_attn.h"
#include <math.h>
#include <stdlib.h>

static double pdist(const double *a, const double *b, size_t dim) {
    double s=0; for(size_t i=0;i<dim;i++){double d=a[i]-b[i];s+=d*d;}
    return sqrt(s);
}

void wubu_hyp_attn_weights(const double *q, const double **keys, int nkeys, size_t dim, double *weights) {
    double *scores = malloc(nkeys*sizeof(double));
    double mx = -1e300;
    for (int i=0;i<nkeys;i++){
        double d = pdist(q, keys[i], dim);
        scores[i] = -d; /* closer = higher */
        if (scores[i]>mx) mx=scores[i];
    }
    double sum=0;
    for (int i=0;i<nkeys;i++){ weights[i]=exp(scores[i]-mx); sum+=weights[i]; }
    for (int i=0;i<nkeys;i++) weights[i]/=sum;
    free(scores);
}