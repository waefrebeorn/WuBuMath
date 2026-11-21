#include "wubu_busemann_mlr.h"
#include <math.h>
#include <stdlib.h>
void wubu_busemann_mlr(const double *x, const double *a, const double *b, int ncls, size_t dim, double *probs) {
    double *logits = malloc(ncls*sizeof(double));
    double mx=-1e300;
    for (int k=0;k<ncls;k++){
        double s=b[k];
        for (size_t i=0;i<dim;i++) s += a[k*dim+i]*x[i];
        logits[k]=s; if (s>mx) mx=s;
    }
    double sum=0;
    for (int k=0;k<ncls;k++){ probs[k]=exp(logits[k]-mx); sum+=probs[k]; }
    for (int k=0;k<ncls;k++) probs[k]/=sum;
    free(logits);
}