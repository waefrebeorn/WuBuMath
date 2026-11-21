#include "wubu_hyp_batchnorm.h"
#include <math.h>
#include <string.h>
void wubu_hyp_batchnorm(const double *in, double *out, int n, size_t dim, double eps) {
    for (size_t d = 0; d < dim; d++) {
        double mean = 0;
        for (int i = 0; i < n; i++) mean += in[i*dim+d];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) { double diff = in[i*dim+d]-mean; var += diff*diff; }
        var /= n;
        double std = sqrt(var + eps);
        for (int i = 0; i < n; i++) out[i*dim+d] = (in[i*dim+d]-mean)/std;
    }
}