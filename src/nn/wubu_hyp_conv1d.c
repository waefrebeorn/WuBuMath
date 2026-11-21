#include "wubu_hyp_conv1d.h"
#include <string.h>
void wubu_hyp_conv1d(const double *in, int n, const double *kernel, int k, double *out) {
    for (int t = 0; t < n; t++) {
        double s = 0;
        for (int j = 0; j < k; j++) {
            int idx = t + j - k/2;
            if (idx >= 0 && idx < n) s += in[idx]*kernel[j];
        }
        out[t] = s;
    }
}