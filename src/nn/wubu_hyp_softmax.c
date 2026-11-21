#include "wubu_hyp_softmax.h"
#include <math.h>
void wubu_hyp_softmax(const double *in, int n, double *out) {
    double maxv = in[0];
    for (int i = 1; i < n; i++) if (in[i] > maxv) maxv = in[i];
    double s = 0;
    for (int i = 0; i < n; i++) { out[i] = exp(in[i]-maxv); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}
double wubu_hyp_cross_entropy(const double *probs, int target, int n) {
    (void)n;
    double p = probs[target] > 1e-12 ? probs[target] : 1e-12;
    return -log(p);
}