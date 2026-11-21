#include "wubu_hyp_vq.h"
#include "wubu_hyp_dist.h"
#include <math.h>
int wubu_hyp_vq(const double *x, const double *codebook, int ncode, size_t dim) {
    int best = 0; double bestd = 1e18;
    for (int c = 0; c < ncode; c++) {
        double d = wubu_poincare_dist(x, &codebook[c*dim], dim, 1.0);
        if (d < bestd) { bestd = d; best = c; }
    }
    return best;
}