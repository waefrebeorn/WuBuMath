#include "wubu_moe_router.h"
#include "wubu_hyp_dist.h"
#include <math.h>
#include <stdlib.h>

void wubu_moe_route(const double *x, const double *protos, int nexperts, size_t dim, int k, int *topk, double *scores) {
    for (int e = 0; e < nexperts; e++) {
        double d = wubu_poincare_dist(x, &protos[e*dim], dim, 1.0);
        scores[e] = -d;
    }
    /* softmax */
    double mx = scores[0];
    for (int e = 1; e < nexperts; e++) if (scores[e] > mx) mx = scores[e];
    double sum = 0;
    for (int e = 0; e < nexperts; e++) { scores[e] = exp(scores[e]-mx); sum += scores[e]; }
    for (int e = 0; e < nexperts; e++) scores[e] /= sum;
    /* top-k by score */
    for (int i = 0; i < k; i++) {
        int best = -1; double bestv = -1;
        for (int e = 0; e < nexperts; e++) {
            int used = 0;
            for (int j = 0; j < i; j++) if (topk[j] == e) used = 1;
            if (!used && scores[e] > bestv) { bestv = scores[e]; best = e; }
        }
        topk[i] = best;
    }
}