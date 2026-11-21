#include "wubu_rd_quant.h"
#include <math.h>

int wubu_rd_quantize(double w, const double *levels, int nlevels,
                     const double *rates, double eta, double lambda) {
    int best = 0;
    double bestcost = 1e300;
    for (int k = 0; k < nlevels; k++) {
        double dist = w - levels[k];
        double cost = eta * dist * dist + lambda * rates[k];
        if (cost < bestcost) { bestcost = cost; best = k; }
    }
    return best;
}
