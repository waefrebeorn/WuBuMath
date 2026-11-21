#include "wubu_renyi.h"
#include <math.h>
double wubu_renyi_entropy(const double *p, int n, double alpha) {
    double s = 0;
    for (int i = 0; i < n; i++) s += pow(p[i], alpha);
    if (fabs(alpha - 1.0) < 1e-9) {
        /* Shannon limit */
        double h = 0;
        for (int i = 0; i < n; i++) if (p[i] > 0) h -= p[i]*log(p[i]);
        return h;
    }
    return log(s) / (1.0 - alpha);
}