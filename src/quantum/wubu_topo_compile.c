#include "wubu_topo_compile.h"
#include <math.h>
double wubu_topo_leakage(const double *target, const double *braid_result) {
    /* Frobenius distance between target and braid result (complex) */
    double s = 0;
    for (int i = 0; i < 32; i++) { double d = target[i]-braid_result[i]; s += d*d; }
    return sqrt(s);
}