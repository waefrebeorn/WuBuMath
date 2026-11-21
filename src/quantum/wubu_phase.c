#include "wubu_phase.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void wubu_phase(int k, int N, double *re, double *im) {
    double ang = 2.0 * M_PI * (double)k / (double)N;
    *re = cos(ang); *im = sin(ang);
}
