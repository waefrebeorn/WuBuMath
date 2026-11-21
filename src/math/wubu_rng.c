#define _GNU_SOURCE
#include "wubu_rng.h"
#include <math.h>
void wubu_rng_seed(wubu_rng *r, uint64_t seed) { r->state = seed ? seed : 1; }
uint64_t wubu_rng_next(wubu_rng *r) {
    uint64_t x = r->state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    r->state = x;
    return x;
}
double wubu_rng_uniform(wubu_rng *r) {
    return (double)wubu_rng_next(r) / (double)UINT64_MAX;
}
double wubu_rng_normal(wubu_rng *r) {
    /* Box-Muller */
    double u1 = wubu_rng_uniform(r);
    double u2 = wubu_rng_uniform(r);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0*log(u1)) * cos(2*M_PI*u2);
}