/* wubu_rng.h */
#ifndef WUBU_RNG_H
#define WUBU_RNG_H
#include <stdint.h>
/* Xorshift64 RNG */
typedef struct { uint64_t state; } wubu_rng;
void wubu_rng_seed(wubu_rng *r, uint64_t seed);
uint64_t wubu_rng_next(wubu_rng *r);
double wubu_rng_uniform(wubu_rng *r);
double wubu_rng_normal(wubu_rng *r);
#endif