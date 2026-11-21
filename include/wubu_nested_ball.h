/* wubu_nested_ball.h */
#ifndef WUBU_NESTED_BALL_H
#define WUBU_NESTED_BALL_H
#include <stddef.h>
/* Nested hyperbolic ball: scale point from level L to level 0 via Mobius scalar mult by phi^L.
   phi = golden ratio. */
void wubu_nested_scale(const double *x, int level, double *out, size_t dim);
#endif