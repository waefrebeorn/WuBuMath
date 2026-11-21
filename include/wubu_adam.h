/* wubu_adam.h */
#ifndef WUBU_ADAM_H
#define WUBU_ADAM_H
#include <stddef.h>
/* Adam optimizer step */
typedef struct { double *m; double *v; int t; } wubu_adam_state;
void wubu_adam_init(wubu_adam_state *s, size_t dim);
void wubu_adam_step(wubu_adam_state *s, double *theta, const double *grad, size_t dim, double lr, double b1, double b2, double eps);
void wubu_adam_free(wubu_adam_state *s);
#endif