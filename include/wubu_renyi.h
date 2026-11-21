/* wubu_renyi.h */
#ifndef WUBU_RENYI_H
#define WUBU_RENYI_H
#include <stddef.h>
/* Renyi entropy of order alpha: H_alpha = (1/(1-alpha)) log(sum p^alpha) */
double wubu_renyi_entropy(const double *p, int n, double alpha);
#endif