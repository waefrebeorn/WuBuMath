/* wubu_curvature.h */
#ifndef WUBU_CURVATURE_H
#define WUBU_CURVATURE_H
#include <stddef.h>
/* Learnable curvature update: gradient step on K (clamped to (Kmin,Kmax)) */
void wubu_curvature_step(double *K, double grad, double lr, double Kmin, double Kmax);
#endif