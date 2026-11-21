/* wubu_gyro.h */
#ifndef WUBU_GYRO_H
#define WUBU_GYRO_H
#include <stddef.h>
/* Gyrovector operations on Poincare ball */
void wubu_gyro_add(const double *x, const double *y, double c, double *out, size_t dim);
double wubu_gyro_norm(const double *x, size_t dim);
void wubu_gyro_scalar_mul(double t, const double *x, double c, double *out, size_t dim);
#endif