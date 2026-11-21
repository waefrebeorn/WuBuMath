#include "wubu_gyro.h"
#include <math.h>
#include <string.h>
static double dot(const double *a, const double *b, size_t dim) {
    double s = 0; for (size_t i = 0; i < dim; i++) s += a[i]*b[i]; return s;
}
static double norm2(const double *x, size_t dim) { return dot(x,x,dim); }
void wubu_gyro_add(const double *x, const double *y, double c, double *out, size_t dim) {
    double nx2 = norm2(x,dim), ny2 = norm2(y,dim);
    double denom = 1.0 + 2.0*c*dot(x,y,dim) + c*c*nx2*ny2;
    for (size_t i = 0; i < dim; i++) {
        out[i] = ((1.0 + 2.0*c*dot(x,y,dim) + c*ny2)*x[i] + (1.0 - c*nx2)*y[i]) / denom;
    }
}
double wubu_gyro_norm(const double *x, size_t dim) { return sqrt(norm2(x,dim)); }
void wubu_gyro_scalar_mul(double t, const double *x, double c, double *out, size_t dim) {
    double nx = wubu_gyro_norm(x, dim);
    if (nx < 1e-12) { memset(out, 0, dim*sizeof(double)); return; }
    double scale = tanh(t*atanh(sqrt(c)*nx)) / (sqrt(c)*nx);
    for (size_t i = 0; i < dim; i++) out[i] = scale * x[i];
}