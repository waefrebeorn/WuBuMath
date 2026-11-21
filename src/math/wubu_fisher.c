#include "wubu_fisher.h"
#include <math.h>
void wubu_fisher_diag(const double *grads, int n, size_t dim, double *F) {
    for (size_t d = 0; d < dim; d++) {
        double s = 0;
        for (int i = 0; i < n; i++) { double g = grads[i*dim+d]; s += g*g; }
        F[d] = s / (n > 0 ? n : 1);
    }
}
void wubu_natural_grad_step(double *theta, const double *grad, const double *F, size_t dim, double lr) {
    for (size_t d = 0; d < dim; d++) {
        double f = F[d] > 1e-12 ? F[d] : 1e-12;
        theta[d] -= lr * grad[d] / f;
    }
}