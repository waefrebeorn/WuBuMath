#include "wubu_grad_clip.h"
#include <math.h>
double wubu_grad_clip_norm(double *grad, size_t n, double max_norm) {
    double norm = 0;
    for (size_t i = 0; i < n; i++) norm += grad[i]*grad[i];
    norm = sqrt(norm);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (size_t i = 0; i < n; i++) grad[i] *= scale;
    }
    return norm;
}
void wubu_grad_clip_value(double *grad, size_t n, double max_val) {
    for (size_t i = 0; i < n; i++) {
        if (grad[i] > max_val) grad[i] = max_val;
        if (grad[i] < -max_val) grad[i] = -max_val;
    }
}