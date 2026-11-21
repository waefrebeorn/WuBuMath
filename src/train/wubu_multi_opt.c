#include "wubu_multi_opt.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Mixed optimizer: Euclidean params use Adam, hyperbolic params use Riemannian SGD.
   Euclidean: standard Adam with bias correction.
   Hyperbolic: project gradient to tangent space at current point, then retract. */

typedef struct { double *m, *v; int t; } AdamState;
static AdamState *euclid_state = NULL;
static size_t euclid_dim = 0;

void wubu_multi_opt_step(double *euclid_params, const double *euclid_grad, size_t ne,
                         double *hyp_params, const double *hyp_grad, size_t nh, double lr) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    /* Lazy init Adam state */
    if (!euclid_state || euclid_dim != ne) {
        if (euclid_state) { free(euclid_state->m); free(euclid_state->v); free(euclid_state); }
        euclid_state = malloc(sizeof(AdamState));
        euclid_state->m = calloc(ne, sizeof(double));
        euclid_state->v = calloc(ne, sizeof(double));
        euclid_state->t = 0;
        euclid_dim = ne;
    }
    euclid_state->t++;
    /* Euclidean Adam */
    for (size_t i = 0; i < ne; i++) {
        euclid_state->m[i] = b1*euclid_state->m[i] + (1-b1)*euclid_grad[i];
        euclid_state->v[i] = b2*euclid_state->v[i] + (1-b2)*euclid_grad[i]*euclid_grad[i];
        double mh = euclid_state->m[i] / (1-pow(b1, euclid_state->t));
        double vh = euclid_state->v[i] / (1-pow(b2, euclid_state->t));
        euclid_params[i] -= lr * mh / (sqrt(vh) + eps);
    }
    /* Hyperbolic: Riemannian gradient = (1 - c||x||^2)/2 * (grad - <grad,x>x) projected to tangent */
    /* Simplified: project to tangent, then retract via exp map approx (scaling) */
    double c = 1.0;
    for (size_t i = 0; i < nh; i++) {
        double nq2 = 0;
        for (size_t j = 0; j < nh; j++) nq2 += hyp_params[j]*hyp_params[j];
        double lam = (1.0 - c*nq2) / 2.0;
        /* tangent projection: grad_tan = grad - <grad, x> * x / ||x||^2 */
        double gx = 0, xn2 = 0;
        for (size_t j = 0; j < nh; j++) { gx += hyp_grad[j]*hyp_params[j]; xn2 += hyp_params[j]*hyp_params[j]; }
        double scale = (xn2 > 1e-12) ? gx / xn2 : 0;
        double grad_tan = lam * (hyp_grad[i] - scale * hyp_params[i]);
        hyp_params[i] -= lr * grad_tan;
    }
    /* Project to ball */
    double r = 0;
    for (size_t i = 0; i < nh; i++) r += hyp_params[i]*hyp_params[i];
    r = sqrt(r);
    if (r >= 0.99) { double s = 0.99/r; for (size_t i = 0; i < nh; i++) hyp_params[i] *= s; }
}
