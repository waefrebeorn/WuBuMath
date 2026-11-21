#include "wubu_horopca.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static void mean_point(const double *pts, int n, size_t dim, double *mean) {
    /* Euclidean mean (simplified; true HoroPCA uses Einstein midpoint) */
    memset(mean, 0, dim*sizeof(double));
    for (int i = 0; i < n; i++)
        for (size_t d = 0; d < dim; d++)
            mean[d] += pts[i*dim+d];
    for (size_t d = 0; d < dim; d++) mean[d] /= (n > 0 ? n : 1);
}

static void project_tangent(const double *x, const double *base, size_t dim, double *out) {
    /* Project x to tangent space at base: v = x - base (Euclidean approx) */
    for (size_t d = 0; d < dim; d++) out[d] = x[d] - base[d];
}

int main_horopca_unused(void); /* avoid unused warning */

void wubu_horopca(const double *pts, int n, size_t dim, double *components, int ncomp) {
    /* 1. Compute mean */
    double *mean = malloc(dim * sizeof(double));
    mean_point(pts, n, dim, mean);
    /* 2. Project all points to tangent space at mean */
    double *tangent = malloc((size_t)n * dim * sizeof(double));
    for (int i = 0; i < n; i++) {
        project_tangent(&pts[i*dim], mean, dim, &tangent[i*dim]);
    }
    /* 3. Compute covariance matrix (dim x dim) */
    double *cov = calloc(dim*dim, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (size_t a = 0; a < dim; a++)
            for (size_t b = 0; b < dim; b++)
                cov[a*dim+b] += tangent[i*dim+a] * tangent[i*dim+b];
    }
    for (size_t a = 0; a < dim; a++)
        for (size_t b = 0; b < dim; b++)
            cov[a*dim+b] /= (n > 1 ? n-1 : 1);
    /* 4. Power iteration for top ncomp eigenvectors */
    for (int c = 0; c < ncomp && c < (int)dim; c++) {
        double *v = &components[c*dim];
        /* Initialize with random-ish vector */
        for (size_t d = 0; d < dim; d++) v[d] = (d == (size_t)c) ? 1.0 : 0.1;
        /* Power iteration */
        for (int iter = 0; iter < 50; iter++) {
            double *nv = malloc(dim * sizeof(double));
            memset(nv, 0, dim*sizeof(double));
            for (size_t a = 0; a < dim; a++)
                for (size_t b = 0; b < dim; b++)
                    nv[a] += cov[a*dim+b] * v[b];
            /* Orthogonalize against previous components */
            for (int prev = 0; prev < c; prev++) {
                double dot = 0;
                for (size_t d = 0; d < dim; d++) dot += nv[d] * components[prev*dim+d];
                for (size_t d = 0; d < dim; d++) nv[d] -= dot * components[prev*dim+d];
            }
            /* Normalize */
            double norm = 0;
            for (size_t d = 0; d < dim; d++) norm += nv[d]*nv[d];
            norm = sqrt(norm);
            if (norm > 1e-12) for (size_t d = 0; d < dim; d++) v[d] = nv[d]/norm;
            free(nv);
        }
    }
    free(mean); free(tangent); free(cov);
}
