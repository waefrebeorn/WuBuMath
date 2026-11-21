#include "wubu_mitm_search.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Meet-in-the-middle braid search:
   Given target (dim*dim doubles) and gate set (ngates * dim*dim doubles),
   search words of length up to half_len*2 for closest match.
   Returns best Frobenius distance found. */

static double frob_dist(const double *A, const double *B, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) { double d = A[i]-B[i]; s += d*d; }
    return sqrt(s);
}

/* Apply a gate (matrix multiply) to current state */
static void apply_gate(const double *gate, double *state, size_t dim) {
    double tmp[64];
    memset(tmp, 0, dim*dim*sizeof(double));
    for (size_t i = 0; i < dim; i++)
        for (size_t k = 0; k < dim; k++)
            for (size_t j = 0; j < dim; j++)
                tmp[i*dim+j] += gate[i*dim+k] * state[k*dim+j];
    memcpy(state, tmp, dim*dim*sizeof(double));
}

double wubu_mitm_search(const double *target, const double *gate_set, int ngates, int half_len) {
    size_t dim = 2; /* 2x2 gates for simplicity */
    size_t mat_size = dim * dim;
    double best = 1e18;
    
    /* Enumerate left half: all words of length 0..half_len */
    /* Store resulting matrices and their "word" (as indices) */
    int max_words = 1;
    for (int i = 0; i < half_len; i++) max_words *= (ngates + 1);
    if (max_words > 10000) max_words = 10000;
    
    double *left_mats = malloc(max_words * mat_size * sizeof(double));
    int n_left = 0;
    
    /* identity */
    double ident[4] = {1,0,0,1};
    memcpy(&left_mats[0], ident, mat_size*sizeof(double));
    n_left = 1;
    
    /* BFS enumerate */
    for (int depth = 0; depth < half_len && n_left < max_words; depth++) {
        int cur_count = n_left;
        for (int w = 0; w < cur_count && n_left < max_words; w++) {
            for (int g = 0; g < ngates && n_left < max_words; g++) {
                memcpy(&left_mats[n_left*mat_size], &left_mats[w*mat_size], mat_size*sizeof(double));
                apply_gate(&gate_set[g*mat_size], &left_mats[n_left*mat_size], dim);
                n_left++;
            }
        }
    }
    
    /* For each left matrix, compute inverse (approx: transpose for unitary) and combine with target */
    /* Right half: search for gate sequence such that left * right ≈ target */
    /* Simplified: for each left matrix, compute residual = left^{-1} * target, then search right words */
    
    /* Enumerate right half similarly */
    double *right_mats = malloc(max_words * mat_size * sizeof(double));
    int n_right = 0;
    memcpy(&right_mats[0], ident, mat_size*sizeof(double));
    n_right = 1;
    for (int depth = 0; depth < half_len && n_right < max_words; depth++) {
        int cur_count = n_right;
        for (int w = 0; w < cur_count && n_right < max_words; w++) {
            for (int g = 0; g < ngates && n_right < max_words; g++) {
                memcpy(&right_mats[n_right*mat_size], &right_mats[w*mat_size], mat_size*sizeof(double));
                apply_gate(&gate_set[g*mat_size], &right_mats[n_right*mat_size], dim);
                n_right++;
            }
        }
    }
    
    /* Combine: find left_i, right_j minimizing ||left_i * right_j - target|| */
    for (int li = 0; li < n_left; li++) {
        for (int ri = 0; ri < n_right; ri++) {
            double combined[4];
            /* combined = left * right */
            for (size_t i = 0; i < dim; i++)
                for (size_t j = 0; j < dim; j++) {
                    combined[i*dim+j] = 0;
                    for (size_t k = 0; k < dim; k++)
                        combined[i*dim+j] += left_mats[li*mat_size+i*dim+k] * right_mats[ri*mat_size+k*dim+j];
                }
            double d = frob_dist(combined, target, mat_size);
            if (d < best) best = d;
        }
    }
    
    free(left_mats);
    free(right_mats);
    return best;
}
