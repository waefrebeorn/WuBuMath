#include "wubu_cabac_ctx_init.h"
#include <math.h>

double wubu_cabac_init_prob(int qp, int slice_type, int ctx_idx) {
    /* simplified: linear in QP, slice type shifts, ctx_idx mixes */
    double base = 0.5 - 0.01*(qp - 26);
    if (slice_type == 1) base += 0.05; /* P slice */
    if (slice_type == 2) base += 0.10; /* B slice */
    base += 0.001 * (ctx_idx % 17);
    if (base < 0.01) base = 0.01;
    if (base > 0.99) base = 0.99;
    return base;
}