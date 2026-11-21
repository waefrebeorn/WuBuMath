#include "wubu_nested_transformer.h"
#include "wubu_so4.h"
#include "wubu_nested_ball.h"
#include <string.h>
#include <stdlib.h>

void wubu_nested_transformer(const double *x, int levels, const double *qL, const double *qR, double *out, size_t dim) {
    /* copy x to out */
    for (size_t i = 0; i < dim; i++) out[i] = x[i];
    for (int lvl = 0; lvl < levels; lvl++) {
        /* scale to level */
        double *scaled = malloc(dim*sizeof(double));
        wubu_nested_scale(out, lvl, scaled, dim);
        /* rotate */
        double x4[4] = {scaled[0], scaled[1], scaled[2], (dim>3)?scaled[3]:0};
        double o4[4];
        wubu_so4_rotate(qL, qR, x4, o4);
        for (size_t i = 0; i < dim && i < 4; i++) out[i] = o4[i];
        free(scaled);
    }
}