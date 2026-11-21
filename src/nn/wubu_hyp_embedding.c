#include "wubu_hyp_embedding.h"
#include <string.h>
void wubu_hyp_embedding(const double *table, int token, size_t dim, double *out) {
    memcpy(out, &table[token*dim], dim*sizeof(double));
}