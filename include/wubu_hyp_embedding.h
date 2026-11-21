/* wubu_hyp_embedding.h */
#ifndef WUBU_HYP_EMBEDDING_H
#define WUBU_HYP_EMBEDDING_H
#include <stddef.h>
/* Token embedding lookup: out = embedding[token] */
void wubu_hyp_embedding(const double *table, int token, size_t dim, double *out);
#endif