/* wubu_manifold_codec_v2.h */
#ifndef WUBU_MANIFOLD_CODEC_V2_H
#define WUBU_MANIFOLD_CODEC_V2_H
#include <stddef.h>
#include <stdint.h>
/* V2: VQ + residual correction for better lossless compression.
   Returns compressed size. */
int wubu_mc2_encode(const double *data, int n_vecs, size_t dim,
    int n_codebook, uint8_t *out_buf, size_t buf_size);
int wubu_mc2_decode(const uint8_t *buf, size_t buf_size,
    double *data, int *n_vecs, size_t dim);
#endif