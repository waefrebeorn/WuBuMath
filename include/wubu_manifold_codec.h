/* wubu_manifold_codec.h */
#ifndef WUBU_MANIFOLD_CODEC_H
#define WUBU_MANIFOLD_CODEC_H
#include <stddef.h>
#include <stdint.h>
/* Manifold codec: embed vectors into Poincare ball, quantize, compress with CABAC.
   Returns compressed size in bytes, or -1 on error. */
int wubu_manifold_encode(const double *data, int n_vecs, size_t dim,
    int n_codebook, uint8_t *out_buf, size_t buf_size);
/* Decode: reconstruct vectors from compressed buffer. Returns 0 on success. */
int wubu_manifold_decode(const uint8_t *buf, size_t buf_size,
    double *data, int *n_vecs, size_t dim);
#endif