#include "wubu_manifold_codec_v2.h"
#include "wubu_cabac.h"
#include "wubu_hyp_vq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdio.h>

int wubu_mc2_encode(const double *data, int n_vecs, size_t dim,
    int n_codebook, uint8_t *out_buf, size_t buf_size) {
    if (n_codebook > n_vecs) n_codebook = n_vecs;
    double *codebook = malloc((size_t)n_codebook * dim * sizeof(double));
    for (int c = 0; c < n_codebook; c++)
        memcpy(&codebook[c*dim], &data[c*dim], dim*sizeof(double));
    /* Compute indices + residuals */
    int *indices = malloc((size_t)n_vecs * sizeof(int));
    double *residuals = calloc((size_t)n_vecs * dim, sizeof(double));
    for (int i = 0; i < n_vecs; i++) {
        indices[i] = wubu_hyp_vq(&data[i*dim], codebook, n_codebook, dim);
        for (size_t d = 0; d < dim; d++)
            residuals[i*dim+d] = data[i*dim+d] - codebook[indices[i]*dim+d];
    }
    /* Write header */
    size_t off = 0;
    memcpy(out_buf + off, &n_vecs, sizeof(int)); off += sizeof(int);
    memcpy(out_buf + off, &n_codebook, sizeof(int)); off += sizeof(int);
    size_t dd = dim;
    memcpy(out_buf + off, &dd, sizeof(size_t)); off += sizeof(size_t);
    memcpy(out_buf + off, codebook, (size_t)n_codebook * dim * sizeof(double));
    off += (size_t)n_codebook * dim * sizeof(double);
    /* CABAC: encode indices (unary) then residuals (as 16-bit fixed-point) */
    WuBuCabacEnc enc;
    wubu_cabac_start(&enc, out_buf + off, buf_size - off);
    CabacContext ctx;
    wubu_cabac_init_contexts(&ctx, 1, 0, 0);
    for (int i = 0; i < n_vecs; i++) {
        for (int b = 0; b < indices[i]; b++) wubu_cabac_encode_bin(&enc, &ctx, 1);
        wubu_cabac_encode_bin(&enc, &ctx, 0);
        for (size_t d = 0; d < dim; d++) {
            int q = (int)(residuals[i*dim+d] * 1000.0);
            wubu_cabac_encode_bin(&enc, &ctx, q < 0 ? 1 : 0);
            int mag = q < 0 ? -q : q;
            for (int b = 15; b >= 0; b--) wubu_cabac_encode_bin(&enc, &ctx, (mag>>b)&1);
        }
    }
    long nbytes = wubu_cabac_finish(&enc);
    off += (size_t)nbytes;
    free(codebook); free(indices); free(residuals);
    return (int)off;
}

int wubu_mc2_decode(const uint8_t *buf, size_t buf_size,
    double *data, int *n_vecs, size_t dim) {
    (void)buf_size;
    size_t off = 0;
    int nv, nc;
    size_t dd;
    memcpy(&nv, buf + off, sizeof(int)); off += sizeof(int);
    memcpy(&nc, buf + off, sizeof(int)); off += sizeof(int);
    memcpy(&dd, buf + off, sizeof(size_t)); off += sizeof(size_t);
    if (dim != 0 && dd != dim) return -1;
    size_t d = (dim != 0) ? dim : dd;
    double *codebook = malloc((size_t)nc * d * sizeof(double));
    memcpy(codebook, buf + off, (size_t)nc * d * sizeof(double));
    off += (size_t)nc * d * sizeof(double);
    /* Decode CABAC */
    WuBuCabacDec dec;
    wubu_cabac_decoder_start(&dec, buf + off, buf_size - off);
    CabacContext ctx;
    wubu_cabac_init_contexts(&ctx, 1, 0, 0);
    for (int i = 0; i < nv; i++) {
        int idx = 0;
        while (wubu_cabac_decode_bin(&dec, &ctx) == 1) idx++;
        /* decode residuals */
        for (size_t dd2 = 0; dd2 < d; dd2++) {
            int sign = wubu_cabac_decode_bin(&dec, &ctx);
            int mag = 0;
            for (int b = 15; b >= 0; b--) mag = (mag << 1) | wubu_cabac_decode_bin(&dec, &ctx);
            double resid = (double)(sign ? -mag : mag) / 1000.0;
            data[i*d+dd2] = codebook[idx*d+dd2] + resid;
        }
    }
    *n_vecs = nv;
    free(codebook);
    return 0;
}