#include "wubu_manifold_codec.h"
#include "wubu_cabac.h"
#include "wubu_hyp_vq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_manifold_encode(const double *data, int n_vecs, size_t dim,
    int n_codebook, uint8_t *out_buf, size_t buf_size) {
    if (n_codebook > n_vecs) n_codebook = n_vecs;
    /* Build codebook from first n_codebook vectors */
    double *codebook = malloc((size_t)n_codebook * dim * sizeof(double));
    for (int c = 0; c < n_codebook; c++)
        memcpy(&codebook[c*dim], &data[c*dim], dim*sizeof(double));
    /* Quantize */
    int *indices = malloc((size_t)n_vecs * sizeof(int));
    for (int i = 0; i < n_vecs; i++)
        indices[i] = wubu_hyp_vq(&data[i*dim], codebook, n_codebook, dim);
    /* Write header: n_vecs, n_codebook, dim */
    size_t off = 0;
    memcpy(out_buf + off, &n_vecs, sizeof(int)); off += sizeof(int);
    memcpy(out_buf + off, &n_codebook, sizeof(int)); off += sizeof(int);
    size_t d = dim;
    memcpy(out_buf + off, &d, sizeof(size_t)); off += sizeof(size_t);
    /* Write codebook */
    memcpy(out_buf + off, codebook, (size_t)n_codebook * dim * sizeof(double));
    off += (size_t)n_codebook * dim * sizeof(double);
    /* CABAC encode indices into remaining buffer */
    WuBuCabacEnc enc;
    wubu_cabac_start(&enc, out_buf + off, buf_size - off);
    CabacContext ctx;
    wubu_cabac_init_contexts(&ctx, 1, 0, 0);
    for (int i = 0; i < n_vecs; i++) {
        for (int b = 0; b < indices[i]; b++) wubu_cabac_encode_bin(&enc, &ctx, 1);
        wubu_cabac_encode_bin(&enc, &ctx, 0);
    }
    long nbytes = wubu_cabac_finish(&enc);
    off += (size_t)nbytes;
    free(codebook); free(indices);
    return (int)off;
}

int wubu_manifold_decode(const uint8_t *buf, size_t buf_size,
    double *data, int *n_vecs, size_t dim) {
    (void)buf_size;
    size_t off = 0;
    int nv, nc;
    size_t d;
    memcpy(&nv, buf + off, sizeof(int)); off += sizeof(int);
    memcpy(&nc, buf + off, sizeof(int)); off += sizeof(int);
    memcpy(&d, buf + off, sizeof(size_t)); off += sizeof(size_t);
    if (dim != 0 && d != dim) return -1;
    size_t dd = (dim != 0) ? dim : d;
    /* Read codebook */
    double *codebook = malloc((size_t)nc * dd * sizeof(double));
    memcpy(codebook, buf + off, (size_t)nc * dd * sizeof(double));
    off += (size_t)nc * dd * sizeof(double);
    /* Decode CABAC indices */
    WuBuCabacDec dec;
    wubu_cabac_decoder_start(&dec, buf + off, buf_size - off);
    CabacContext ctx;
    wubu_cabac_init_contexts(&ctx, 1, 0, 0);
    for (int i = 0; i < nv; i++) {
        int idx = 0;
        while (wubu_cabac_decode_bin(&dec, &ctx) == 1) idx++;
        /* lookup codebook */
        memcpy(&data[i*dd], &codebook[idx*dd], dd*sizeof(double));
    }
    *n_vecs = nv;
    free(codebook);
    return 0;
}
