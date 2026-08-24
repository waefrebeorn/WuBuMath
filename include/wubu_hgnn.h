/* GAP-B015: hyperbolic graph convolution layer */
#ifndef WUBU_HGNN_H
#define WUBU_HGNN_H
#ifdef __cplusplus
extern "C" {
#endif
/* x: [N,D] on-ball. adj in CSR format. out: [N,D]. Returns 0 on success. */
int wubu_hgnn_layer(const float* x,const int* adj_idx,const int* adj_ptr,
                    const float* edge_weight,const float* W,
                    int N,int D,float c,float* out);
#ifdef __cplusplus
}
#endif
#endif
