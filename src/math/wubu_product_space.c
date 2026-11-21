#include "wubu_product_space.h"
#include <string.h>
void wubu_product_pack(const double *e, size_t ed, const double *h, size_t hd, const double *s, size_t sd, double *out) {
    size_t off=0;
    if (e) { memcpy(out+off, e, ed*sizeof(double)); off+=ed; }
    if (h) { memcpy(out+off, h, hd*sizeof(double)); off+=hd; }
    if (s) { memcpy(out+off, s, sd*sizeof(double)); off+=sd; }
}