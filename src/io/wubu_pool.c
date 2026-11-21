#include "wubu_pool.h"
#include <string.h>
void wubu_pool_init(wubu_pool *p, void *buf, size_t size) {
    p->base = buf; p->size = size; p->used = 0;
}
void *wubu_pool_alloc(wubu_pool *p, size_t n) {
    if (p->used + n > p->size) return NULL;
    void *ptr = p->base + p->used;
    p->used += n;
    return ptr;
}
void wubu_pool_reset(wubu_pool *p) { p->used = 0; }