/* wubu_pool.h */
#ifndef WUBU_POOL_H
#define WUBU_POOL_H
#include <stddef.h>
/* Simple bump allocator pool */
typedef struct { char *base; size_t size; size_t used; } wubu_pool;
void wubu_pool_init(wubu_pool *p, void *buf, size_t size);
void *wubu_pool_alloc(wubu_pool *p, size_t n);
void wubu_pool_reset(wubu_pool *p);
#endif