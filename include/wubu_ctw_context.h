/* wubu_ctw_context.h -- Context-Tree Weighting context model for CABAC */
#ifndef WUBU_CTW_CONTEXT_H
#define WUBU_CTW_CONTEXT_H
#include <stddef.h>
/* CTW binary context tree, depth D. Probability estimate via weighted
   mixture of node statistics. */
typedef struct wubu_ctw_node {
    int count0, count1;
    struct wubu_ctw_node *child[2];
} wubu_ctw_node_t;

typedef struct {
    wubu_ctw_node_t *root;
    int depth;
} wubu_ctw_t;

wubu_ctw_t *wubu_ctw_create(int depth);
void wubu_ctw_free(wubu_ctw_t *c);
void wubu_ctw_update(wubu_ctw_t *c, const int *history, int bit);
double wubu_ctw_prob(wubu_ctw_t *c, const int *history, int bit);
#endif