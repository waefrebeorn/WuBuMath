#include "wubu_ctw_context.h"
#include <stdlib.h>
#include <math.h>

wubu_ctw_t *wubu_ctw_create(int depth) {
    wubu_ctw_t *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->depth = depth;
    c->root = calloc(1, sizeof(wubu_ctw_node_t));
    return c;
}

static void free_node(wubu_ctw_node_t *n) {
    if (!n) return;
    free_node(n->child[0]);
    free_node(n->child[1]);
    free(n);
}

void wubu_ctw_free(wubu_ctw_t *c) {
    if (!c) return;
    free_node(c->root);
    free(c);
}

static wubu_ctw_node_t *get_child(wubu_ctw_node_t *n, int bit) {
    if (!n->child[bit]) n->child[bit] = calloc(1, sizeof(wubu_ctw_node_t));
    return n->child[bit];
}

void wubu_ctw_update(wubu_ctw_t *c, const int *history, int bit) {
    /* walk tree following history (most recent first), update counts */
    wubu_ctw_node_t *n = c->root;
    n->count0 += (bit == 0);
    n->count1 += (bit == 1);
    for (int d = 0; d < c->depth && d < c->depth; d++) {
        int h = history[d];
        n = get_child(n, h);
        if (!n) return;
        n->count0 += (bit == 0);
        n->count1 += (bit == 1);
    }
}

double wubu_ctw_prob(wubu_ctw_t *c, const int *history, int bit) {
    /* weighted mixture: P = sum over depths weighted by KT estimator */
    double w = 1.0;
    double p = 0.5;
    wubu_ctw_node_t *n = c->root;
    int cnt0 = n->count0, cnt1 = n->count1;
    /* KT estimate at root */
    double kt_root = (cnt0 + 0.5) / (cnt0 + cnt1 + 1.0);
    p = (bit == 0) ? kt_root : (1.0 - kt_root);
    double total_w = 1.0;
    for (int d = 0; d < c->depth; d++) {
        int h = history[d];
        if (!n->child[h]) break;
        n = n->child[h];
        int c0 = n->count0, c1 = n->count1;
        double kt = (c0 + 0.5) / (c0 + c1 + 1.0);
        double pi = (bit == 0) ? kt : (1.0 - kt);
        w *= 0.5; /* downweight deeper */
        p += w * pi;
        total_w += w;
    }
    return p / total_w;
}
